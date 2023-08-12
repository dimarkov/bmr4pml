import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu
import tensorflow_probability.substrates.jax.stats as stats
import equinox as eqx

from tqdm import trange
from multipledispatch import dispatch
from numpyro import handlers
from numpyro.infer import SVI, NUTS, MCMC, Predictive, log_likelihood
from jax import random as jr
from jax import nn, devices, device_put, lax

from .models import BayesRegression, SVIRegression, BMRRegression

gpus = devices('gpu')
if len(gpus) < 2:
    dev2 = devices('cpu')[0]
else:
    dev2 = gpus[1]

def mean_ll(log_like):

    S = log_like.shape[0]
    ll = nn.logsumexp(log_like - jnp.log(S), axis=0) 
    lpd = nn.logsumexp(log_like.sum(-1) - jnp.log(S))

    return ll, lpd 

def tests(model, sample, x, y, **kwargs):
    ll = log_likelihood(model, sample, x, batch_ndims=1, y=y, **kwargs)

    ll, lpd = mean_ll(ll['obs'])

    probs = sample['probs'].mean(0)
    pred_labels = probs.argmax(-1)

    hit = pred_labels == y
    acc = hit.mean(-1)

    ece = stats.expected_calibration_error_quantiles(
        hit, jnp.log(probs.max(-1)), num_buckets=20
    )

    return {
        'acc': acc,
        'ece': ece[0],
        'nll': -ll.mean(),
        'lpd': lpd.item() / len(ll),
    }

def run(key, svi, state, params, num_iters, *args, progress_bar=False, **kwargs):
    key, _key = jr.split(key)
    results = svi.run(
        _key, 
        num_iters,
        *args,
        progress_bar=progress_bar,
        init_state=state,
        init_params=params,
        **kwargs
    )

    state = results.state
    params = results.params
    losses = results.losses
    return key, state, params, losses

@dispatch(BayesRegression, dict, dict)
def fit(regression, train_ds, opts):

    model = regression.model

    num_warmup = opts['num_warmup']
    num_samples = opts['num_samples']
    num_chains = opts['num_chains']
    progress_bar = opts['progress_bar']
    summary = opts['summary']
    key = opts['key']

    key, _key = jr.split(key)

    nuts_kernel = NUTS(model)
    mcmc = MCMC(
        nuts_kernel, 
        num_warmup=num_warmup, 
        num_samples=num_samples, 
        num_chains=num_chains,
        chain_method='vectorized',
        progress_bar=progress_bar
    )
    
    mcmc.run(_key, train_ds['input'], y=train_ds['target'], **opts['model_kwargs'])

    if summary:
        mcmc.print_summary()

    samples = mcmc.get_samples(group_by_chain=False)
    return samples, mcmc

@dispatch(SVIRegression, dict, dict, dict)
def fit_and_test(regression, train_ds, test_ds, opts):
    model = regression.model
    guide = regression.guide
    optimizer = regression.optim
    loss = regression.loss

    num_epochs = opts['num_epochs']
    num_iters = opts['num_iters']
    num_samples = opts['num_samples']
    key = opts['key']
    model_kwargs = opts['model_kwargs']
    test_kwargs = model_kwargs | {'batch_size': None, 'inference': True}
    
    svi = SVI(model, guide, optimizer, loss)

    key, _key = jr.split(key)
    x = train_ds['image']
    y = train_ds['label']
    x_test = test_ds['image']
    y_test = test_ds['label']
    cpu_x_test = device_put(x_test, dev2)
    cpu_y_test = device_put(y_test, dev2)
    state = svi.init(_key, x, y=y, **model_kwargs)
    params = None
    
    losses = []
    results = []
    with trange(1, num_epochs + 1) as t:
        for i in t:
            key, _key = jr.split(key)
            key, state, params, loss = run(
                key, svi, state, params, num_iters, x, y=y, **model_kwargs
            )
            
            losses.append(loss)
            avg_loss = np.nanmean(loss)
            t.set_postfix_str(
                "init loss: {:.4f}, avg. loss [epoch {}]: {:.4f}".format(
                    losses[0][0], i, avg_loss
                ),
                refresh=False,
            )
            key, _key = jr.split(key)
            cpu_params = device_put(params, dev2)

            pred = Predictive(guide, params=cpu_params, num_samples=num_samples)
            samples = pred(_key, cpu_x_test, y=cpu_y_test, **test_kwargs)
            pred = Predictive(model, posterior_samples=samples, return_sites=['probs'])
            samples = pred(_key, cpu_x_test, y=cpu_y_test, **test_kwargs) | samples
            results.append( tests(model, samples, cpu_x_test, cpu_y_test, **test_kwargs) )

    results = jtu.tree_map(lambda *args: np.stack(list(args)), *tuple(results))
    results['losses'] = np.stack(losses)

    pred = Predictive(model, posterior_samples=device_put(samples, dev2))
    samples = pred(_key, cpu_x_test, y=cpu_y_test, **test_kwargs)

    results['samples'] = samples
    
    return results

def pruned_fraction(gammas, params, delta=1e-15, **kwargs):
    count = 0
    size = 0
    for l in range(len(gammas) - 1):
        name = f'layer{l}.weight'
        gamma = gammas[name]
        mu = params[f'{name}.loc']
        prune = jnp.broadcast_to(gamma < delta, mu.shape)

        count += prune.sum()
        size  += prune.shape[0] * prune.shape[1]
        active_neurons = ~jnp.all(prune, -1)
        
    mu = params[f'layer{l+1}.weight.loc']
    count += mu.shape[0] * (1 - active_neurons).sum()
    size += mu.shape[0] * ( mu.shape[1] - 1)

    return count/size

@dispatch(BMRRegression, dict, dict, dict)
def fit_and_test(regression, train_ds, test_ds, opts):
    model = regression.model
    guide = regression.guide

    pruning = regression.pruning
    optimizer = regression.optim
    loss = regression.loss
    
    num_epochs = opts['num_epochs']
    num_iters = opts['num_iters']
    warmup_iters = opts.pop('warmup_iters', None)
    num_samples = opts['num_samples']
    key = opts['key']
    model_kwargs = opts['model_kwargs']
    test_kwargs = model_kwargs | {'batch_size': None, 'inference': True}

    svi = SVI(model, guide, optimizer, loss)

    key, _key = jr.split(key)
    x = train_ds['image']
    y = train_ds['label']
    x_test = test_ds['image']
    y_test = test_ds['label']
    cpu_x_test = device_put(x_test, dev2)
    cpu_y_test = device_put(y_test, dev2)
    state = opts.pop('state', svi.init(_key, x, y=y, **model_kwargs))
    params = opts.pop('params', None)
    pruning_kwargs = opts.pop('pruning_kwargs', {})

    key, _key = jr.split(key)
    if warmup_iters is not None:
        key, state, params, loss = run(
            key, svi, state, params, warmup_iters, x, y=y, **model_kwargs
        )
        return state, params

    if params is not None:
        gammas = pruning(_key, params, **pruning_kwargs)

    losses = []
    results = []
    with trange(1, num_epochs + 1) as t:
        for i in t:
            key, _key = jr.split(key)
            key, state, params, loss = run(
                key, svi, state, params, num_iters, x, y=y, **model_kwargs
            )
            
            losses.append(loss)
            avg_loss = np.nanmean(loss)
            t.set_postfix_str(
                "init loss: {:.4f}, avg. loss [epoch {}]: {:.4f}".format(
                    losses[0][0], i, avg_loss
                ),
                refresh=False,
            )
            key, _key = jr.split(key)

            zip = pruned_fraction(gammas, params, **pruning_kwargs)

            cpu_params = lax.stop_gradient(device_put(params, dev2))
            cpu_gammas = lax.stop_gradient(device_put(gammas, dev2))

            pred = Predictive(guide, params=cpu_params, num_samples=num_samples)
            cpu_samples = pred(_key, cpu_x_test, y=cpu_y_test, **test_kwargs)
            pred = Predictive(model, posterior_samples=cpu_samples, return_sites=['probs'])
            cpu_samples = pred(_key, cpu_x_test, y=cpu_y_test, gammas=cpu_gammas, **test_kwargs) | cpu_samples
            out = tests(model, cpu_samples, cpu_x_test, cpu_y_test, gammas=cpu_gammas, **test_kwargs)
            out['zip'] = zip
            results.append( out )

            if i < num_epochs:
                gammas = pruning(_key, params, **pruning_kwargs)


    results = jtu.tree_map(lambda *args: np.stack(list(args)), *tuple(results))
    results['losses'] = np.stack(losses)

    pred = Predictive(model, posterior_samples=cpu_samples)
    samples = pred(_key, cpu_x_test, y=cpu_y_test, gammas=cpu_gammas, **test_kwargs)

    results['samples'] = samples
    return results