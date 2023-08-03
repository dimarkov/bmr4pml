import tensorflow_probability.substrates.jax.stats as stats
import tensorflow as tf
import jax.numpy as jnp
from numpyro.infer import log_likelihood
from numpyro.infer import Predictive
from jax import nn, random, jit

def mean_ll(log_like):

    S = log_like.shape[0]
    ll = nn.logsumexp(log_like - jnp.log(S), axis=0) 
    lpd = nn.logsumexp(log_like.sum(-1) - jnp.log(S))

    return ll, lpd 

def test_smpl(rng_key, model, sample, labels):
    ll = log_likelihood(model, sample, parallel=True, obs=labels)

    ll, lpd = mean_ll(ll['obs'])

    probs = sample['probs'].mean(0)
    pred_labels = probs.argmax(-1)
    logits = jnp.log(probs)

    hit = pred_labels == labels
    acc = hit.mean(-1)

    return {
        'acc': acc,
        'nll': -ll.mean(),
        'lpd': lpd.item(),
    }

def compression(L, samples, cuttoff=1.):
    compressed_smpl = samples.copy()
    nonzero = 0
    count = 0
    for l in range(L):
        mu = samples[f'layer{l}.weight'].mean(0)
        std = samples[f'layer{l}.weight'].std(0)
        z = mu/std
        zeros = jnp.abs(z) > cuttoff
        nonzero += zeros.sum()
        count += z.shape[0] * z.shape[1]
        compressed_smpl[f'layer{l}.weight'] = samples[f'layer{l}.weight'] * zeros
        compressed_smpl[f'layer{l}.weight_base'] = samples[f'layer{l}.weight_base'] * zeros

    return compressed_smpl, 1 - nonzero/count

def compression2(key, L, params, cuttoff=1.):
    compressed_smpl = {}
    nonzero = 0
    count = 0
    for l in range(L):
        mu = params[f'layer{l}.weight.loc'].mean(0)
        std = jnp.sqrt( jnp.square(params[f'layer{l}.weight.scale']) ).mean(0)
        z = mu/std
        zeros = jnp.abs(z) > cuttoff
        nonzero += zeros.sum()
        count += z.shape[0] * z.shape[1]
        compressed_smpl[f'layer{l}.weight_base'] =jnp.where(zeros, mu + std * random.normal(key, (100,) + mu.shape), 0.)
        compressed_smpl[f'layer{l}.weight'] = compressed_smpl[f'layer{l}.weight_base']

    return compressed_smpl, 1 - nonzero/count