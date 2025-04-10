import equinox as eqx
import jax.tree_util as jtu
from jax.scipy.special import polygamma
from jax import nn, lax, numpy as jnp, vmap, random as jr
from tensorflow_probability.substrates.jax.stats import expected_calibration_error as compute_ece
from functools import partial
import optax

from blrax.states import ScaleByIvonState
from blrax.utils import noisy_value_and_grad, get_scale, sample_posterior

from bmr4pml.delta_f import ΔF_mf_delta
import augmax
from math import prod

def get_number_of_parameters(model):

    params, _ = eqx.partition(model, eqx.is_array)
    leafs = jtu.tree_flatten(params)[0]

    return sum([prod(l.shape) for l in leafs])

def resize_images(img, img_size):
    func = augmax.Chain(
        augmax.ByteToFloat(),
        augmax.Resize(img_size, img_size)
    )

    return vmap(func, in_axes=(None, 0))(jr.PRNGKey(0), img)


# define data augmentation
def augmentdata(img, key=None, **kwargs):
    img_size = img.shape[1:-1]
    norm = augmax.Normalize(**kwargs)
    if key is None:
        func = augmax.Chain(
            norm,
        )
        return vmap(func, in_axes=(None, 0))(jr.PRNGKey(0), img)
    else:
        keys = jr.split(key, img.shape[0])
        func = augmax.Chain(
                    norm,
                    augmax.RandomSizedCrop(*img_size),
                    augmax.HorizontalFlip()
                )
        return vmap(func)(keys, img)

@eqx.filter_jit
def prune_parameters(params, sigma, sparsity_prior, mask, prior_scale=1.):
    p_leaf, tree_def = jtu.tree_flatten(params)
    s_leaf = jtu.tree_leaves(sigma)
    m_leaf = jtu.tree_leaves(mask)

    # update iterativly sparsity params and posterior probability of pruning
    _sp = sparsity_prior
    _mask = []
    _params = []
    for mu, scale, msk in zip(p_leaf, s_leaf, m_leaf):
        _sp = sparsity_prior
        for _ in range(4):
            eta = polygamma(1, _sp[0]) - polygamma(1, _sp[1])
            
            # probability of mask = 1
            p = msk * nn.sigmoid(- ΔF_mf_delta(mu, scale, prior_scale=prior_scale) + eta)

            # posterior sparsity params
            _sp = sparsity_prior + jnp.array([p.sum(), (1 - p).sum()])
        
        m = p >= .5  # keep elements for which posterior probability is larger than 1/2
        _mask.append( m )
        _params.append( jnp.where(m, mu, 0.) )

    return (
        jtu.tree_unflatten(tree_def, _params),
        jtu.tree_unflatten(tree_def, _mask),
    )

def run_training(
    key,
    nnet,
    optim,
    data_augmentation,
    train_ds,
    test_ds,
    opt_state=None,
    mask=None,
    mc_samples=(),
    num_epochs=1,
    batch_size=32,
    start_pruning=100,
    alpha=0.0,
    pi=0.5
    ):
    """
    Train a neural network using Equinox and Optax.
    
    Args:
        key: JAX PRNG key
        nnet: Equinox neural network
        optim: Optax (or blrax) optimizer
        data_augmentation: Jax compatible data augmentation function
        train_ds: Training dataset dictionary with 'image' and 'label' keys
        test_ds: Test dataset dictionary with 'image' and 'label' keys
        opt_state: Initial optimizer state, if None it is initiated localy
        mask: Initial mask for the model parameters
        num_epochs: Number of epochs to train
        batch_size: Batch size for training
        start_pruning: Start BMR based model pruning after given epoch
        alpha: Label smoothing factor
        pi: Expected prior sparsity factor
    """
    
    
    params, static = eqx.partition(nnet, eqx.is_array)  # split model into params and static fields
    mask = jtu.tree_map(lambda x: jnp.ones_like(x, jnp.bool), params)  if mask is None else mask # initialize mask
    sparsity_prior = jnp.array([10 * pi, 10 * (1 - pi)])
    opt_state = optim.init(params) if opt_state is None else opt_state  # initialize optimizer state

    num_classes = len(jnp.unique(test_ds['label']))
    n_samples = len(train_ds['image'])
    img_shape = train_ds['image'].shape[-3:]
    steps_per_epoch = n_samples // batch_size
    
    # Cross entropy loss function
    def loss_fn(params, x, y, key=None):
        model = eqx.combine(params, static)
        logits = vmap(partial(model, key=key))(x)
        _y = optax.smooth_labels(nn.one_hot(y, num_classes), alpha=alpha)
        return optax.safe_softmax_cross_entropy(logits, _y).mean()
    
    # Training step function
    @eqx.filter_jit
    def train_step(loss_fn, params, opt_state, x, y, mask, key):
        keys = jr.split(key, mc_samples)
        loss_value, grads = noisy_value_and_grad(loss_fn, opt_state[0], params, x, y, key=keys, mask=mask)
        updates, opt_state = optim.update(grads, opt_state, params)
        updates = jtu.tree_map(lambda g, m: g * m, updates, mask)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_value
    
    # Evaluation function
    @eqx.filter_jit
    def evaluate(params, images, labels):
        def evnet(params):
            model = eqx.nn.inference_mode(eqx.combine(params, static))
            aug_images = data_augmentation(images, key=None)
            return vmap(model)(aug_images)
        
        logits = evnet(params)
            
        predictions = jnp.argmax(logits, axis=1)
        acc = jnp.mean(predictions == labels)
        nll = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        ece = compute_ece(20, logits=logits, labels_true=labels, labels_predicted=predictions)
        return acc, nll, ece
    
    # Inner training loop (one epoch)
    def train_epoch(carry, xs):
        params, mask, opt_state = carry
        key, epoch = xs
        
        # Shuffle training data
        key, _key = jr.split(key)
        perm = jr.permutation(_key, n_samples)
        train_images = train_ds['image'][perm]
        train_labels = train_ds['label'][perm]
        
        def train_step_scan(carry, xs):
            params, opt_state, key = carry
            
            batch_images, batch_labels = xs
            key, _key = jr.split(key)
            aug_batch_images = data_augmentation(batch_images, key=_key)
            key, _key = jr.split(key)
            params, opt_state, loss_value = train_step(
                loss_fn, params, opt_state, aug_batch_images, batch_labels, mask, _key
            )
            return (params, opt_state, key), loss_value
        
        # Run training steps for one epoch
        data = (
            train_images[:steps_per_epoch * batch_size].reshape(steps_per_epoch, batch_size, *img_shape),
            train_labels[:steps_per_epoch * batch_size].reshape(steps_per_epoch, batch_size)
        )
        init_carry = (params, opt_state, key)
        (params, opt_state, key), losses = lax.scan(
            train_step_scan,
            init_carry,
            data
        )

        start = epoch >= start_pruning

        def true_fun(*args):
            params, mask = prune_parameters(*args)
            pruned_fraction = jtu.tree_map( lambda m: jnp.mean(~m), mask)
            return params, mask, pruned_fraction
        
        def false_fun(*args):
            params = args[0]
            mask = args[3]
            pruned_fraction = jtu.tree_map( lambda m: 0., mask)
            return params, mask, pruned_fraction
        

        if isinstance(opt_state[0], ScaleByIvonState):
            state = opt_state[0]
            sigma = get_scale(state)
            prior_scale = state.num_datapoints * state.weight_decay
            _params, _mask, pruned_fraction = lax.cond(
                start, true_fun, false_fun, params, sigma, sparsity_prior, mask, prior_scale
            )
        else:
            pruned_fraction = 0.
            _params = params
            _mask = mask
        
        # Calculate metrics
        key, _key = jr.split(key)
        acc, nll, ece = evaluate(
            params,
            test_ds['image'],
            test_ds['label'],
        )
        
        metrics = {
            'loss': losses.sum() / steps_per_epoch,
            'acc': acc,
            'ece': ece,
            'nll': nll,
            'pf': pruned_fraction
        }
        
        return (_params, _mask, opt_state), metrics
    
    # Run training for multiple epochs
    keys = jr.split(key, num_epochs)
    init_carry = (params, mask, opt_state)
    (params, final_mask, final_opt_state), metrics = lax.scan(
        train_epoch,
        init_carry,
        (keys, jnp.arange(num_epochs))
    )
    trained_model = eqx.combine(params, static)
    return trained_model, final_opt_state, final_mask, metrics