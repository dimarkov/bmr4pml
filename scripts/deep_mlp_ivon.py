import os
import argparse

# do not prealocate memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import jax.numpy as jnp
import equinox as eqx
import optax
import augmax

import jax.numpy as jnp
import equinox as eqx
import optax
import augmax

from functools import partial

from tensorflow_probability.substrates.jax.stats import expected_calibration_error as compute_ece
from jax import random as jr, nn, vmap, lax, config
from jax.scipy.special import polygamma
import jax.tree_util as jtu

from mlpox.networks import DeepMlp, MlpMixer

from blrax.optim import ivon
from blrax.states import ScaleByIvonState
from blrax.utils import noisy_value_and_grad, get_scale, sample_posterior

from bmr4pml.datasets import load_data
from bmr4pml.delta_f import ΔF_mf_delta

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
        _params.append( m * mu )

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
    mc_samples=(),
    num_epochs=1,
    batch_size=32,
    start_pruning=100,
    alpha=.0,
    pi=.5
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
        num_epochs: Number of epochs to train
        batch_size: Batch size for training
        start_pruning: Start BMR based model pruning after given epoch
        alpha: Label smoothing factor
        pi: Expected prior sparsity factor
    """
    
    
    params, static = eqx.partition(nnet, eqx.is_array)  # split model into params and static fields
    mask = jtu.tree_map(lambda x: jnp.ones_like(x, jnp.bool), params)  # initialize mask
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

        start = epoch > start_pruning
        select = partial(jnp.where, start)

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
    (params, mask, final_opt_state), metrics = lax.scan(
        train_epoch,
        init_carry,
        (keys, jnp.arange(num_epochs))
    )
    trained_model = eqx.combine(params, static)
    return trained_model, final_opt_state, metrics

def main(args, network, m_config, o_config):
    dataset = args.dataset
    seed = args.seed
    num_epochs = args.epochs
    batch_size = args.batch_size
    save_every = args.save_every
    platform = args.device

    key = jr.PRNGKey(seed)

    # load data
    train_ds, test_ds = load_data(dataset, platform=platform, id=0)  # load data to device 0

    datasize = len(train_ds['image'])
    num_iters = num_epochs * datasize // batch_size

    # define data augmentation
    img_size = m_config['img_size']
    def augdata(img, key=None):
        if key is None:
            func = augmax.Chain(
                augmax.ByteToFloat(),
                augmax.Normalize(),
                augmax.Resize(img_size),
            )
            return vmap(func, in_axes=(None, 0))(jr.PRNGKey(0), img)
        else:
            keys = jr.split(key, img.shape[0])
            func = augmax.Chain(
                        augmax.ByteToFloat(),
                        augmax.Normalize(),
                        augmax.Resize(img_size),
                        augmax.RandomSizedCrop(img_size, img_size),
                        augmax.HorizontalFlip()
                    )
            return vmap(func)(keys, img)
        
    # load model
    key, _key = jr.split(key)
    if network in ["smlp", "bmlp"]:
        nnet = DeepMlp(**m_config, key=_key)
    else:
        nnet = MlpMixer(**m_config, key=_key)

    # set optimizer
    if 'lion' in o_config:
        optim = optax.lion(**o_config['lion'])
        mc_samples = ()
    elif 'ivon' in o_config:
        lr_conf = o_config['lr']
        lr_conf['decay_steps'] = num_iters
        lr_conf['warmup_steps'] = num_iters // 10
        lr_schd = optax.schedules.warmup_cosine_decay_schedule(
            **lr_conf
        )
        key, _key = jr.split(key)
        conf = o_config['ivon']
        conf['num_data'] = datasize
        optim = ivon(_key, lr_schd, **conf)
        mc_samples = o_config['ivon']['mc_samples']
        nnet = eqx.nn.inference_mode(nnet, True)  # remove dropouts

    # run training
    opt_state = None
    s_prune = args.start_bmr
    for i in range(num_epochs // save_every):
        key, _key = jr.split(key)
        nnet, opt_state, metrics = run_training(
            _key,
            nnet, 
            optim,
            augdata, 
            train_ds, 
            test_ds,
            opt_state=opt_state,
            mc_samples=mc_samples,
            num_epochs=save_every,
            batch_size=batch_size,
            start_pruning=s_prune,
            alpha=args.label_smooth,
            pi=0.5
        )
        s_prune = max(1, s_prune - save_every)

        #TODO: save model checkpoint, opt_state, and test metrics
        to_save = {"nnet": nnet, "opt_state": opt_state, "metrics": metrics}
        vals = jtu.tree_map(lambda x: x[-1], metrics)
        print(i, nn_type, [(name, vals[name].item()) for name in vals if name != 'pf'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="deep MLP training")
    parser.add_argument("-o", "--optimizer", nargs='?', default='ivon', type=str)
    parser.add_argument("-n", "--networks", nargs='+', default=['smlp', 'bml', 'mixer'], type=str)
    parser.add_argument("--device", nargs='?', default='gpu', type=str)
    parser.add_argument("--seed", nargs='?', default=137, type=int)
    parser.add_argument("-ds", "--dataset", nargs='?', default='cifar10', type=str)
    parser.add_argument("--save-every", nargs='?', default=10, type=int)
    parser.add_argument("-e", "--epochs", nargs='?', default=100, type=int)
    parser.add_argument("-bs", "--batch-size", nargs='?', default=64, type=int)
    parser.add_argument("-ls", "--label-smooth", nargs='?', default=0.0, type=float)
    parser.add_argument("-nb", "--num-blocks", nargs='?', default=6, type=int)
    parser.add_argument("-ed", "--embed-dim", nargs='?', default=256, type=int)
    parser.add_argument("-sbmr", "--start-bmr", nargs='?', default=100, type=int)
    parser.add_argument("-mc", "--mc-samples", nargs='?', default=1, type=int)

    args = parser.parse_args()
    config.update("jax_platform_name", args.device)

    num_classes = 1
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    
    # specify configurations for different neural networks
    # for now this is shared across both types, but it should 
    # become obsolete if we simply load pretrained models
    m_config = {
        'img_size': 64,
        'in_chans': 3,
        'embed_dim': args.embed_dim,
        'num_blocks': args.num_blocks,
        'num_classes': num_classes
        }

    if args.optimizer == 'lion':
        o_config = {'lion': {'learning_rate': 1e-4, 'weight_decay': 1e-5}}
    if args.optimizer == 'ivon':
        o_config = {
            'ivon': {'s0': 1., 'h0': 1., 'mc_samples': args.mc_samples, 'clip_radius': 1e3},
            'lr': {
                'init_value': 1e-2,
                'peak_value': 1e-1,
                'end_value': 1e-4
            }
        }

    for nn_type in args.networks:
        if nn_type == 'smlp':
            m_config['mlp_type'] = 'standard'
        elif nn_type == 'bmlp':
            m_config['mlp_type'] = 'bottleneck'
        
        main(args, nn_type, m_config, o_config)