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
from math import prod

from jax import random as jr, vmap, config
import jax.tree_util as jtu

from mlpox.networks import DeepMlp, MlpMixer

from blrax.optim import ivon
from bmr4pml.datasets import load_data

from .funcs import run_training, resize_images, augmentdata, get_number_of_parameters


def main(args, network, m_config, o_config):
    dataset = args.dataset
    seed = args.seed
    num_epochs = args.epochs
    num_warmup_epochs = args.warmup
    batch_size = args.batch_size
    save_every = args.save_every
    platform = args.device

    key = jr.PRNGKey(seed)

    # load data
    train_ds, test_ds = load_data(dataset, platform=platform, id=0)  # load data to device 0

    datasize = len(train_ds['image'])
    num_iters = num_epochs * datasize // batch_size
    warmup_steps = num_warmup_epochs * datasize // batch_size

    # define data augmentation
    train_ds['image'] = resize_images(train_ds['image'], m_config['img_size'])
    test_ds['image'] = resize_images(test_ds['image'], m_config['img_size'])

    mean = train_ds['image'].mean(axis=(0, 1, 2))
    std = train_ds['image'].std(axis=(0, 1, 2))
    augdata = partial(augmentdata, mean=mean, std=std)
        
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
        lr_conf['warmup_steps'] = warmup_steps
        lr_schd = optax.schedules.warmup_cosine_decay_schedule(
            **lr_conf
        )
        key, _key = jr.split(key)
        conf = o_config['ivon']
        conf['num_data'] = num_epochs * datasize  # because of augmentation we increase the effective datasize
        optim = ivon(_key, lr_schd, **conf)
        mc_samples = o_config['ivon']['mc_samples']
        nnet = eqx.nn.inference_mode(nnet, True)  # remove dropouts
    
    num_params = get_number_of_parameters(nnet)
    print(f"Number of parameters of {network} is {num_params}.")

    # run training
    opt_state = None
    mask = None
    s_prune = args.start_bmr
    for i in range(num_epochs // save_every):
        key, _key = jr.split(key)
        nnet, opt_state, mask, metrics = run_training(
            _key,
            nnet, 
            optim,
            augdata, 
            train_ds, 
            test_ds,
            opt_state=opt_state,
            mask=mask,
            mc_samples=mc_samples,
            num_epochs=save_every,
            batch_size=batch_size,
            start_pruning=s_prune,
            alpha=args.label_smooth,
            pi=0.5
        )
        s_prune = max(0, s_prune - save_every)

        total_pf = 0.
        for ms in jtu.tree_flatten(mask)[0]:
            total_pf += jnp.sum(1 - ms)
        
        total_pf = total_pf.item() / num_params

        #TODO: save model checkpoint, opt_state, and test metrics
        to_save = {"nnet": nnet, "opt_state": opt_state, "metrics": metrics}
        vals = jtu.tree_map(lambda x: x[-1], metrics)
        print(i, nn_type, [(name, f'{vals[name].item():.3f}') for name in vals if name != 'pf'] + [('pruned_frac', f'{total_pf:.3f}')])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="deep MLP training")
    parser.add_argument("-o", "--optimizer", nargs='?', default='ivon', type=str)
    parser.add_argument("-n", "--networks", nargs='+', default=['smlp', 'bmlp', 'mixer'], type=str)
    parser.add_argument("--device", nargs='?', default='gpu', type=str)
    parser.add_argument("--seed", nargs='?', default=137, type=int)
    parser.add_argument("-ds", "--dataset", nargs='?', default='cifar10', type=str)
    parser.add_argument("--save-every", nargs='?', default=10, type=int)
    parser.add_argument("-e", "--epochs", nargs='?', default=100, type=int)
    parser.add_argument("-w", "--warmup", nargs='?', default=10, type=int)
    parser.add_argument("-bs", "--batch-size", nargs='?', default=512, type=int)
    parser.add_argument("-ls", "--label-smooth", nargs='?', default=0.0, type=float)
    parser.add_argument("-nb", "--num-blocks", nargs='?', default=6, type=int)
    parser.add_argument("-ed", "--embed-dim", nargs='?', default=512, type=int)
    parser.add_argument("-sbmr", "--start-bmr", nargs='?', default=10, type=int)
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
        o_config = {'lion': {'learning_rate': 1e-5, 'weight_decay': 1e-3}}
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