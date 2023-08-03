import os
# do not prealocate memory
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import argparse
import warnings
import jax.numpy as jnp
import numpyro
import equinox as eqx

from math import prod
from jax import random, nn, vmap

from bmr4pml.models import SVIRegression, BMRRegression
from bmr4pml.nn import MLP, LeNet, VisionTransformer, resnet18, resnet34, resnet50
from bmr4pml.datasets import load_data
from bmr4pml.inference import fit_and_test

warnings.filterwarnings('ignore')

def standardize(train_images, test_images, num_channels=1):
    mean = train_images.reshape(-1, num_channels).mean(0)
    std = train_images.reshape(-1, num_channels).std(0)

    return (train_images - mean) / std, (test_images - mean) / std

def run_inference(rng_key, nnet, opts_regression, opts_fitting, train_ds, test_ds, reg_model=SVIRegression):

    reg = reg_model(
        nnet,
        **opts_regression
    )

    rng_key, opts_fitting['key'] = random.split(rng_key)
    return fit_and_test(reg, train_ds, test_ds, opts_fitting)


def main(dataset_name, nn_type, methods, platform, seed):
    rng_key = random.PRNGKey(seed)
    num_epochs = 5
    num_iters = 80_000

    # load data
    train_ds, test_ds = load_data(dataset_name, platform=platform, id=0)

    num_channels = train_ds['image'].shape[-1]
    out_size = len(jnp.unique(train_ds['label']))

    train_ds['image'], test_ds['image'] = standardize(train_ds['image'], test_ds['image'], num_channels=num_channels)

    if nn_type != 'dense' and 'mnist' in dataset_name:
        train_ds['image'] = jnp.pad(train_ds['image'].transpose(0, 3, 1, 2), ((0, 0), (0, 0), (2, 2), (2, 2)))
        test_ds['image'] = jnp.pad(test_ds['image'].transpose(0, 3, 1, 2), ((0, 0), (0, 0), (2, 2), (2, 2)))
    else:
        train_ds['image'] = train_ds['image'].transpose(0, 3, 1, 2)
        test_ds['image'] = test_ds['image'].transpose(0, 3, 1, 2)

    in_size = train_ds['image'].shape[-3:]

    print('input size:', in_size)

    try:
        results = jnp.load(f'results/{dataset_name}.npz', allow_pickle=True)['results'].item()
        results[nn_type] = results.pop(nn_type, {})
    except:
        results = {nn_type: {}}

    opts_regression = {
        'regtype': 'multinomial',
        'gamma0': 0.05,
        'autoguide': 'delta',
        'optim_kwargs': {
            'learning_rate': 1e-4
        }
    }

    opts_fitting = {
        'num_epochs': num_epochs,
        'num_iters': num_iters,
        'num_samples': 1,
        'model_kwargs': {
            'batch_size': 512, 
            'with_hyperprior': False
        }
    }

    rng_key, key = random.split(rng_key)
    if nn_type == 'dense':
        depth = 5
        num_neurons = 500
        rng_key, key = random.split(rng_key)
        nnet = MLP( prod(in_size), out_size, num_neurons, depth, activation=nn.swish, dropout_rate=0.5, key=key)
    elif nn_type == 'lenet':
        rng_key, key = random.split(rng_key)
        nnet = LeNet(in_size, activation=nn.swish, dropout_rate=0.5, key=key)
    elif nn_type == 'resnet':
        rng_key, key = random.split(rng_key)
        nnet = resnet18(num_channels=num_channels, num_classes=out_size, activation=nn.swish, key=key)
    elif nn_type == 'vit':
        rng_key, key = random.split(rng_key)
        nnet = VisionTransformer(
                img_size = in_size[1],
                patch_size = 8,
                in_chans = in_size[0],
                num_classes = out_size,
                embed_dim = 192,
                depth = 12,
                num_heads = 3,
                drop_rate = 0.0,
                attn_drop_rate = 0.0,
                drop_path_rate = 0.0,
                key = key
            )
    else:
        raise NotImplementedError

    if 'Flat-MAP' in methods:
        print('Flat-MAP')
        rng_key, key = random.split(rng_key)
        output = run_inference(key, nnet, opts_regression, opts_fitting, train_ds, test_ds, reg_model=SVIRegression)
        results[nn_type]['Flat-MAP'] = output
        jnp.savez(f'results/{dataset_name}.npz', results=results)

    method_opts_reg = {
        'Flat-FF': {'autoguide': 'mean-field', 'optim_kwargs': {'learning_rate': 1e-3}},
        'Flat-STR': {'autoguide': 'structured', 'optim_kwargs': {'learning_rate': 1e-3}},
        'Tiered-FF': {'tau0': 1e-2, 'reduced': True, 'autoguide': 'mean-field', 'optim_kwargs': {'learning_rate': 1e-3}},
        'Tiered-STR': {'tau0': 1e-2, 'reduced': True, 'autoguide': 'structured', 'optim_kwargs': {'learning_rate': 1e-3}}
    }

    method_opts_fit = {
        'Flat-FF': {'num_samples': 100, 'model_kwargs': {'batch_size': 512, 'with_hyperprior': False}},
        'Flat-STR': {'num_samples': 100, 'model_kwargs': {'batch_size': 512, 'with_hyperprior': False}},
        'Tiered-FF': {'num_samples': 100, 'model_kwargs': {'batch_size': 512, 'with_hyperprior': True}},
        'Tiered-STR': {'num_samples': 100, 'model_kwargs': {'batch_size': 512, 'with_hyperprior': True}}
    }

    # turn off dropout for Bayesian estimation
    nnet = eqx.tree_inference(nnet, value=True)

    for method in ['Flat-FF', 'Flat-STR', 'Tiered-FF', 'Tiered-STR']:
        if method in methods:
            print(method)
            opts_regression = opts_regression | method_opts_reg[method]
            opts_fitting = opts_fitting | method_opts_fit[method]

            output = run_inference(key, nnet, opts_regression, opts_fitting, train_ds, test_ds, reg_model=SVIRegression)
            results[nn_type][method] = output
            jnp.savez(f'results/{dataset_name}.npz', results=results)

    
    method_opts_reg = {
        'BMR-S&S':  {
        'gamma0': 0.05, 
        'pruning': 'spike-and-slab', 
        'posterior': 'normal', 
        'optim_kwargs': {
                'learning_rate': 1e-3
            },
        },
        
        'BMR-RHS': {
            'tau0': 1e-2, # if 'mnist' in dataset_name else 1e-4, 
            'reduced': True, 
            'pruning': 'regularised-horseshoe'
        }
    }

    opts_fitting = opts_fitting | {
        'num_iters': num_iters - 10_000, 
        'num_samples': 100, 
        'pruning_kwargs': {'delta': 1e-5},
        'model_kwargs': {'batch_size': 512, 'with_hyperprior': False}    
    }

    rng_key, key = random.split(rng_key)
    opts_regression = opts_regression | method_opts_reg['BMR-S&S']
    state, params = run_inference(
        key, nnet, opts_regression, opts_fitting | {'warmup_iters': 400_000}, train_ds, test_ds, reg_model=BMRRegression
    )

    for method in ['BMR-S&S', 'BMR-RHS']:
        if method in methods:
            print(method)
            opts_regression = opts_regression | method_opts_reg[method]
            opts_fitting = opts_fitting | {'state': deepcopy(state), 'params': deepcopy(params)}

            rng_key, key = random.split(rng_key)
            output = run_inference(key, nnet, opts_regression, opts_fitting, train_ds, test_ds, reg_model=BMRRegression)
            results[nn_type][method] = output
            jnp.savez(f'results/{dataset_name}.npz', results=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bayesian deep neural networks training")
    parser.add_argument("-n", "--network", nargs='?', default='dense', type=str)
    parser.add_argument("--device", nargs='?', default='gpu', type=str)
    parser.add_argument("--seed", nargs='?', default=137, type=int)
    parser.add_argument("-ds", "--data-set", nargs='?', default='fashion_mnist', type=str)

    default = [
        'Flat-MAP', 
        'Flat-FF',
        'Flat-STR',
        'Tiered-FF',
        'Tiered-STR',
        'BMR-S&S',
        'BMR-RHS'
    ]
    parser.add_argument(
        "-m", "--methods", nargs='+', default=default, type=str
    )

    args = parser.parse_args()
    numpyro.set_platform(args.device)

    main(args.data_set, args.network, args.methods, args.device, args.seed)
