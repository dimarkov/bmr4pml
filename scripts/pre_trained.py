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

from jax import random as jr, nn, vmap, config
import jax.tree_util as jtu

from mlpox.networks import DeepMlp, MlpMixer
from mlpox.load_models import load_model

from blrax.optim import ivon
from bmr4pml.datasets import load_data
from .funcs import run_training, resize_images, augmentdata, get_number_of_parameters, compute_ece
from .data_stats import MEAN_DICT, STD_DICT

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

    mean = MEAN_DICT[args.dataset]
    std = STD_DICT[args.dataset]
    augdata = partial(augmentdata, mean=mean, std=std)
        
    # load model
    name = f"B_{m_config['num_blocks']}-Wi_{m_config['embed_dim']}_res_64_in21k"
    if args.pretrained == 'in21k':
        nnet = load_model(name)
        key, _key = jr.split(key)
        fc = eqx.nn.Linear(
            m_config['embed_dim'],
            m_config['num_classes'],
            key=_key
        )
        nnet = eqx.tree_at(lambda m: m.fc, nnet, fc)
    elif args.pretrained == 'in21k_cifar':
        nnet = load_model(name + '_' + args.dataset)

    # get pretrained network test stats
    def evaluate_pretrained(images, labels):
        aug_images = augdata(images, key=None)
        logits = vmap(nnet)(aug_images)
        
        predictions = jnp.argmax(logits, axis=1)
        acc = jnp.mean(predictions == labels)
        nll = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        ece = compute_ece(20, logits=logits, labels_true=labels, labels_predicted=predictions)
        return acc, nll, ece

    acc, nll, ece = evaluate_pretrained(test_ds['image'], test_ds['label'])
    print(f'pre-trained test acc={acc:.3f}, ece={ece:.3f}, nll={nll:.3f}')

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
        conf['num_data'] = num_epochs * datasize
        optim = ivon(_key, lr_schd, **conf)
        mc_samples = o_config['ivon']['mc_samples']
        nnet = eqx.nn.inference_mode(nnet, True)  # remove dropouts
    
    num_params = get_number_of_parameters(nnet)
    print(f"Number of parameters of {network} is {num_params}.")

    # run training
    opt_state = None
    mask = None
    s_prune = args.start_bmr
    trained_nnet = nnet
    for i in range(num_epochs // save_every):
        key, _key = jr.split(key)
        trained_nnet, opt_state, mask, metrics = run_training(
            _key,
            trained_nnet, 
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
        to_save = {"nnet": trained_nnet, "opt_state": opt_state, "metrics": metrics}
        vals = jtu.tree_map(lambda x: x[-1], metrics)
        print(i, nn_type, [(name, f'{vals[name].item():.3f}') for name in vals if name != 'pf'] + [('pruned_frac', f'{total_pf:.3f}')])

    opt_state = None
    key, _key = jr.split(key)
    lt_nnet, opt_state, _, metrics = run_training(
        _key,
        nnet, 
        optim,
        augdata, 
        train_ds, 
        test_ds,
        opt_state=opt_state,
        mask=mask,
        mc_samples=mc_samples,
        num_epochs=num_epochs,
        batch_size=batch_size,
        start_pruning=num_epochs + 1,
        alpha=args.label_smooth
    )

    #TODO: save model checkpoint, opt_state, and test metrics
    to_save = {"nnet": lt_nnet, "opt_state": opt_state, "metrics": metrics}
    vals = jtu.tree_map(lambda x: x[-1], metrics)
    print("lt_nnet", nn_type, [(name, f'{vals[name].item():.3f}') for name in vals if name != 'pf'])


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
    parser.add_argument("-bs", "--batch-size", nargs='?', default=64, type=int)
    parser.add_argument("-ls", "--label-smooth", nargs='?', default=0.0, type=float)
    parser.add_argument("-nb", "--num-blocks", nargs='?', default=6, type=int)
    parser.add_argument("-ed", "--embed-dim", nargs='?', default=256, type=int)
    parser.add_argument("-sbmr", "--start-bmr", nargs='?', default=10, type=int)
    parser.add_argument("-mc", "--mc-samples", nargs='?', default=1, type=int)
    parser.add_argument("--pretrained", nargs='?', default='in21k', type=str)
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
        o_config = {'lion': {'learning_rate': 5e-5, 'weight_decay': 1e-2}}
    if args.optimizer == 'ivon':
        o_config = {
            'ivon': {'s0': 1., 'h0': 1., 'mc_samples': args.mc_samples, 'clip_radius': 1e3},
            'lr': {
                'init_value': 5e-4,
                'peak_value': 1e-2,
                'end_value': 5e-5
            }
        }

    for nn_type in args.networks:
        if nn_type == 'smlp':
            m_config['mlp_type'] = 'standard'
        elif nn_type == 'bmlp':
            m_config['mlp_type'] = 'bottleneck'
        
        main(args, nn_type, m_config, o_config)