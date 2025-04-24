import jax.numpy as jnp

MEAN_DICT = {
    "imagenet21": jnp.asarray([0.485, 0.456, 0.406]),
    "imagenet": jnp.asarray([0.485, 0.456, 0.406]),
    "imagenet_real": jnp.asarray([0.485, 0.456, 0.406]),
    "tinyimagenet": jnp.asarray([0.485, 0.456, 0.406]),
    "cifar10": jnp.asarray([0.49139968, 0.48215827, 0.44653124]),
    "cifar100": jnp.asarray([0.49139968, 0.48215827, 0.44653124]),
    "stl10": jnp.asarray([0.4914, 0.4822, 0.4465]),
}


STD_DICT = {
    "imagenet21": jnp.asarray([0.229, 0.224, 0.225]),
    "imagenet": jnp.asarray([0.229, 0.224, 0.225]),
    "imagenet_real": jnp.asarray([0.229, 0.224, 0.225]),
    "tinyimagenet": jnp.asarray([0.229, 0.224, 0.225]),
    "cifar10": jnp.asarray([0.24703233, 0.24348505, 0.26158768]),
    "cifar100": jnp.asarray([0.24703233, 0.24348505, 0.26158768]),
    "stl10": jnp.asarray([0.2471, 0.2435, 0.2616]),
}