from math import prod
from typing import Optional, Callable, Sequence, List, Tuple, Union

import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
from jax import lax
from jaxtyping import Array, PRNGKeyArray

from equinox import nn
from equinox import Module, field, static_field


class LeNet(Module):
    """A simple convolutional network. Adapted example from 
    https://github.com/FluxML/model-zoo/tree/master/vision/conv_mnist
    """

    layers: List[nn.Linear]
    activation: Callable
    final_activation: Callable
    dropout: Callable
    pool: Callable
    in_size: Tuple[int] = field(static=True)
    conv_features: Sequence[int] = field(static=True)
    dense_features: Sequence[int] = field(static=True)
    use_bias: bool = field(static=True)

    def __init__(
        self,
        in_size: Tuple[int],
        conv_features: Sequence[int] = [6, 16, 120],
        dense_features: Sequence[int] = [84, 10],
        kernel_size: Sequence[int] = (5, 5),
        window_shape: Sequence[int] = (2, 2),
        stride: Sequence[int] = (2, 2),
        activation: Callable = jnn.relu,
        final_activation: Callable = nn.Identity(),
        dropout_rate: float = 0.,
        use_bias: bool = True,
        *,
        key: PRNGKeyArray,
        **kwargs,
    ):
        
        depth = len(conv_features) + len(dense_features)
        super().__init__(**kwargs)
        keys = jr.split(key, depth + 1)
        layers = ()
        layers = ()

        self.activation = activation
        self.final_activation = final_activation

        self.in_size = in_size

        self.conv_features = conv_features
        self.dense_features = dense_features

        self.use_bias = use_bias

        self.pool = nn.AvgPool2d(kernel_size=window_shape, stride=stride)

        self.dropout = nn.Dropout(p=dropout_rate)

        in_channels = in_size[0]
        nc = len(conv_features)
        _x = jnp.zeros(in_size)
        for i, out_channels in enumerate(conv_features):
            key = keys[i]
            conv = nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                use_bias = use_bias,
                key = key,
            )
            _x = conv(_x)
            if i < nc - 1:
                _x = self.pool(_x)

            layers += (conv,)

            in_channels = out_channels
        
        in_features = prod(_x.shape)
        for i, out_features in enumerate(dense_features):
            key = keys[nc + i]
            layers += (nn.Linear(in_features=in_features, out_features=out_features, key=key),)
            in_features = out_features

        self.layers = layers


    def __call__(
        self, x: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> Array:
        """**Arguments:**

        - `x`: A JAX array with shape `(in_size,)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array with shape `(dense_features[-1],)`.
        """
        nc = len(self.conv_features)

        for i in range(nc):
            x = self.layers[i](x)
            x = self.activation(x)
            if i < nc - 1:
                x = self.pool(x)
        
        x = jnp.ravel(x)
        for layer in self.layers[nc:-1]:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x, key=key)

        x = self.layers[-1](x)
        x = self.final_activation(x)
        return x
