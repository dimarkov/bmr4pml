from math import prod
from typing import Optional, Callable, Sequence, List, Tuple, Union

import jax
import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
from jax import lax
from jaxtyping import Array, PRNGKeyArray

from equinox import nn

class MLP(nn.MLP):
    """Standard Multi-Layer Perceptron; also known as a feed-forward network."""

    dropout: Callable

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        activation: Callable = jnn.relu,
        final_activation: Callable = nn.Identity(),
        use_bias: bool = True,
        use_final_bias: bool = True,
        *,
        dropout_rate: float = 0.,
        key: PRNGKeyArray,
    ):
        """**Arguments**:

        - `in_size`: The size of the input layer.
        - `out_size`: The size of the output layer.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers.
        - `activation`: The activation function after each hidden layer. Defaults to
            ReLU.
        - `final_activation`: The activation function after the output layer. Defaults
            to the identity.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """

        super().__init__(in_size, out_size, width_size, depth, activation, final_activation, use_bias, use_final_bias, key=key)
        self.dropout = nn.Dropout(p=dropout_rate)

    def __call__(
        self, x: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> Array:
        """**Arguments:**

        - `x`: A JAX array with shape `(in_size,)` after ravel is applied.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for calculating
            which elements to dropout. (Keyword only argument.)

        **Returns:**

        A JAX array with shape `(out_size,)`.
        """
        x = jnp.ravel(x)
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
            x = self.dropout(x, key=key)

        x = self.layers[-1](x)
        x = self.final_activation(x)
        return x