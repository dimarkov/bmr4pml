from math import prod
from typing import Optional, Callable, Sequence, List, Tuple, Union

import jax.numpy as jnp
import jax.nn as jnn
import jax.random as jr
from jax import vmap
from jaxtyping import Array, PRNGKeyArray

from equinox import nn, Module
from .utils import Patch, PatchConvEmbed, PatchLinearEmbed

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
    

class _MlpBlock(Module):
    activation: Callable
    layers: Sequence[nn.Linear]

    def __init__(
        self,
        in_features,
        hidden_dim, 
        out_features,
        activation,
        *,
        key: PRNGKeyArray
    ):
        super().__init__()
        keys = jr.split(key, 2)
        self.activation = activation
        
        self.layers = (
            nn.Linear(in_features=in_features, out_features=hidden_dim, key=keys[0]),
            nn.Linear(in_features=hidden_dim, out_features=out_features, key=keys[1])
        )

    def __call__(self, x):
        y = self.layers[0](x)
        y = self.activation(y)
        return self.layers[1](y)


class _MixerBlock(Module):
    """Mixer block layer."""
    norm1: Callable
    norm2: Callable
    blocks: Sequence[_MlpBlock]

    def __init__(
        self,
        tokens_dim,
        embed_dim,
        tokens_hidden_dim,
        embed_hidden_dim,
        activation,
        *,
        key: PRNGKeyArray
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6, use_weight=False, use_bias=False)

        keys = jr.split(key, 2)
        self.blocks = [
            _MlpBlock(tokens_dim, tokens_hidden_dim, tokens_dim, activation, key=keys[0]),  # token_mixing
            _MlpBlock(embed_dim, embed_hidden_dim, embed_dim, activation, key=keys[1])  # channel_mixing
        ]
    
    def __call__(self, x):
        y = vmap(self.norm1)(x)
        y = jnp.swapaxes(y, 0, 1)
        y = vmap(self.blocks[0])(y)
        y = jnp.swapaxes(y, 0, 1)
        x = x + y
        y = vmap(self.norm2)(x)
        return x + vmap(self.blocks[1])(y)


class MlpMixer(Module):
    """MLP Mixer architecture proted from https://github.com/google-research/vision_transformer/blob/linen/vit_jax/models_mixer.py."""
    mixers: Sequence[_MixerBlock]
    norm: nn.LayerNorm
    patch_embed: Module
    fc: nn.Linear
    augmentation: Callable

    def __init__(
            self,
            img_size=32,
            patch_size=4,
            in_channels=1,
            embed_dim=256,
            tokens_hidden_dim=128,
            hidden_dim_ratio=4,
            num_classes=10,
            num_blocks=8,
            activation=jnn.gelu,
            patch_embed=PatchConvEmbed,
            augmentation=None,
            *,
            key: PRNGKeyArray
        ):
        super().__init__()
        embed_hidden_dim = hidden_dim_ratio * tokens_hidden_dim
        keys = jr.split(key, num_blocks + 2)

        self.augmentation = augmentation if augmentation is not None else lambda key, x: x

        self.patch_embed = patch_embed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_channels,
            embed_dim=embed_dim,
            key=keys[-2]
        )

        tokens_dim = self.patch_embed.num_patches
        self.mixers = [
            _MixerBlock(
                tokens_dim, 
                embed_dim, 
                tokens_hidden_dim, 
                embed_hidden_dim, 
                activation, 
                key=keys[i]
            ) 
            for i in range(num_blocks)
        ]

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6, use_bias=False, use_weight=False)

        # Classifier head
        self.fc = nn.Linear(embed_dim, num_classes, key=keys[-1])

    def __call__(
        self, x: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> Array:
        """**Arguments:**

        - `x`: A JAX array with shape `(in_size,)` after ravel is applied.
        - `key`: A `jax.random.PRNGKey` not used; present for syntax consistency. (Keyword only argument.)

        **Returns:**

        A JAX array with shape `(out_size,)`.
        """
        if key is not None:
            x = self.augmentation(key, x).transpose(2, 0, 1)
        else:
            x = x.transpose(2, 0, 1)
        
        x = self.patch_embed(x)
        # x shape is (h w) embed_dim
        for mixer in self.mixers:
            x = mixer(x)

        x = vmap(self.norm)(x)
        x = jnp.mean(x, axis=-2)
        return self.fc(x)
    

class _StandardMlpBlock(Module):
    norm: Callable
    activation: Callable
    linear: nn.Linear

    def __init__(
        self,
        in_features,
        out_features,
        activation,
        *,
        key: PRNGKeyArray
    ):
        super().__init__()
        self.activation = activation
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, key=key)
        self.norm = nn.LayerNorm(in_features, use_weight=False, use_bias=False)


    def __call__(self, x):
        x = self.norm(x)
        x = self.linear(x)
        return self.activation(x)
    

class _BottlneckMlpBlock(Module):
    block: _StandardMlpBlock
    linear: nn.Linear

    def __init__(
        self,
        in_features,
        activation,
        ratio=4,
        *,
        key: PRNGKeyArray
    ):
        super().__init__()
        keys = jr.split(key, 2)
        self.block = _StandardMlpBlock(in_features, ratio * in_features, activation, key=keys[0])
        self.linear = nn.Linear(in_features=ratio * in_features, out_features=in_features, key=keys[1])

    def __call__(self, x):
        y = self.block(x)
        return x + self.linear(y)
    

class DeepMlp(Module):
    """Deep Standard and Inverted Bottlneck MLP pofted from https://github.com/gregorbachmann/scaling_mlps."""
    layers: Union[Sequence[_BottlneckMlpBlock],Sequence[_StandardMlpBlock]]
    linear_embed: nn.Linear
    fc: nn.Linear
    augmentation: Callable
    inference: bool

    def __init__(
            self,
            img_size=32,
            in_channels=1,
            embed_dim=256,
            hidden_dim_ratio=4,
            num_classes=10,
            num_blocks=8,
            activation=jnn.gelu,
            augmentation=None,
            type="bottleneck",
            inference=False,
            *,
            key: PRNGKeyArray
        ):
        super().__init__()
        keys = jr.split(key, num_blocks + 2)

        self.inference = inference

        self.augmentation = augmentation if augmentation is not None else lambda key, x: x
        
        img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        in_features = prod(img_size) * in_channels
        self.linear_embed = nn.Linear(in_features=in_features, out_features=embed_dim, key=keys[-1])
        
        if type == "bottleneck":
            self.layers = [
                _BottlneckMlpBlock(
                    embed_dim,
                    activation,
                    ratio=hidden_dim_ratio,
                    key=keys[i]
                ) 
                for i in range(num_blocks)
            ]
        elif type == "standard":
            self.layers = [
                _StandardMlpBlock(
                    embed_dim, 
                    embed_dim,
                    activation, 
                    key=keys[i]
                ) 
                for i in range(num_blocks)
            ]
        else:
            raise NotImplementedError

        # Classifier head
        self.fc = nn.Linear(embed_dim, num_classes, key=keys[-2])

    def __call__(
        self, x: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> Array:
        """**Arguments:**

        - `x`: A JAX array with shape `img_size + (num_channels,)`.
        - `key`: A `jax.random.PRNGKey` not used; present for syntax consistency. (Keyword only argument.)

        **Returns:**

        A JAX array with shape `(num_classes,)`.
        """
        if self.inference:
            x = x.reshape(-1)
        else:
            if key is not None:
                x = self.augmentation(key, x).reshape(-1)
            else:
                x = x.reshape(-1)
        
        x = self.linear_embed(x)
        # x shape is embed_dim
        for layer in self.layers:
            x = layer(x)

        return self.fc(x)