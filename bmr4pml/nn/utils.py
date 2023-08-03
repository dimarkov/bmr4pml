import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
import jax.random as jrandom

from jaxtyping import Array, PRNGKeyArray
from typing import Callable, Tuple, Union, Optional


class DropPath(eqx.Module):
    """Effectively dropping a sample from the call.
    Often used inside a network along side a residual connection.
    Equivalent to `torchvision.stochastic_depth`."""

    p: float
    inference: bool
    mode: str

    def __init__(self, p: float = 0.0, inference: bool = False, mode="global"):
        """**Arguments:**

        - `p`: The probability to drop a sample entirely during forward pass
        - `inference`: Defaults to `False`. If `True`, then the input is returned unchanged
        This may be toggled with `equinox.tree_inference`
        - `mode`: Can be set to `global` or `local`. If `global`, the whole input is dropped or retained.
                If `local`, then the decision on each input unit is computed independently. Defaults to `global`

        !!! note

            For `mode = local`, an input `(channels, dim_0, dim_1, ...)` is reshaped and transposed to
            `(channels, dims).transpose()`. For each `dim x channels` element,
            the decision to drop/keep is made independently.

        """
        self.p = p
        self.inference = inference
        self.mode = mode

    def __call__(self, x, *, key: PRNGKeyArray) -> Array:
        """**Arguments:**

        - `x`: An any-dimensional JAX array to drop
        - `key`: A `jax.random.PRNGKey` used to provide randomness for calculating
            which elements to dropout. (Keyword only argument.)
        """
        if self.inference or self.p == 0.0:
            return x
        if key is None:
            raise RuntimeError(
                "DropPath requires a key when running in non-deterministic mode. Did you mean to enable inference?"
            )

        keep_prob = 1 - self.p
        if self.mode == "global":
            noise = jrandom.bernoulli(key, p=keep_prob)
        else:
            noise = jnp.expand_dims(
                jrandom.bernoulli(key, p=keep_prob, shape=[x.shape[0]]).reshape(-1),
                axis=[i for i in range(1, len(x.shape))],
            )
        if keep_prob > 0.0:
            noise /= keep_prob
        return x * noise


class MlpProjection(eqx.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    fc1: eqx.Module
    act: Callable
    drop1: nn.Dropout
    fc2: eqx.Module
    drop2: nn.Dropout

    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        lin_layer: nn.Linear = nn.Linear,
        act_layer: Callable = None,
        drop: Union[float, Tuple[float]] = 0.0,
        *,
        key: PRNGKeyArray = None
    ):
        """**Arguments:**

        - `in_features`: The expected dimension of the input
        - `hidden_features`: Dimensionality of the hidden layer
        - `out_features`: The dimension of the output feature
        - `lin_layer`: Linear layer to use. For transformer like architectures, `Linear2d` can be easier to integrate.
        - `act_layer`: Activation function to be applied to the intermediate layers
        - `drop`: The probability associated with `Dropout`
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
        initialisation. (Keyword only argument.)
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = drop if isinstance(drop, tuple) else (drop, drop)
        keys = jrandom.split(key, 2)
        self.fc1 = lin_layer(in_features, hidden_features, key=keys[0])
        self.act = act_layer
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = lin_layer(hidden_features, out_features, key=keys[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def __call__(self, x: Array, *, key: PRNGKeyArray) -> Array:
        """**Arguments:**

        - `x`: The input `JAX` array
        - `key`: Utilised by few layers in the network such as `Dropout` or `DropPath`
        """
        keys = jrandom.split(key, 2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x, key=keys[0])
        x = self.fc2(x)
        x = self.drop2(x, key=keys[1])
        return x


class PatchEmbed(eqx.Module):
    """2D Image to Patch Embedding ported from Timm"""

    img_size: Tuple[int]
    patch_size: Tuple[int]
    grid_size: Tuple[int]
    num_patches: int
    flatten: bool
    proj: eqx.nn.Conv2d
    norm: eqx.Module

    def __init__(
        self,
        img_size: Union[int, Tuple[int]] = 224,
        patch_size: Union[int, Tuple[int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: "eqx.Module" = None,
        flatten: bool = True,
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        """
        **Arguments:**

        - `img_size`: The size of the input image. Defaults to `(224, 224)`
        - `patch_size`: Size of the patch to construct from the input image. Defaults to `(16, 16)`
        - `in_chans`: Number of input channels. Defaults to `3`
        - `embed_dim`: The dimension of the resulting embedding of the patch. Defaults to `768`
        - `norm_layer`: The normalisation to be applied on an input. Defaults to None
        - `flatten`: If enabled, the `2d` patches are flattened to `1d`
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        super().__init__()
        self.img_size = (
            img_size if isinstance(img_size, tuple) else (img_size, img_size)
        )
        self.patch_size = (
            patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size)
        )
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        if key is None:
            key = jrandom.PRNGKey(0)
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, key=key
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def __call__(
        self, x: Array, *, key: Optional[PRNGKeyArray] = None
    ) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape`(in_chans, img_size[0], img_size[1])`.
        - `key`: Ignored
        """
        C, H, W = x.shape
        if H != self.img_size[0] or W != self.img_size[1]:
            raise ValueError(
                f"Input image height ({H},{W}) doesn't match model ({self.img_size})."
            )

        x = self.proj(x)
        if self.flatten:
            x = jax.vmap(jnp.ravel)(x)
            x = jnp.moveaxis(x, 0, -1)  # CHW -> NC
        x = self.norm(x)
        return x
