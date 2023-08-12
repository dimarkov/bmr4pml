from typing import Optional, Sequence, Tuple, Union, List

import equinox as eqx
import equinox.nn as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

from .utils import DropPath, MlpProjection, PatchConvEmbed, PatchLinearEmbed, Patch


class _VitAttention(eqx.Module):
    num_heads: int
    scale: float
    qkv: nn.Linear
    attn_drop: nn.Dropout
    proj: nn.Linear
    proj_drop: nn.Dropout

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale=None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        *,
        key: PRNGKeyArray = None,
    ):
        """**Arguments:**

        - `dim`: The feature dimensions of the input.
        - `num_heads`: The number of equal parts to split the input along the `dim`.
        - `qkv_bias`: To add `bias` within the `qkv` computation.
        - `qk_scale`: For scaling the `query` `value` computation.
        - `attn_drop`: Dropout rate for the attention.
        - `proj_drop`: Dropout rate for the projection.
        - key: A `jax.random.PRNGKey` used to provide randomness for parameter
        initialisation. (Keyword only argument.)
        """
        super().__init__()
        keys = jrandom.split(key, 2)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, use_bias=qkv_bias, key=keys[0])
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, key=keys[1])
        self.proj_drop = nn.Dropout(proj_drop)

    def __call__(self, x: Array, *, key: "jax.random.PRNGKey") -> Sequence[Array]:
        """**Arguments:**

        - `x`: The input `JAX` array.
        - `key`: Utilised by few layers in the network such as `Dropout` or `BatchNorm`.
        """
        N, C = x.shape
        keys = jrandom.split(key, 2)
        qkv = jax.vmap(self.qkv)(x)
        qkv = jnp.reshape(qkv, (N, 3, self.num_heads, C // self.num_heads))
        qkv = jnp.transpose(qkv, axes=(1, 2, 0, 3))
        q, k, v = jnp.split(qkv, indices_or_sections=3)

        attn = (q @ jnp.transpose(k, (0, 1, 3, 2))) * self.scale
        attn = jnn.softmax(attn, axis=-1)
        attn = self.attn_drop(attn, key=keys[0])

        x = jnp.reshape(jnp.transpose((attn @ v), axes=(0, 2, 1, 3)), (N, C))
        x = jax.vmap(self.proj)(x)
        x = self.proj_drop(x, key=keys[1])
        return x, attn


class _VitBlock(eqx.Module):
    norm1: eqx.Module
    attn: _VitAttention
    drop_path: DropPath
    norm2: eqx.Module
    mlp: MlpProjection

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=jnn.swish,
        norm_layer=nn.LayerNorm,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments**

        - `dim`: The feature dimensions of the input.
        - `num_heads`: The number of equal parts to split the input along the `dim`.
        - `mlp_ratio`: For computing hidden dimension of the `mlp` (=`dim * mlp_ratio`).
        - `qkv_bias`: To add `bias` within the `qkv` computation.
        - `qk_scale`: For scaling the `query` `value` computation.
        - `drop`: Dropout rate for the `mlp`.
        - `attn_drop`: Dropout rate for the attention.
        - `proj_drop`: Dropout rate for the projection.
        - `act_layer`: Activation applied on to the intermediate outputs.
        - `norm_layer`: Normalisation applied to the intermediate outputs.
        - key: A `jax.random.PRNGKey` used to provide randomness for parameter
        initialisation. (Keyword only argument.)
        """
        super().__init__()
        keys = jrandom.split(key, 2)
        self.norm1 = norm_layer(dim)
        self.attn = _VitAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            key=keys[0],
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MlpProjection(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            key=keys[1],
        )

    def __call__(
        self, x: Array, return_attention=False, *, key: "jax.random.PRNGKey"
    ) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array with `3` channels.
        - `return_attention`: For returning the self-attention computed by the block.
        - `key`: Utilised by few layers in the network such as `Dropout` or `BatchNorm`.
        """
        keys = jrandom.split(key, 4)
        y = jax.vmap(self.norm1)(x)
        y, attn = self.attn(y, key=keys[0])
        if return_attention:
            return attn
        x = x + self.drop_path(y, key=keys[1])
        y = jax.vmap(self.norm2)(x)
        y = jax.vmap(self.mlp)(y, key=jrandom.split(keys[2], x.shape[0]))
        x = x + self.drop_path(y, key=keys[3])
        return x


class Params(eqx.nn.Linear):
    weight: Array

    def __init__(self, shapes: List[Tuple[int]], *, key: PRNGKeyArray):

        keys = jrandom.split(key, len(shapes) + 1)
        w = []
        for k, shape in zip(keys[:-1], shapes): 
            w.append( 
                jrandom.truncated_normal(
                    k, lower=-2, upper=2, shape=shape
                )
             )

        weight = jnp.concatenate(w, 0)
        super().__init__(weight.shape[-1], weight.shape[0], use_bias = False, key=keys[-1])
        self.weight = weight 

class VisionTransformer(eqx.Module):
    """Vision Transformer ported from https://github.com/facebookresearch/dino/blob/main/vision_transformer.py"""

    num_features: int
    inference: bool
    params: Params
    patch_embed: Patch
    pos_drop: nn.Dropout
    blocks: Sequence[_VitBlock]
    norm1: nn.LayerNorm
    norm2: nn.LayerNorm
    fc: nn.Linear
    

    def __init__(
        self,
        img_size: Union[int, Tuple[int]] = 224,
        patch_size: Union[int, Tuple[int]] = 16,
        in_chans: int = 3,
        num_classes: int = 0,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale=None,
        drop_rate=0.0,
        activation=jnn.gelu,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        patch_embed=PatchConvEmbed,
        *,
        key: Optional["jax.random.PRNGKey"] = None,
    ):
        """**Arguments:**

        - `img_size`: The size of the input image. Defaults to `(224, 224)`
        - `patch_size`: Size of the patch to construct from the input image. Defaults to `(16, 16)`
        - `in_chans`: Number of input channels. Defaults to `3`
        - `num_classes`: Number of classes in the classification task.
                         Also controls the final output shape `(num_classes,)`. If `num_classes=0`,
                         then the final layer is replaced with `nn.Identity`
        - `embed_dim`: The dimension of the resulting embedding of the patch. Defaults to `768`
        - `depth`: Number of `VitBlocks` in the network
        - `num_heads`: The number of equal parts to split the input along the `dim`
        - `mlp_ratio`: For computing hidden dimension of the `mlp`
        - `qkv_bias`: To add `bias` within the `qkv` computation
        - `qk_scale`: For scaling the `query` `value` computation
        - `drop_rate`: Dropout rate used within the `MlpProjection`
        - `attn_drop_rate`: Dropout rate used within the attention modules
        - `drop_path_rate`: Dropout rate used within `VitBlock`s
        - `norm_layer`: Normalisation applied to the intermediate outputs. Defaults to `equinox.nn.LayerNorm`
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
        initialisation. (Keyword only argument.)

        """
        super().__init__()
        if key is None:
            key = jrandom.PRNGKey(0)
        
        keys = jrandom.split(key, depth + 3)
        self.inference = False
        self.num_features = embed_dim
        self.patch_embed = patch_embed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            key=keys[0]
        )
        num_patches = self.patch_embed.num_patches


        self.params = Params([(1, embed_dim), (num_patches + 1, embed_dim)], key=keys[1])

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = jnp.linspace(0, drop_path_rate, depth)
        self.blocks = [
            _VitBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                act_layer=activation,
                norm_layer=norm_layer,
                key=keys[i + 2],
            )
            for i in range(depth)
        ]
        self.norm1 = norm_layer(embed_dim)

        self.norm2 = norm_layer(embed_dim)
        
        # Classifier head
        self.fc = nn.Identity() if num_classes == 0 else nn.Linear(embed_dim, num_classes, key=keys[-1])
    
    @property
    def cls_token(self):
        return self.params.weight[:1]

    @property
    def pos_embed(self):
        return self.params.weight[1:]

    def __call__(self, x: Array, *, key: "jax.random.PRNGKey") -> Array:
        """**Arguments:**

        - `x`: The input `JAX` array
        - `key`: Required parameter. Utilised by few layers such as `Dropout` or `DropPath`
        """
        keys = jrandom.split(key, len(self.blocks))
        x = self.patch_embed(x)
        x = jax.vmap(self.norm1)(x)
        x = jnp.concatenate([self.cls_token, x], axis=0) + self.pos_embed
        for key_, blk in zip(keys, self.blocks):
            x = blk(x, key=key_)
        
        x = jax.vmap(self.norm2)(x)
        return self.fc(x[0])

    def get_last_self_attention(self, x: Array, *, key: "jax.random.PRNGKey") -> Array:
        """**Arguments:**

        - `x`: The input `JAX` array
        - `key`: Utilised by few layers in the network such as `Dropout` or `BatchNorm`
        """
        if not self.inference:
            raise ValueError(
                "Model being evaluated outside inference mode. Try in inference mode."
            )
        keys = jrandom.split(key, len(self.blocks))
        x = self.patch_embed(x)
        x = jnp.concatenate([self.cls_token, x], axis=0) + self.pos_embed
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, key=keys[i])
            else:
                return blk(x, return_attention=True, key=key)


def vit_tiny(
    patch_size: str = 16,
    embed_dim: str = 192,
    depth: str = 12,
    num_heads: str = 3,
    mlp_ratio: str = 4,
    *,
    key: Optional["jax.random.PRNGKey"] = None,
    **kwargs,
):
    """**Arguments:**

    - `patch_size`: Size of the patch to construct from the input image
    - `embed_dim`: The dimension of the resulting embedding of the patch
    - `depth`: Number of `VitBlocks` in the network
    - `num_heads`: The number of equal parts to split the input along the `dim`
    - `mlp_ratio`: For computing hidden dimension of the `mlp`
    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
        initialisation. (Keyword only argument.)
    - `kwargs`: Parameters passed on to the `VisionTransformer`
    """
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        key=key,
        **kwargs,
    )

    return model


def vit_small(
    patch_size: int = 16,
    embed_dim: int = 384,
    depth: int = 12,
    num_heads: int = 6,
    mlp_ratio: int = 4,
    *,
    key: Optional["jax.random.PRNGKey"] = None,
    **kwargs,
):
    """**Arguments:**

    - `patch_size`: Size of the patch to construct from the input image
    - `embed_dim`: The dimension of the resulting embedding of the patch
    - `depth`: Number of `VitBlocks` in the network
    - `num_heads`: The number of equal parts to split the input along the `dim`
    - `mlp_ratio`: For computing hidden dimension of the `mlp`
    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
        initialisation. (Keyword only argument.)
    - `kwargs`: Parameters passed on to the `VisionTransformer`
    """
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        key=key,
        **kwargs,
    )

    return model


def vit_base(
    patch_size: int = 16,
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    mlp_ratio: int = 4,
    *,
    key: Optional["jax.random.PRNGKey"] = None,
    **kwargs,
):
    """**Arguments:**

    - `patch_size`: Size of the patch to construct from the input image
    - `embed_dim`: The dimension of the resulting embedding of the patch
    - `depth`: Number of `VitBlocks` in the network
    - `num_heads`: The number of equal parts to split the input along the `dim`
    - `mlp_ratio`: For computing hidden dimension of the `mlp`
    - `torch_weights`: A `Path` or `URL` for the `PyTorch` weights. Defaults to `None`
    - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
        initialisation. (Keyword only argument.)
    - `kwargs`: Parameters passed on to the `VisionTransformer`
    """
    model = VisionTransformer(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        key=key,
        **kwargs,
    )

    return model
