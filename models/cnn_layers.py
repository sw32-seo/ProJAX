from typing import Any, Callable, Sequence, Optional, NewType
from jax import lax, vmap, numpy as jnp
import flax
from flax import linen as nn


class ResDownBlock(nn.Module):
    """Single ResBlock w/ downsample"""
    dim_out: Any = 64

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        f_x = nn.relu(nn.GroupNorm(self.dim_out)(x))
        x = nn.Conv(features=self.dim_out, kernel_size=(1, 1), strides=(2, 2))(x)
        f_x = nn.Conv(features=self.dim_out, kernel_size=(3, 3), strides=(2, 2))(f_x)
        f_x = nn.relu(nn.GroupNorm(self.dim_out)(f_x))
        f_x = nn.Conv(features=self.dim_out, kernel_size=(3, 3))(f_x)
        x = f_x + x
        return x


class ConcatConv2D(nn.Module):
    """Concat dynamics to hidden layer"""
    dim_out: Any = 64

    @nn.compact
    def __call__(self, inputs, t):
        """inputs is batch of images in [B, H, W, C]"""
        x = inputs
        tt = jnp.ones_like(x[:, :, :, :1]) * t
        ttx = jnp.concatenate([tt, x], -1)
        return nn.Conv(features=self.dim_out, kernel_size=(3, 3))(ttx)

