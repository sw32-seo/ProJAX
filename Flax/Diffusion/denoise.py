import functools
import os
import shutil
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import flax
import jax
import optax
import orbax.checkpoint
from absl import app, flags, logging
from flax import linen as nn
from flax import struct, traverse_util
from flax.core import freeze, unfreeze
from flax.training import train_state
from jax import lax
from jax import numpy as jnp
from jax import random
from jax.experimental import mesh_utils
from jax.lax import with_sharding_constraint
from jax.sharding import Mesh, NamedSharding, PartitionSpec


# Define a simple MLP model
class SimpleModule(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat)(x)
            if i < len(self.features) - 1:
                x = nn.relu(x)
        return x


# Define a forward diffusion process
def forward_diffusion(params, x, timesteps, noise):
    def body_fn(carry, _):
        params, x = carry
        rng, noise = noise
        rng, key = random.split(rng)
        noise = random.normal(key, x.shape)
        grad = jax.grad(lambda p, x: jnp.sum(p(x)))
        grad_params = grad(lambda p: jnp.sum(p(x)))(params)
        grad_x = grad(lambda x: jnp.sum(params(x)))(x)
        dx = grad_params + grad_x + noise
        x = x + 0.1 * dx
        return (params, x), None

    _, (params, x) = lax.scan(body_fn, (params, x), None, length=timesteps)
    return x
