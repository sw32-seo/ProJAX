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

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
print(f"We have 8 fake JAX devices: {jax.devices()}")

# Create a mesh and annotate each axis with a name.
device_mesh = mesh_utils.create_device_mesh((2, 1))
print(f"Device mesh: {device_mesh}")

mesh = Mesh(devices=device_mesh, axis_names=('data', 'model'))
print(f"Mesh: {mesh}")


def mesh_sharding(pspec: PartitionSpec) -> NamedSharding:
    return NamedSharding(mesh, pspec)


class DotReluDot(nn.Module):
    depth: int
    dense_init: Callable = nn.initializers.xavier_normal()

    @nn.compact
    def __call__(self, x):
        y = nn.Dense(self.depth,
                     kernel_init=nn.with_partitioning(self.dense_init,
                                                      (None, 'model')),
                     use_bias=False)(x)
        y = nn.relu(y)
        # Force a local sharding annotation
        y = with_sharding_constraint(
            y, mesh_sharding(PartitionSpec('data', 'model')))

        W2 = self.param('W2',
                        nn.with_partitioning(self.dense_init, ('model', None)),
                        (self.depth, x.shape[-1]))

        z = jnp.dot(y, W2)
        # Force a local sharding annotation
        z = with_sharding_constraint(
            z, mesh_sharding(PartitionSpec('data', None)))
        return z, None


class MLP(nn.Module):
    num_layers: int
    depth: int
    use_scan: bool

    @nn.compact
    def __call__(self, x):
        if self.use_scan:
            x, _ = nn.scan(DotReluDot,
                           length=self.num_layers,
                           variable_axes={"params": 0},
                           split_rngs={'params': True},
                           metadata_params={nn.PARTITION_NAME:
                                            None})(self.depth)(x)
        else:
            for _ in range(self.num_layers):
                x, _ = DotReluDot(self.depth)(x)
        return x


# MLP hyperparameters
BATCH, LAYERS, DEPTH, USE_SCAN = 8, 4, 1024, False
# Create fake inputs.
x = jnp.ones((BATCH, DEPTH), jnp.float32)
# Initialize the model.
k = random.PRNGKey(0)

# Create an Optax optimizer.
optimizer = optax.adam(1e-3)
model = MLP(LAYERS, DEPTH, USE_SCAN)

# Specify input's sharding
x_sharding = mesh_sharding(PartitionSpec('data', None))
x = jax.device_put(x, x_sharding)
jax.debug.visualize_array_sharding(x)


# Specify output's sharding
def init_fn(key, x, model, optimizer):
    variables = model.init(key, x)
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer,
    )
    return state


abstract_variables = jax.eval_shape(
    functools.partial(init_fn, model=model, optimizer=optimizer), k, x)
state_sharding = nn.get_sharding(abstract_variables, mesh)
print(state_sharding)

# Compile the code
jit_init_fn = jax.jit(init_fn,
                      static_argnums=(2, 3),
                      in_shardings=(mesh_sharding(None), x_sharding),
                      out_shardings=state_sharding)

initialized_state = jit_init_fn(k, x, model, optimizer)

jax.debug.visualize_array_sharding(
    initialized_state.params['DotReluDot_0']['Dense_0']['kernel'].value)
jax.debug.visualize_array_sharding(
    initialized_state.params['DotReluDot_0']['W2'].value)

# Inspect the Module output
print(type(initialized_state.params['DotReluDot_0']['Dense_0']['kernel']))
print(type(
    initialized_state.params['DotReluDot_0']['Dense_0']['kernel'].value))
print(initialized_state.params['DotReluDot_0']['Dense_0']['kernel'].names)
print(
    initialized_state.params['DotReluDot_0']['Dense_0']['kernel'].value.shape)

initialized_state.params['DotReluDot_0']['Dense_0']['kernel'].value.sharding
print(initialized_state.step)
initialized_state.step.sharding

diff = jax.tree_map(lambda a, b: a - b,
                    initialized_state.params['DotReluDot_0'],
                    initialized_state.params['DotReluDot_0'])
print(jax.tree_map(jnp.shape, diff))
diff_array = diff['Dense_0']['kernel'].value
print(type(diff_array))
print(diff_array.shape)


# Compile the train step and inference
@functools.partial(jax.jit,
                   in_shardings=(state_sharding, x_sharding),
                   out_shardings=state_sharding)
def train_step(state, x):

    def loss_unrolled(params):
        y = state.apply_fn({'params': params}, x)
        return y.sum()

    grad_fn = jax.grad(loss_unrolled)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


with mesh:
    new_state = train_step(initialized_state, x)

print('Sharding of Weight 1:')
jax.debug.visualize_array_sharding(
    initialized_state.params['DotReluDot_0']['Dense_0']['kernel'].value)
print('Sharding of Weight 2:')
jax.debug.visualize_array_sharding(
    initialized_state.params['DotReluDot_0']['W2'].value)


@functools.partial(jax.jit,
                   in_shardings=(state_sharding, x_sharding),
                   out_shardings=x_sharding)
def apply_fn(state, x):
    return state.apply_fn({'params': state.params}, x)


with mesh:
    y = apply_fn(new_state, x)
print(type(y))
print(y.dtype)
print(y.shape)
jax.debug.visualize_array_sharding(y)
