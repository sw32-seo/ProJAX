from typing import Any, Callable, Optional, Sequence, Tuple, Union

import flax
import jax
import optax
from flax import linen as nn
from jax import numpy as jnp
from jax import random


class ExplicitMLP(nn.Module):
    features: Sequence[int]

    def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.features]

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i < len(self.layers) - 1:
                x = nn.relu(x)
        return x


key1, key2 = random.split(random.key(1), 2)
x = random.uniform(key1, (4, 4))

model = ExplicitMLP(features=[3, 4, 5])
params = model.init(key2, x)
y = model.apply(params, x)

print('initialized parameter shapes:\n', jax.tree_map(jnp.shape, params))
print('output:\n', y)


class SimpleMLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat)(x)
            if i < len(self.features) - 1:
                x = nn.relu(x)
        return x


key1, key2 = random.split(random.key(1), 2)
x = random.uniform(key1, (4, 4))

model = SimpleMLP(features=[3, 4, 5])
params = model.init(key2, x)
y = model.apply(params, x)

print('initialized parameter shapes:\n', jax.tree_map(jnp.shape, params))
print('output:\n', y)


class BiasAdderWithRunningMean(nn.Module):
    decay: float = 0.99

    @nn.compact
    def __call__(self, x):
        is_initialized = self.has_variable('batch_stats', 'mean')
        ra_mean = self.variable('batch_stats', 'mean', lambda s: jnp.zeros(s),
                                x.shape[1:])
        bias = self.param('bias', lambda rng, shape: jnp.zeros(shape),
                          x.shape[1:])
        if is_initialized:
            ra_mean.value = self.decay * ra_mean.value + (
                1. - self.decay) * jnp.mean(x, axis=0, keepdims=True)

        return x - ra_mean.value + bias


key1, key2 = random.split(random.key(1), 2)
x = jnp.ones((10, 5))
model = BiasAdderWithRunningMean()
variables = model.init(key1, x)
print('initialized variables:\n', variables)
y, updated_states = model.apply(variables, x, mutable=['batch_stats'])
print('updated state:\n', updated_states)

for val in [1.0, 2.0, 3.0]:
    x = val * jnp.ones((10, 5))
    y, updated_states = model.apply(variables, x, mutable=['batch_stats'])
    old_state, params = flax.core.pop(variables, 'params')
    variables = flax.core.freeze({'params': params, **updated_states})
    print('updated state:\n', updated_states)

from functools import partial


@partial(jax.jit, static_argnums=(0, 1))
def update_step(tx, apply_fn, x, opt_state, params, state):

    def loss(params):
        y, updated_state = apply_fn({
            'params': params,
            **state
        },
                                    x,
                                    mutable=list(state.keys()))
        l = ((x - y)**2).sum()
        return l, updated_state

    (l, state), grads = jax.value_and_grad(loss, has_aux=True)(params)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params, state


x = jnp.ones((10, 5))
variables = model.init(random.key(0), x)
state, params = flax.core.pop(variables, 'params')
del variables
tx = optax.sgd(0.01)
opt_state = tx.init(params)

for _ in range(3):
    opt_state, params, state = update_step(tx, model.apply, x, opt_state,
                                           params, state)
    print('Updated state:', state)
