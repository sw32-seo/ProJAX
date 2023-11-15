# https://flax.readthedocs.io/en/latest/guides/flax_basics.html

from typing import Any, Callable, Sequence

import flax
import jax
import optax
from flax import linen as nn
from flax import serialization
from jax import numpy as jnp
from jax import random

# Linear regression w/ Flax
model = nn.Dense(features=5)

# Initialize parameters
key1, key2 = random.split(random.PRNGKey(0))
x = random.normal(key1, (10, ))  # Dummy input
params = model.init(key2, x)
print(jax.tree_util.tree_map(lambda x: x.shape, params))

# output:   {'params': {'bias': (5,), 'kernel': (10, 5)}}

print(model.apply(params, x))

# Gradient descent
n_samples = 20
x_dim = 10
y_dim = 5

key = random.key(0)
k1, k2 = random.split(key)
W = random.normal(k1, (x_dim, y_dim))
b = random.normal(k2, (y_dim, ))
true_params = flax.core.freeze({'params': {'kernel': W, 'bias': b}})

key_sample, key_noise = random.split(key)
x_samples = random.normal(key_sample, (n_samples, x_dim))
y_samples = jnp.dot(x_samples,
                    W) + b + 0.1 * random.normal(key_noise, (n_samples, y_dim))
print('x_shape: ', x_samples.shape, 'y_shape:', y_samples.shape)


@jax.jit
def mse(params, x_batched, y_batched):
    # define square loss for a single (x, y) sample
    def squared_error(x, y):
        pred = model.apply(params, x)
        return jnp.inner(pred - y, pred - y) / 2.0

    return jnp.mean(jax.vmap(squared_error)(x_batched, y_batched), axis=0)


learning_rate = 0.3  # Gradient step size.
print('Loss for "true" W,b: ', mse(true_params, x_samples, y_samples))
loss_grad_fn = jax.value_and_grad(mse)


@jax.jit
def update_params(params, learning_rate, grads):
    params = jax.tree_util.tree_map(lambda p, g: p - learning_rate * g, params,
                                    grads)
    return params


for i in range(101):
    # Perform one gradient update.
    loss_val, grads = loss_grad_fn(params, x_samples, y_samples)
    params = update_params(params, learning_rate, grads)
    if i % 10 == 0:
        print(f'Loss step {i}: ', loss_val)

# Optimizing Optax
tx = optax.adam(learning_rate=learning_rate)
opt_state = tx.init(params)
loss_grad_fn = jax.value_and_grad(mse)

for i in range(101):
    loss_val, grads = loss_grad_fn(params, x_samples, y_samples)
    updates, opt_state = tx.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    if i % 10 == 0:
        print(f'Loss step {i}: ', loss_val)

# Serialization the result
# `Serialization` is saving model parameter by flax
bytes_output = serialization.to_bytes(params)
dict_output = serialization.to_state_dict(params)
print('Dict output')
print(dict_output)
print('Bytes output')
print(bytes_output)
