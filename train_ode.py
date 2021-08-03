from functools import partial
import jax
from typing import Any, Callable, Sequence, Optional, NewType
from jax import lax, random, vmap, numpy as jnp
import flax
from flax.training import train_state
import optax
import tensorflow_datasets as tfds
import numpy as np
from tqdm import tqdm
import os
from models import FullODENet


# Define loss
@jax.jit
def cross_entropy_loss(logits, labels):
    one_hot_labels = jax.nn.one_hot(labels, num_classes=10)
    return -jnp.mean(jnp.sum(one_hot_labels * logits, axis=-1))


# Metric computation
def compute_metrics(logits, labels, nfe):
    loss = cross_entropy_loss(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        'loss': loss,
        'accuracy': accuracy,
        'nfe': nfe
    }
    return metrics


def get_datasets():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder('mnist')
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split='train', batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split='test', batch_size=-1))
    train_ds['image'] = jnp.float32(train_ds['image']) / 255.
    test_ds['image'] = jnp.float32(test_ds['image']) / 255.
    return train_ds, test_ds


def create_train_state(rng, learning_rate, tol):
    """Creates initial 'TrainState'."""
    odenet = FullODENet(tol=tol)
    params = odenet.init(rng, jnp.ones([1, 28, 28, 1]))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=odenet.apply, params=params, tx=tx
    )


# Training step
@partial(jax.jit, static_argnums=(2,))
def train_step(state, batch, tol):
    """Train for a single step."""
    def loss_fn(params):
        logits, nfe = FullODENet(tol=tol).apply({'params': params}, batch['image'])
        loss = cross_entropy_loss(logits=logits, labels=batch['label'])
        return (loss, (logits, nfe))
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits_nfe), grads = grad_fn(state.params)
    logits, nfe = logits_nfe
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=batch['label'], nfe=nfe)
    return state, metrics


# Evaluation step
@partial(jax.jit, static_argnums=(2,))
def eval_step(params, batch, tol):
    logits, nfe = FullODENet(tol=tol).apply({'params': params}, batch['image'])
    return compute_metrics(logits=logits, labels=batch['label'], nfe=nfe)


# Train function
def train_epoch(state, train_ds, batch_size, epoch, rng, tol):
    """Train for a single epoch"""
    train_ds_size = len(train_ds['image'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds['image']))
    perms = perms[:steps_per_epoch * batch_size]    # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    batch_metrics = []
    for perm in tqdm(perms):
        batch = {k: v[perm, ...] for k, v in train_ds.items()}
        state, metrics = train_step(state, batch, tol)
        batch_metrics.append(metrics)

        # compute mean of metrics across each batch in epoch.
        batch_metrics_np = jax.device_get(batch_metrics)
        epoch_metrics_np = {
            k: np.mean([metrics[k] for metrics in batch_metrics_np])
            for k in batch_metrics_np[0]
        }
    print('train epoch: %d, loss: %.4f, accuracy: %.2f, forward nfe: %.2f' % (
        epoch, epoch_metrics_np['loss'], epoch_metrics_np['accuracy'] * 100, epoch_metrics_np['nfe']
    ))

    return state


# Eval function
def eval_model(params, test_ds, tol):
    metrics = eval_step(params, test_ds, tol)
    metrics = jax.device_get(metrics)
    summary = jax.tree_map(lambda x: x.item(), metrics)
    return summary['loss'], summary['accuracy']


def train_and_evaluate(learning_rate, n_epoch, batch_size, tol):
    train_ds, test_ds = get_datasets()
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)

    state = create_train_state(init_rng, learning_rate, tol)
    del init_rng  # Must not be used anymore.

    for epoch in range(1, n_epoch + 1):
        rng, input_rng = jax.random.split(rng)
        state = train_epoch(state, train_ds, batch_size, epoch, input_rng, tol)
        test_loss, test_accuracy = eval_model(state.params, test_ds, tol)
        print(' test epoch: %d, loss: %.2f, accuracy: %.2f' % (
            epoch, test_loss, test_accuracy * 100
        ))


if __name__ == '__main__':
    train_and_evaluate(0.0001, 10, 128, 1e-5)
