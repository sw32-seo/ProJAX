# Originally from https://github.com/deepmind/dm-haiku/blob/main/examples/mnist.py
"""MNIST classifier example"""

from typing import Generator, Mapping, Tuple

from absl import app
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_datasets as tfds
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
Batch = Mapping[str, np.ndarray]    # Mapping is dict-like which can use __getitem__


def net_fn(batch: Batch) -> jnp.ndarray:
    """Standard LeNet-300-100 MLP network."""
    x = batch["image"].astype(jnp.float32) / 255.
    mlp = hk.Sequential([
        hk.Flatten(),
        hk.Linear(300), jax.nn.relu,
        hk.Linear(100), jax.nn.relu,
        hk.Linear(10),
    ])
    return mlp(x)


def load_dataset(
        split: str,
        *,
        is_training: bool,
        batch_size: int,
) -> Generator[Batch, None, None]:
    """Loads the dataset as a generator of batches."""
    ds = tfds.load("mnist:3.*.*", split=split).cache().repeat()
    if is_training:
        ds = ds.shuffle(10 * batch_size, seed=0)
    ds = ds.batch(batch_size)
    return iter(tfds.as_numpy(ds))


def main(_):
    # Make the network and optimizer.
    net = hk.without_apply_rng(hk.transform(net_fn))
    opt = optax.adam(1e-3)

    # Training loss (cross-entropy).
    def loss(params: hk.Params, batch: Batch) -> jnp.ndarray:
        """Compute the loss of the network, including L2."""
        
