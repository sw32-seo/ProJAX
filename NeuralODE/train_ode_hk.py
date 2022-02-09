import os
import jax
import jax.numpy as jnp
import haiku as hk
import tensorflow as tf
import tensorflow_datasets as tfds
import optax
from absl import app, logging

from neuralode_hk import ResDownBlock, ODEBlock, ODEfunc, PostODE, conv1x1

# For me, tfds and jax struggles to get GPU memory. Below will prohibit using GPU memory by TF.
tf.config.experimental.set_visible_devices([], "GPU")

# TODO: Add absl argument for hyperparameters


def load_dataset(
        split: str,
        *,
        is_training: bool,
        batch_size: int,
):
    """Loads the dataset as a generator of batches."""
    ds = tfds.load("mnist:3.*.*", split=split).cache().repeat()
    if is_training:
        ds = ds.shuffle(10 * batch_size, seed=0)
    ds = ds.batch(batch_size)
    return iter(tfds.as_numpy(ds))


def _foward_fn_node(batch):
    x = batch["image"].astype(jnp.float32) / 255.
    conv = hk.Conv2D(64, 3, 1)
    module1 = ResDownBlock(64, 64, stride=2, downsample=conv1x1(64, 2))
    module2 = ResDownBlock(64, 64, stride=2, downsample=conv1x1(64, 2))
    feature_layer = ODEBlock(dim=64, tol=1e-3)
    fc_layers = PostODE(64, 10)

    out = conv(x)
    out = module1(out)
    out = module2(out)
    out = feature_layer(out)
    out = fc_layers(out)

    return out


def main(_):
    odenet = hk.without_apply_rng(hk.transform(_foward_fn_node))
    opt = optax.adam(learning_rate=1e-4)

    # Training loss (cross-entropy).
    @jax.jit
    def loss(params, batch):
        """Compute the loss of the network, including L2."""
        logits = odenet.apply(params, batch)
        labels = jax.nn.one_hot(batch["label"], 10)

        l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))    # l2_norm of parameters
        softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
        softmax_xent /= labels.shape[0]

        return softmax_xent + 1e-4 * l2_loss

    @jax.jit
    def accuracy(params, batch):
        predictions = odenet.apply(params, batch)
        return jnp.mean(jnp.argmax(predictions, axis=-1) == batch["label"])

    @jax.jit
    def update(params, opt_state, batch):
        grads = jax.grad(loss)(params, batch)
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, opt_state

    @jax.jit
    def ema_update(params, avg_params):
        return optax.incremental_update(params, avg_params, step_size=0.001)

    # Make datasets.
    train = load_dataset("train", is_training=True, batch_size=128)
    train_eval = load_dataset("train", is_training=False, batch_size=10000)
    test_eval = load_dataset("test", is_training=False, batch_size=10000)

    # Initialize network and optimizer; note we draw an input to get shapes.
    params = avg_params = odenet.init(rng=jax.random.PRNGKey(42), batch=next(train))
    opt_state = opt.init(params)

    # Train/eval loop.
    for step in range(10001):
        if step % 1000 == 0:
            # Periodically evaluate classification accuracy on train & test sets.
            train_accuracy = accuracy(avg_params, next(train_eval))
            test_accuracy = accuracy(avg_params, next(test_eval))
            train_loss = loss(params, next(train_eval))
            train_loss = jax.device_get(train_loss)
            train_accuracy, test_accuracy = jax.device_get(
                (train_accuracy, test_accuracy))

            logging.info(f"Step {step}] Train / Test accuracy / Train loss: "
                         f"{train_accuracy:.3f} / {test_accuracy:.3f} / {train_loss:.3f}.")

        # Do SGD on a batch of training examples.
        params, opt_state = update(params, opt_state, next(train))
        avg_params = ema_update(params, avg_params)


if __name__ == '__main__':
    app.run(main)
