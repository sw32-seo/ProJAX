# https://flax.readthedocs.io/en/latest/guides/flax_basics.html
import os
import shutil
from functools import partial
from typing import Any, Callable, Optional, Sequence

import flax
import jax
import numpy as np
import optax
import orbax.checkpoint
from flax import linen as nn
from flax import serialization, traverse_util
from flax.training import orbax_utils, train_state
from jax import numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec


class GPT2Block(nn.Module):
    d_model: int
    d_ff: int
    n_head: int
    w_init: nn.initializers.Initializer
    b_init: nn.initializers.Initializer
    drop_rate: float = 0.3

    @nn.compact
    def __call__(self,
                 obs: jnp.ndarray,
                 mask: jnp.ndarray,
                 training=True) -> jnp.ndarray:
        """GPT2Block
        Args:
            obs (jnp.ndarray): observation
            mask: attention mask
        """
        residual = obs
        hidden_state = nn.LayerNorm()(obs)

        # for the case to see the attention map
        attn_out = nn.MultiHeadDotProductAttention(
            num_heads=self.n_head,
            dropout_rate=self.drop_rate,
            deterministic=not training,
        )(inputs_q=hidden_state, mask=mask)

        # residual connection
        hidden_state = attn_out + residual

        residual = hidden_state

        hidden_state = nn.LayerNorm()(hidden_state)
        linear_out = nn.Dense(features=self.d_ff,
                              kernel_init=self.w_init,
                              bias_init=self.b_init)(hidden_state)
        linear_out = nn.gelu(linear_out)
        linear_out = nn.Dense(features=self.d_model,
                              kernel_init=self.w_init,
                              bias_init=self.b_init)(linear_out)
        linear_out = nn.Dropout(self.drop_rate,
                                deterministic=not training)(linear_out)

        # residual connection
        hidden_state = linear_out + residual

        return hidden_state, mask, training


class GPT2Model(nn.Module):
    n_layer: int
    n_embd: int
    d_ff: int
    n_head: int
    vocab_size: int
    drop_rate: float
    w_init: nn.initializers.Initializer = jax.nn.initializers.truncated_normal(
        0.02)
    b_init: nn.initializers.Initializer = jax.nn.initializers.zeros

    @nn.compact
    def __call__(self, obs: jnp.ndarray, training=True) -> jnp.ndarray:

        position_id = jnp.arange(obs.shape[-1])
        pad_attn_mask = nn.attention.make_attention_mask(obs != 0, obs != 0)
        causal_mask = nn.attention.make_causal_mask(obs)

        attn_mask = nn.attention.combine_masks(pad_attn_mask, causal_mask)

        input_embd = nn.Embed(num_embeddings=self.vocab_size,
                              features=self.n_embd,
                              name='Input_Embd')(obs)
        position_embd = nn.Embed(num_embeddings=300,
                                 features=self.n_embd,
                                 name='Learned_PosEmbd')(position_id)
        hidden_states = input_embd + position_embd

        hidden_states = nn.Dropout(rate=self.drop_rate,
                                   deterministic=not training)(hidden_states)

        for _ in range(self.n_layer):
            hidden_states, _, _ = GPT2Block(
                d_model=self.n_embd,
                d_ff=self.d_ff,
                n_head=self.n_head,
                w_init=self.w_init,
                b_init=self.b_init,
                drop_rate=self.drop_rate,
            )(hidden_states, attn_mask, training=training)

        return hidden_states


class FlaxGPT2BaseModel(nn.Module):
    n_layer: int
    n_embd: int
    d_ff: int
    n_head: int
    vocab_size: int
    drop_rate: float
    w_init: nn.initializers.Initializer = nn.initializers.truncated_normal(
        stddev=0.02)
    b_init: nn.initializers.Initializer = nn.initializers.zeros
    pad_id: int = 42

    def setup(self):
        self.input_embd = nn.Embed(num_embeddings=self.vocab_size,
                                   features=self.n_embd,
                                   name='Input_Embd')
        self.position_embd = nn.Embed(num_embeddings=300,
                                      features=self.n_embd,
                                      name='Learned_PosEmbd')
        self.dropout = nn.Dropout(rate=self.drop_rate)

        self.gpt2_blocks = nn.Sequential([
            GPT2Block(d_model=self.n_embd,
                      d_ff=self.d_ff,
                      n_head=self.n_head,
                      w_init=self.w_init,
                      b_init=self.b_init,
                      drop_rate=self.drop_rate) for _ in range(self.n_layer)
        ])

    def __call__(self, obs: jnp.ndarray, training=True):
        """
        Compute the output of the block.

        Args:
            obs (jnp.ndarray): Input observations.
            training (bool, optional): If True, apply dropout. Defaults to True.

        Returns:
            jnp.ndarray: Output of the block.
        """
        position_id = jnp.arange(obs.shape[-1])

        # Create pad mask when obs element is <pad>
        pad_attn_mask = nn.make_attention_mask(obs != self.pad_id, obs
                                               != self.pad_id)
        causal_attn_mask = nn.make_causal_mask(obs)
        attn_mask = nn.combine_masks(pad_attn_mask, causal_attn_mask)

        input_embd = self.input_embd(obs)
        position_embd = self.position_embd(position_id)
        hidden_states = input_embd + position_embd

        hidden_states = self.dropout(hidden_states, deterministic=not training)

        hidden_states, _, _ = self.gpt2_blocks(hidden_states, attn_mask,
                                               training)

        return hidden_states


class GPT2LMHead(nn.Module):
    n_layer: int
    n_embd: int
    d_ff: int
    n_head: int
    vocab_size: int
    drop_rate: float
    n_output: int
    w_init: nn.initializers.Initializer = jax.nn.initializers.truncated_normal(
        0.02)
    b_init: nn.initializers.Initializer = jax.nn.initializers.zeros

    def setup(self):
        self.gpt2_base = FlaxGPT2BaseModel(n_layer=self.n_layer,
                                           n_embd=self.n_embd,
                                           d_ff=self.d_ff,
                                           n_head=self.n_head,
                                           vocab_size=self.vocab_size,
                                           drop_rate=self.drop_rate,
                                           w_init=self.w_init,
                                           b_init=self.b_init)

        self.lm_head = nn.Dense(features=self.n_output,
                                kernel_init=self.w_init,
                                bias_init=self.b_init)

        self.layer_norm = nn.LayerNorm()

    def __call__(self, obs: jnp.ndarray, training=True) -> jnp.ndarray:
        hidden_states = self.gpt2_base(obs, training=training)
        hidden_states = self.layer_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits


root_key = jax.random.PRNGKey(seed=0)
main_key, params_key, dropout_key = jax.random.split(key=root_key, num=3)

x = jax.random.randint(main_key, (64, 10), 0, 10)
model = GPT2LMHead(n_layer=2,
                   n_embd=16,
                   d_ff=32,
                   n_head=2,
                   vocab_size=41,
                   drop_rate=0.5,
                   n_output=41)
variables = model.init(params_key, x, training=False)
params = variables['params']
pred = model.apply({'params': params},
                   x,
                   training=True,
                   rngs={'dropout': dropout_key})
print(pred.shape)


class TrainState(train_state.TrainState):
    key: jax.Array


state = TrainState.create(
    apply_fn=model.apply,
    params=params,
    key=dropout_key,
    tx=optax.adam(0.01),
)


@jax.jit
def train_step(state: TrainState, batch, dropout_key):
    dropout_train_key = jax.random.fold_in(key=dropout_key, data=state.step)

    def loss_fn(params):
        logits = state.apply_fn(
            {'params': params},
            obs=batch['input'][:, :-1],
            training=True,
            rngs={'dropout': dropout_train_key},
        )
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['input'][:, 1:])
        loss = loss.mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


batch = jnp.empty((0, 10), dtype=jnp.int32)
for i in range(32):
    batch = jnp.concatenate([batch, jnp.arange(i, i + 10).reshape(1, -1)])
peusudo_batch = {'input': batch}
loss = jnp.inf

for _ in range(100):
    state, loss = train_step(state, peusudo_batch, dropout_key)
    print(loss)

# sample a sequence from the model
main_key, sample_key = jax.random.split(main_key)
for j in range(10):
    sample = jnp.array([j])
    for i in range(10):
        pred = model.apply({'params': state.params}, sample, training=False)
        pred = jnp.argmax(pred, axis=-1)[-1]
        sample = jnp.concatenate([sample, jnp.array([pred])])
    print(sample)

# Transfer learning
variables = {'params': state.params}
base_model, base_model_variables = model.bind(variables).gpt2_base.unbind()


# Creating a classifier
class Classifier(nn.Module):
    num_classes: int
    backbone: nn.Module

    @nn.compact
    def __call__(self, obs):
        hidden_states = self.backbone(obs, training=False)[:, -1]
        logits = nn.Dense(features=self.num_classes,
                          name='head',
                          kernel_init=nn.zeros)(hidden_states)
        return logits


num_classes = 3
model = Classifier(num_classes=num_classes, backbone=base_model)
variables = model.init(params_key, x)
params = variables['params']
params['backbone'] = base_model_variables['params']

partition_optimizers = {
    'trainable': optax.adam(5e-3),
    'frozen': optax.set_to_zero()
}
param_partitions = traverse_util.path_aware_map(
    lambda path, v: 'frozen' if 'backbone' in path else 'trainable', params)

tx = optax.multi_transform(partition_optimizers, param_partitions)

flat = list(traverse_util.flatten_dict(param_partitions).items())
traverse_util.unflatten_dict(dict(flat[:2] + flat[2:]))

# Creating the `TrainState` as usual:
state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx,
)

config = {'dimensions': np.array([5, 3])}

ckpt = {'model': state, 'config': config, 'data': [x]}

# Save and load checkpoints with orbax
orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(ckpt)
if os.path.exists('/tmp/ckpt/'):
    shutil.rmtree('/tmp/ckpt/')

orbax_checkpointer.save('/tmp/ckpt/orbax/single_save',
                        ckpt,
                        save_args=save_args)

options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=2, create=True)
checkpoint_manager = orbax.checkpoint.CheckpointManager(
    '/tmp/ckpt/orbax/managed', orbax_checkpointer, options=options)

for step in range(5):
    checkpoint_manager.save(step, ckpt, save_kwargs={'save_args': save_args})

print(os.listdir('/tmp/ckpt/orbax/managed'))

# Restore the checkpoints with orbax
raw_restored = orbax_checkpointer.restore('/tmp/ckpt/orbax/single_save')
print(raw_restored)

raw_managed_restored = checkpoint_manager.restore(4)

# Restore with custom dataclasses
# you should first provide an example pytree to let Orbax or Flax know exactly which structure to restore to.

empty_state = train_state.TrainState.create(
    apply_fn=model.apply,
    params=jax.tree_map(np.zeros_like, variables['params']),
    tx=tx,
)

empty_config = {'dimensions': np.array([0, 0])}
target = {
    'model': empty_state,
    'config': empty_config,
    'data': [jnp.zeros_like(x)]
}
state_restored = orbax_checkpointer.restore('/tmp/ckpt/orbax/single_save',
                                            item=target)

print(state_restored)

# Restore when checkpoint structures differ


class CustomTrainState(train_state.TrainState):
    batch_stats: Any = None


custom_state = CustomTrainState.create(apply_fn=model.apply,
                                       params=params,
                                       tx=tx,
                                       batch_stats=np.arange(10))

empty_config = {'dimensions': np.array([0, 0])}

custom_ckpt = {'model': custom_state, 'config': empty_config, 'data': [x]}
# Use a custom_state to read the old `TrainState` checkpoint
custom_target = {
    'model': custom_state,
    'config': None,
    'data': [jnp.zeros_like(x)]
}

# save it on Orbax
custom_save_args = orbax_utils.save_args_from_target(custom_ckpt)
checkpoint_manager.save(5,
                        custom_ckpt,
                        save_kwargs={'save_args': custom_save_args})

## Scenario 1: When a reference object is partial.
restored = checkpoint_manager.restore(5, items=target)
assert not hasattr(restored, 'batch_stats')
assert type(restored['model']) == train_state.TrainState

## Scenario 2: When a checkpoint is partial
try:
    checkpoint_manager.restore(4, items=custom_target)
except KeyError as e:
    print(f'KeyError when target state has an unmentioned field: {e}')
    print('')

# Step 4 is an original `TrainState`, without the `batch_stats`
custom_restore_args = orbax_utils.restore_args_from_target(custom_target)
restored = checkpoint_manager.restore(4,
                                      items=custom_target,
                                      restore_kwargs={
                                          'transforms': {},
                                          'restore_args': custom_restore_args
                                      })
assert type(restored['model']) == CustomTrainState
np.testing.assert_equal(restored['model'].batch_stats,
                        custom_target['model'].batch_stats)
restored

# Asynchronized checkpointing
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
jax.distributed.initialize("localhost:8889", num_processes=1, process_id=0)

async_checkpointer = orbax.checkpoint.AsyncCheckpointer(
    orbax.checkpoint.PyTreeCheckpointHandler(), timeout_secs=50)

# Save your job:
async_checkpointer.save('/tmp/flax_ckpt/orbax/single_save_async',
                        ckpt,
                        save_args=save_args)
# ... Continue with your work...

# ... Until a time when you want to wait until the save completes:
async_checkpointer.wait_until_finished()
print(
    async_checkpointer.restore('/tmp/flax_ckpt/orbax/single_save_async',
                               item=target))

# if you are using Orbax `CheckpointManager`, just pass in the async_checkpointer when initializing it.
async_checkpoint_manager = orbax.checkpoint.CheckpointManager(
    '/tmp/flax_ckpt/orbax/managed_async', async_checkpointer, options)
async_checkpoint_manager.wait_until_finished()

# Multi-host/multi-process checkpointing
## Create an array sharded across multiple devices.
mesh_shape = (4, 2)
devices = np.asarray(jax.devices()).reshape(*mesh_shape)
mesh = jax.sharding.Mesh(devices, ('x', 'y'))

mp_array = jax.device_put(
    np.arange(8 * 2).reshape(8, 2), NamedSharding(mesh,
                                                  PartitionSpec('x', 'y')))

# Make it a pytree.
mp_ckpt = {'model': mp_array}
async_checkpoint_manager.save(0, mp_ckpt)
async_checkpoint_manager.wait_until_finished()
