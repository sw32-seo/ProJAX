import os
import shutil
from functools import partial
from typing import Any, Callable, Optional, Sequence

import flax
import jax
import optax
import orbax.checkpoint
from flax import linen as nn
from flax import serialization, traverse_util
from flax.training import orbax_utils, train_state
from jax import numpy as jnp


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
        self.position_embd = nn.Embed(num_embeddings=2048,
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

main_key, sub_key = jax.random.split(key=main_key, num=2)

x = jax.random.randint(sub_key, (64, 10), 0, 10)
model = GPT2LMHead(n_layer=2,
                   n_embd=16,
                   d_ff=32,
                   n_head=2,
                   vocab_size=41,
                   drop_rate=0.5,
                   n_output=41)
variables = model.init(params_key, x, training=False)
params = variables['params']


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


def rollout(state, max_step):
    """Rollout function using jax.lax.scan
    Args:
        state (TrainState): TrainState
        max_step (int): max rollout step
    """

    def body_fn(carry, x):
        state, input_seqs = carry
        sub_key = jax.random.fold_in(state.key, x)
        logits = state.apply_fn({'params': state.params},
                                input_seqs,
                                training=False)
        next_token = jax.random.categorical(sub_key, logits[:, x], axis=-1)
        input_seqs = input_seqs.at[:, x + 1].set(next_token)
        return (state, input_seqs), next_token

    # pass max_step directly as length and create a dummy sequence of the correct length
    key, subkey = jax.random.split(state.key)
    (_,
     input_seqs), x = jax.lax.scan(body_fn,
                                   (state, jnp.zeros(
                                       (10, max_step), jnp.int32)),
                                   jnp.arange(max_step))
    return input_seqs, x


batch = jnp.empty((0, 10), dtype=jnp.int32)
for i in range(32):
    batch = jnp.concatenate([batch, jnp.arange(i, i + 10).reshape(1, -1)])
peusudo_batch = {'input': batch}
loss = jnp.inf

for _ in range(100):
    state, loss = train_step(state, peusudo_batch, dropout_key)
    print(loss)

main_key, sample_key = jax.random.split(main_key)
for j in range(10):
    sample = jnp.array([j])
    for i in range(10):
        pred = model.apply({'params': state.params}, sample, training=False)
        pred = jnp.argmax(pred, axis=-1)[-1]
        sample = jnp.concatenate([sample, jnp.array([pred])])
    print(sample)

jit_rollout = jax.jit(rollout, static_argnums=(1, ))
a, x = jit_rollout(state, 10)
print(a)
