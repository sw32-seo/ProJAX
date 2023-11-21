# https://flax.readthedocs.io/en/latest/guides/flax_basics.html

from functools import partial
from typing import Any, Callable, Optional, Sequence

import flax
import jax
import optax
from flax import linen as nn
from flax import serialization
from flax.training import train_state
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

        return hidden_state


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

        for i in range(self.n_layer):
            hidden_states = GPT2Block(
                d_model=self.n_embd,
                d_ff=self.d_ff,
                n_head=self.n_head,
                w_init=self.w_init,
                b_init=self.b_init,
                drop_rate=self.drop_rate,
            )(hidden_states, attn_mask, training=training)

        hidden_states = nn.LayerNorm()(hidden_states)

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
        self.gpt2 = GPT2Model(n_layer=self.n_layer,
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

    def __call__(self, obs: jnp.ndarray, training=True) -> jnp.ndarray:
        hidden_states = self.gpt2(obs, training=training)
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

#
variables = {'params': state.params}
base_model, base_model_variables = model.bind(variables).gpt2.unbind()
