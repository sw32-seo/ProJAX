# https://flax.readthedocs.io/en/latest/guides/flax_basics.html

from typing import Any, Callable, Optional, Sequence

import flax
import jax
import optax
from flax import linen as nn
from flax import serialization
from jax import numpy as jnp
from jax import random


class GPT2Block(nn.Module):
    d_ff: int
    n_head: int
    w_init: nn.initializers.Initializer
    b_init: nn.initializers.Initializer
    drop_rate: float = 0.3
    deterministic: bool = True

    @nn.compact
    def __call__(self, obs: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
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
            dropout_rate=self.dropout_rate,
            deterministic=self.deterministic,
        )(inputs_q=hidden_state, mask=mask)

        # residual connection
        hidden_state = attn_out + residual

        residual = hidden_state

        hidden_state = nn.LayerNorm()(hidden_state)
        linear_out = nn.Dense(features=self.d_ff,
                              kernel_init=self.w_init,
                              bias_init=self.b_init)(hidden_state)
        linear_out = nn.gelu(linear_out)
        linear_out = nn.Dropout(rate=self.dropout_rate)(
            linear_out, deterministic=self.deterministic)

        # residual connection
        hidden_state = linear_out + residual

        return hidden_state


class GPT2Head(nn.Module):
    n_layer: int
    n_embd: int
    d_ff: int
    n_head: int
    vocab_size: int
    w_init: nn.initializers.Initializer = jax.nn.initializers.truncated_normal(
        0.02)
    b_init: nn.initializers.Initializer = jax.nn.initializers.zeros
    drop_rate: float = 0.3
    deterministic: bool = True

    @nn.compact
    def __call__(self, obs: jnp.ndarray, is_training=True) -> jnp.ndarray:
        position_id = jnp.arange(obs.shape[-1])
        pad_attn_mask = nn.attention.make_attention_mask(obs != 0, obs != 0)
        causal_mask = nn.attention.make_causal_mask(obs)

        attn_mask = nn.attention.combine_masks(pad_attn_mask, causal_mask)

        input_embd = nn.Embed(num_embeddings=self.vocab_size,
                              features=self.n_embd)(obs)
        position_embd = nn.Embed(num_embeddings=300,
                                 features=self.n_embd)(position_id)
        total_embd = input_embd + position_embd

        hidden_states = nn.Dropout(self.drop_rate)(
            total_embd, deterministic=not is_training)

        for i in range(self.n_layer):
            hidden_states = GPT2Block(d_ff=self.d_ff,
                                      n_head=self.n_head,
                                      w_init=self.w_init,
                                      b_init=self.b_init,
                                      drop_rate=self.drop_rate,
                                      deterministic=True)(hidden_states,
                                                          attn_mask)

        return hidden_states


key1, key2 = random.split(random.PRNGKey(0))
x = random.randint(key1, (10, ), 0, 10)
model = GPT2Head(n_layer=2, n_embd=16, d_ff=32, n_head=2, vocab_size=10)
params = model.init(key2, x)
print(jax.tree_util.tree_map(lambda x: x.shape, params))
print(model.apply(params, x))
