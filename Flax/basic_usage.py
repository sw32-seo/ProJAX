import flax.linen as nn
import jax.numpy as jnp
import jax


class MLP(nn.Module):
    out_dims: int

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(self.out_dims)(x)
        return x


model = MLP(out_dims=10)

x = jnp.empty((4, 28, 28, 1))
params = model.init(jax.random.PRNGKey(42), x)
y = model.apply(params, x)

print(y.shape)  # (4, 10)
