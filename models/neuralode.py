from typing import Any, Callable, Sequence, Optional, NewType
from jax import lax, vmap, random, numpy as jnp
from experimental.ode import odeint
import flax
from flax import linen as nn
from .cnn_layers import ResDownBlock, ConcatConv2D


# Define Neural ODE for mnist example.
class ODEfunc(nn.Module):
    """ODE function which replace ResNet"""
    dim_out: Any = 64

    @nn.compact
    def __call__(self, inputs, t):

        x = inputs
        out = nn.GroupNorm(self.dim_out)(x)
        out = nn.relu(out)
        out = ConcatConv2D(self.dim_out)(out, t)
        out = nn.GroupNorm(self.dim_out)(out)
        out = nn.relu(out)
        out = ConcatConv2D(self.dim_out)(out, t)
        out = nn.GroupNorm(self.dim_out)(out)

        return out


class ODEBlock(nn.Module):
    """ODE block which contains odeint"""
    tol: Any = 1.

    @nn.compact
    def __call__(self, inputs, params):
        ode_func = ODEfunc()
        ode_func_apply = lambda x, t: ode_func.apply(variables={'params': params}, inputs=x, t=t)
        states, nfe = odeint(ode_func_apply,
                             inputs, jnp.array([0., 1.]),
                             rtol=self.tol, atol=self.tol)
        return states[-1], nfe


class FullODENet(nn.Module):
    """Full ODE net which contains two downsampling layers, ODE block and linear classifier.
       From Neural ODE paper"""
    dim_out: int = 64
    n_cls: int = 10
    tol: Any = 1.

    @nn.compact
    def __call__(self, inputs):
        x = inputs
        x = nn.Conv(features=self.dim_out, kernel_size=(3, 3))(x)
        x = ResDownBlock()(x)
        x = ResDownBlock()(x)

        ode_func = ODEfunc()
        init_fn = lambda rng, x: ode_func.init(random.split(rng)[-1], x, 0.)['params']
        ode_func_params = self.param('ode_func', init_fn, jnp.ones_like(x))
        x, nfe = ODEBlock(tol=self.tol)(x, ode_func_params)

        x = nn.GroupNorm(self.dim_out)(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, (1, 1))

        x = x.reshape((x.shape[0], -1))     # flatten

        x = nn.Dense(features=self.n_cls)(x)
        x = nn.log_softmax(x)

        return x, nfe
