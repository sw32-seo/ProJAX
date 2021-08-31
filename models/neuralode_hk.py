import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint


class ResDownBlock(hk.Module):
    def __init__(self, dim_out=64):
        super(ResDownBlock, self).__init__()
        self.dim_out = dim_out
        self.groupnorm = hk.GroupNorm(self.dim_out)

    def __call__(self, x):
        f_x = jax.nn.relu(self.groupnorm(x))
        x = hk.Conv2D(self.dim_out, 1, 2)(x)
        f_x = hk.Conv2D(self.dim_out, 3, 2)(f_x)
        f_x = jax.nn.relu(hk.GroupNorm(self.dim_out)(f_x))
        f_x = hk.Conv2D(self.dim_out, 3, 1)(f_x)
        x = f_x + x
        return x


class ConcatConv2D(hk.Module):
    """Concat dynamics to hidden layer"""
    def __init__(self, dim_out=64):
        super(ConcatConv2D, self).__init__()
        self.dim_out = dim_out

    def __call__(self, x, t):
        """x is batch of images in [B, H, W, C]"""
        tt = jnp.ones_like(x[:, :, :, :1]) * t
        ttx = jnp.concatenate([tt, x], -1)
        return hk.Conv2D(self.dim_out, 3, 1)(ttx)


class ODEfunc(hk.Module):
    """ODE function which replace ResNet"""
    def __init__(self, dim_out=64):
        super(ODEfunc, self).__init__()
        self.dim_out = dim_out

    def __call__(self, x, t):
        # nfe = hk.get_state("nfe", shape=[], dtype=jnp.int32, init=jnp.zeros)
        # hk.set_state("nfe", nfe + 1)
        out = hk.GroupNorm(self.dim_out)(x)
        out = jax.nn.relu(out)
        out = ConcatConv2D(self.dim_out)(out, t)
        out = jax.nn.relu(hk.GroupNorm(self.dim_out)(out))
        out = ConcatConv2D(self.dim_out)(out, t)
        out = hk.GroupNorm(self.dim_out)(out)

        return out


class PreODE(hk.Module):
    """PreODEBlock"""
    def __init__(self, dim_out):
        super(PreODE, self).__init__()
        self.dim_out = dim_out

    def __call__(self, x):
        x = hk.Conv2D(self.dim_out, 3, 1)(x)
        x = ResDownBlock(64)(x)
        x = ResDownBlock(64)(x)

        return x


class ODEBlock(hk.Module):
    """ODE block"""
    def __init__(self, odefunc, tol=1.):
        super(ODEBlock, self).__init__()
        self.tol = tol
        self.odefunc_forward = odefunc

    def __call__(self, x, params):
        odefunc_apply = lambda x, t: self.odefunc_forward.apply(params, x=x, t=t)
        states = odeint(odefunc_apply,
                        x, jnp.array([0., 1.]),
                        rtol=self.tol, atol=self.tol)

        return states[-1]


class PostODE(hk.Module):
    """Post ODE Block"""
    def __init__(self, dim_out, n_cls):
        super(PostODE, self).__init__()
        self.dim_out = dim_out
        self.n_cls = n_cls

    def __call__(self, x):
        x = hk.GroupNorm(self.dim_out)(x)
        x = jax.nn.relu(x)
        x = hk.AvgPool(2, 1, "SAME")(x)

        x = hk.Flatten()(x)
        x = hk.Linear(output_size=self.n_cls)(x)
        x = jax.nn.log_softmax(x)

        return x


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    def _forward_pre_ode(x):
        module = PreODE(64)
        return module(x)

    def _forward_ode_func(x, t):
        # nfe = hk.get_state("NFE", shape=[], dtype=jnp.int32, init=jnp.ones)
        # hk.set_state("NFE", nfe + 1)
        module = ODEfunc(64)
        return module(x, t)

    def _forward_post_ode(x):
        module = PostODE(64, 10)
        return module(x)

    dummy_x = jnp.ones([1, 28, 28, 1])
    dummy_t = 1.
    rng_key = jax.random.PRNGKey(2021)

    pre_forward = hk.without_apply_rng(hk.transform(_forward_pre_ode))
    pre_params = pre_forward.init(rng=rng_key, x=dummy_x)
    pre_fn = pre_forward.apply

    dummy_x2 = pre_fn(params=pre_params, x=dummy_x)

    odefunc_forward = hk.without_apply_rng(hk.transform(_forward_ode_func))
    odefunc_params = odefunc_forward.init(rng=rng_key, x=dummy_x2, t=dummy_t)
    odefunc_fn = odefunc_forward.apply

    def odeblock(x, params, tol):
        odefunc_apply = lambda x, t: odefunc_fn(params=params, x=x, t=t)
        states = odeint(odefunc_apply,
                        x, jnp.array([0., 1.]),
                        rtol=tol, atol=tol)
        return states[-1]

    post_forward = hk.without_apply_rng(hk.transform(_forward_post_ode))
    post_params = post_forward.init(rng=rng_key, x=dummy_x2)
    post_fn = post_forward.apply

    output = pre_fn(params=pre_params, x=dummy_x)
    output = odeblock(x=output, params=odefunc_params, tol=1.)
    output = post_fn(params=post_params, x=output)

    print("Dummy output: \n", output)
    # print("NFE: %d" % nfe)
