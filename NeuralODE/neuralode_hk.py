import haiku as hk
import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
import os


def conv3x3(out_planes, stride=1):
    return hk.Conv2D(output_channels=out_planes, kernel_shape=3, stride=stride,
                     padding='SAME', with_bias=False)


def conv1x1(out_planes, stride=1):
    return hk.Conv2D(output_channels=out_planes, kernel_shape=1, stride=stride,
                     padding='SAME', with_bias=False)


def norm(dim):
    return hk.GroupNorm(min(32, dim))


class ResBlock(hk.Module):
    """Standard ResBlock w/o downsampling"""
    def __init__(self, inplanes, planes, stride=1):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = jax.nn.relu
        self.conv1 = conv3x3(planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes)

    def __call__(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class ResDownBlock(hk.Module):
    """Standard ResBlock w/ downsampling"""
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
    def __init__(self, dim_out, ksize=3, stride=1, padding='SAME', bias=True):
        super(ConcatConv2D, self).__init__()
        self._layer = hk.Conv2D(output_channels=dim_out, kernel_shape=ksize,
                                stride=stride, padding=padding, with_bias=bias,
                                )

    def __call__(self, x, t):
        """x is batch of images in [B, H, W, C]"""
        tt = jnp.ones_like(x[:, :, :, :1]) * t
        ttx = jnp.concatenate([tt, x], -1)
        return self._layer(ttx)


class ODEfunc(hk.Module):
    """ODE function which replace ResNet"""
    def __init__(self, dim=64):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = jax.nn.relu
        self.conv1 = ConcatConv2D(dim, 3, 1, 'SAME')
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2D(dim, 3, 1, 'SAME')
        self.norm3 = norm(dim)

    def __call__(self, x, t):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out, t)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out, t)
        out = self.norm3(out)
        return out


class ODEBlock(hk.Module):
    """ODE block"""
    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc

    def __call__(self, x):
        integration_time = jnp.array([0., 1.], dtype=x.dtype)
        if hk.running_init():
            out = self.odefunc(x, 1.)
            return out
        else:
            out = odeint(self.odefunc, x, integration_time, rtol=1e-3, atol=1e-3)
            return out[-1]


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

    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    def _forward_pre_ode(x):
        module = PreODE(64)
        return module(x)

    def _forward_ode_func(x, t):
        nfe = hk.get_state("NFE", shape=[], dtype=jnp.int32, init=jnp.ones)
        hk.set_state("NFE", nfe + 1)
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

    odefunc_forward = hk.without_apply_rng(hk.transform_with_state(_forward_ode_func))
    odefunc_params, state = odefunc_forward.init(rng=rng_key, x=dummy_x2, t=dummy_t)
    odefunc_fn = odefunc_forward.apply

    def odeblock(x, params, state, tol):
        odefunc_apply = lambda x, t, state: odefunc_fn(params=params, state=state, x=x, t=t)
        states = odeint(odefunc_apply,
                        x, jnp.array([0., 1.]), state,
                        rtol=tol, atol=tol)
        return states[-1]

    post_forward = hk.without_apply_rng(hk.transform(_forward_post_ode))
    post_params = post_forward.init(rng=rng_key, x=dummy_x2)
    post_fn = post_forward.apply

    output = pre_fn(params=pre_params, x=dummy_x)
    output = odeblock(x=output, params=odefunc_params, state=state, tol=1.)
    output = post_fn(params=post_params, x=output)

    print("Dummy output: \n", output)
    # print("NFE: %d" % nfe)
