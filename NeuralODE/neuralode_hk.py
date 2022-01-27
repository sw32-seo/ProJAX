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

    def __init__(self, inplanes, planes, stride=2, downsample=None):
        super(ResDownBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = jax.nn.relu
        self.downsample = downsample
        self.conv1 = conv3x3(planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes)

    def __call__(self, x):
        out = self.relu(self.norm1(x))

        shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


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
    def __init__(self, odefunc, tol):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.tol = tol

    def __call__(self, x):
        integration_time = jnp.array([0., 1.], dtype=x.dtype)
        if hk.running_init():
            out = self.odefunc(x, 1.)
            return out
        else:
            out = odeint(self.odefunc, x, integration_time, rtol=self.tol, atol=self.tol)
            return out[-1]


class PostODE(hk.Module):
    """Post ODE Block"""
    def __init__(self, dim, n_cls):
        super(PostODE, self).__init__()
        self.norm = norm(dim)
        self.relu = jax.nn.relu
        self.avgpool = hk.AvgPool(window_shape=(1, 1), strides=1, padding='SAME')
        self.flatten = hk.Flatten(1)
        self.linear = hk.Linear(output_size=n_cls)

    def __call__(self, x):
        x = self.norm(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


if __name__ == '__main__':

    def _foward_fn_node(x):
        conv = hk.Conv2D(64, 3, 1)
        module1 = ResDownBlock(64, 64, stride=2, downsample=conv1x1(64, 2))
        module2 = ResDownBlock(64, 64, stride=2, downsample=conv1x1(64, 2))
        feature_layer = ODEBlock(ODEfunc(64), 1e-3)
        fc_layers = PostODE(64, 10)

        out = conv(x)
        out = module1(out)
        out = module2(out)
        out = feature_layer(out)
        out = fc_layers(out)

        return out

    forward_node = hk.without_apply_rng(hk.transform(_foward_fn_node))

    dummy_x = jnp.ones([1, 28, 28, 1])
    dummy_t = 1.
    rng_key = jax.random.PRNGKey(42)

    params = forward_node.init(rng=rng_key, x=dummy_x)

    print(params)
