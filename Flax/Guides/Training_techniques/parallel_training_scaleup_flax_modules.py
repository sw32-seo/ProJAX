import functools
import os
import shutil
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import flax
import jax
import optax
import orbax.checkpoint
from absl import app, flags, logging
from flax import linen as nn
from flax import struct, traverse_util
from flax.core import freeze, unfreeze
from flax.training import train_state
from jax import lax
from jax import numpy as jnp
from jax import random
from jax.sharding import NamedSharding, PartitionSpec

os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
print(f"We have 8 fake JAX devices: {jax.devices()}")
