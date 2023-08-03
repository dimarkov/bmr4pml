import jax.numpy as jnp
from jax import jit

@jit
def mish(x):
    # x * tanh(softplus(x))
    z = jnp.exp(x)
    u = z * (z + 2)
    return x * u / (u + 2)