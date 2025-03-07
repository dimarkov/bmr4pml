import functools
from typing import Any, NamedTuple, Optional, Union

import chex
import jax

from jax import random as jr, tree_util as jtu, numpy as jnp

from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import utils
from optax._src import transform

_abs_sq = numerics.abs_sq

ScalarOrSchedule = Union[float, jax.Array, base.Schedule]

@functools.partial(jax.jit, inline=True)
def bias_correction(moment, decay, count):
  """Performs bias correction. It becomes a no-op as count goes to infinity."""
  # The conversion to the data type of the moment ensures that bfloat16 remains
  # bfloat16 in the optimizer state. This conversion has to be done after
  # `bias_correction_` is calculated as calculating `decay**count` in low
  # precision can result in it being rounded to 1 and subsequently a
  # "division by zero" error.
  bias_correction_ = 1 - decay**count

  # Perform division in the original precision.
  return jax.tree_util.tree_map(lambda t: t / bias_correction_.astype(t.dtype), moment)

def update_moment(updates, moments, means, decay, lam):
  """Compute the exponential moving average of the `order`-th moment."""
  return jax.tree_util.tree_map(
      lambda g, t, mu: (1 - decay) * (g + mu * lam) + decay * t, updates, moments, means)


def update_moment_per_elem_norm(updates, moments, decay, order):
  """Compute the EMA of the `order`-th moment of the element-wise norm."""

  def orderth_norm(g):
    if jnp.isrealobj(g):
      return g ** order
    else:
      half_order = order / 2
      # JAX generates different HLO for int and float `order`
      if half_order.is_integer():
        half_order = int(half_order)
      return _abs_sq(g) ** half_order

  return jax.tree_util.tree_map(
      lambda g, t: (1 - decay) * orderth_norm(g) + decay * t, updates, moments)


def update_second_moment(updates, moments, noise, decay, N, lam):
  """Compute the EMA of the `order`-th moment of the element-wise norm."""

  γ = 1 - decay
  s = jax.tree_util.tree_map(
      lambda g, s, ε: jnp.exp( jnp.log(s) + γ * (g * ε * jnp.sqrt(N * (s + lam)) - s ) ), 
      updates, 
      moments, 
      noise
    )

  return s

class ScaleByLieState(NamedTuple):
  """State for the algorithms."""
  key: chex.PRNGKey
  N: int
  noise: float
  m_u: base.Updates
  m_v: base.Updates
  A: base.Updates
  ε: base.Updates


# group-exponential
def grpexp(rho, x, eps=1e-6):
    return jax.lax.select(
        jnp.abs(x) < eps, 
        - rho + x * (rho ** 2.0) / 2, 
        (jnp.exp(- rho * x) - 1) / x
    )

def scale_by_lie(
    key: chex.PRNGKey,
    lr: float,
    init_scale: float,
    temperature: float,
    train_set_size: int,
    num_samples: int,
    init_noise: float = 1.0,
    a1: float = 1.0,
    a2: float = 1.0,
    b1: float = 0.9,
    b2: float = 0.999,
    c_X: float = 2.0,
    c_y: float = 1.0,
    m_dtype: Optional[chex.ArrayDType] = None
) -> base.GradientTransformation:
  """Rescale updates according to the Lie-Group Bayesian Learning Rule (BLR) algorithm.

  References:
    [EM Kıral et al, 2023](https://arxiv.org/abs/2303.04397)

  Args:
    scaled_prior_precision: prior precision on parameters divided by the number of data points.
    scaled_initi_precision: initial precision for variational distribution q divided by the number of data points.
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    m_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    A `GradientTransformation` object.
  """

  m_dtype = utils.canonicalize_dtype(m_dtype)

  s_init = init_scale
  n_init = init_noise
  N = train_set_size
  S = num_samples
  t_tau = temperature / N

  def init_fn(key, params):
    m_u = jtu.tree_map(lambda t: jnp.zeros_like(t, dtype=m_dtype), params)  # first moment
    m_v = jtu.tree_map(lambda t: jnp.zeros_like(t, dtype=m_dtype), params) # second moment
    A = jtu.tree_map(lambda t: s_init * jnp.ones_like(t, dtype=m_dtype), params)  # scale
    
    p_names = list(params.keys())
    key, _key = jr.split(key)
    keys = {k:v for k, v in zip(p_names, list(jr.split(_key, len(p_names))))}
    ε = jtu.tree_map(lambda t, key: n_init * jr.normal(key, shape=(S, *t.shape), dtype=m_dtype), params, keys)  # noise
    return ScaleByLieState(key=key, m_u=m_u, m_v=m_v, A=A, ε=ε, N=N, noise=n_init)

  def update_fn(grads, state, params):

    key, _key = jr.split(state.key)

    V_k = jtu.tree_map(lambda s, g: s * g, state.A, grads)
    V = jtu.tree_map(lambda v: v.mean(0), V_k)
    U = jtu.tree_map(lambda v, e: jnp.mean(e * v, 0), V_k, state.ε)

    m_u = jtu.tree_map(lambda m, u: b2 * m + (1 - b2) * u, state.m_u, U)
    m_v = jtu.tree_map(lambda m, v: b1 * m + (1 - b1) * v, state.m_v, V)

    A = jtu.tree_map(lambda a, m: a * jnp.exp(- lr * a2 * (m - t_tau)), state.A, m_u)
    updates = jtu.tree_map(
        lambda a, m1, m2: c_X * a * m2 * grpexp(lr * a1, (m1 - t_tau)) / c_y, state.A, m_u, m_v
      )
    
    (m_u, m_v, A) = utils.cast_tree((m_u, m_v, A), m_dtype)

    p_names = list(params.keys())
    keys = {k:v for k, v in zip(p_names, list(jr.split(_key, len(p_names))))}
    ε = jtu.tree_map(lambda t, key: n_init * jr.normal(key, shape=(S, *t.shape), dtype=m_dtype), params, keys)  # noise

    return updates, ScaleByLieState(key=key, m_u=m_u, m_v=m_v, A=A, ε=ε, N=N, noise=n_init)

  return base.GradientTransformation(functools.partial(init_fn, key), update_fn)


def scale_by_von(
    key: chex.PRNGKey,
    scaled_prior_precision: float,
    scaled_init_precision: float,
    train_set_size: int,
    b1: float = 0.9,
    b2: float = 0.999,
    eps_root: float = 0.0,
    m_dtype: Optional[chex.ArrayDType] = None
) -> base.GradientTransformation:
  """Rescale updates according to the Adam algorithm.

  References:
    [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)

  Args:
    scaled_prior_precision: prior precision on parameters divided by the number of data points.
    scaled_initi_precision: initial precision for variational distribution q divided by the number of data points.
    b1: Decay rate for the exponentially weighted average of grads.
    b2: Decay rate for the exponentially weighted average of squared grads.
    eps_root: Term added to the denominator inside the square-root to improve
      numerical stability when backpropagating gradients through the rescaling.
    m_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    A `GradientTransformation` object.
  """

  m_dtype = utils.canonicalize_dtype(m_dtype)

  s_init = max(1e-16, scaled_init_precision - scaled_prior_precision)
  t_lam = scaled_prior_precision

  def init_fn(params):
    m = jax.tree_util.tree_map(lambda t: jnp.zeros_like(t, dtype=m_dtype), params)  # First moment
    s = jax.tree_util.tree_map(lambda t: s_init * jnp.ones_like(t), params)  # Second moment
    ε = jax.tree_util.tree_map(lambda t: jnp.zeros_like(t, dtype=m_dtype), params) # noise
    return ScaleByState(count=jnp.zeros([], jnp.int32), m=m, s=s, ε=ε)

  def update_fn(updates, state, params):
    m = update_moment(updates, state.m, params, b1, t_lam)
    s = update_second_moment(updates, state.s, state.ε, b2, train_set_size, t_lam)
    count_inc = numerics.safe_int32_increment(state.count)
    m_hat = bias_correction(m, b1, count_inc)
    s_hat = bias_correction(s, b2, count_inc)
    updates = jax.tree_util.tree_map(
        lambda m, v: m / (jnp.sqrt(v + eps_root) + t_lam), m_hat, s_hat)
    
    m = utils.cast_tree(m, m_dtype)
    return updates, ScaleByState(count=count_inc, m=m, s=s, ε=state.ε)

  return base.GradientTransformation(init_fn, update_fn)

def _scale_by_learning_rate(learning_rate: ScalarOrSchedule, flip_sign=True):
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return transform.scale_by_schedule(lambda count: m * learning_rate(count))
  return transform.scale(m * learning_rate)
  
def lieblr(
    key: chex.PRNGKey,
    learning_rate: ScalarOrSchedule,
    init_scale: float = 1.0,
    init_noise: float = 1.0,
    temperature: float = 1.0,
    train_set_size: int = 1,
    num_samples: int = 1,
    a1: float = 1.0,
    a2: float = 1.0,
    b1: float = 0.9,
    b2: float = 0.999,
    c_X: float = 2.0,
    c_y: float = 1.0,
    m_dtype: Optional[Any] = None
) -> base.GradientTransformation:
  lr = learning_rate #_scale_by_learning_rate(learning_rate, flip_sign=False)
  return scale_by_lie(
    key,
    lr,
    init_scale,
    temperature,
    train_set_size,
    num_samples,
    init_noise=init_noise,
    a1=a1,
    a2=a2,
    b1=b1,
    b2=b2,
    c_X=c_X,
    c_y=c_y,
    m_dtype=m_dtype
    )