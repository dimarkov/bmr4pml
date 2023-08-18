import functools
from typing import Any, Callable, NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp

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
  return jax.tree_util.tree_map(
      lambda t: t / bias_correction_.astype(t.dtype), moment)

def update_moment(updates, moments, means, decay, lam, order):
  """Compute the exponential moving average of the `order`-th moment."""
  return jax.tree_util.tree_map(
      lambda g, t, mu: (1 - decay) * (g ** order + mu * lam) + decay * t, updates, moments, means)


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


class ScaleByState(NamedTuple):
  """State for the algorithms."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  m: base.Updates
  s: base.Updates


def scale_by_vadam(
    scaled_prior_precision: float,
    scaled_init_precision: float,
    b1: float = 0.9,
    b2: float = 0.999,
    eps_root: float = 0.0,
    m_dtype: Optional[chex.ArrayDType] = None,
    *,
    train_set_size: int = 1
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

  s_init = scaled_init_precision - scaled_prior_precision
  t_lam = scaled_prior_precision

  def init_fn(params):
    m = jax.tree_util.tree_map(  # First moment
        lambda t: jnp.zeros_like(t, dtype=m_dtype), params)
    s = jax.tree_util.tree_map(lambda x: s_init * jnp.ones_like(x), params)  # Second moment
    return ScaleByState(count=jnp.zeros([], jnp.int32), m=m, s=s)

  def update_fn(updates, state, params):
    m = update_moment(updates, state.m, params, b1, t_lam, 1)
    s = update_moment_per_elem_norm(updates, state.s, b2, 2)
    count_inc = numerics.safe_int32_increment(state.count)
    m_hat = bias_correction(m, b1, count_inc)
    s_hat = bias_correction(s, b2, count_inc)
    updates = jax.tree_util.tree_map(
        lambda m, v: m / (jnp.sqrt(v + eps_root) + t_lam), m_hat, s_hat)
    
    m = utils.cast_tree(m, m_dtype)
    return updates, ScaleByState(count=count_inc, m=m, s=s)

  return base.GradientTransformation(init_fn, update_fn)


def scale_by_vadabelief(
    scaled_prior_precision: float,
    scaled_init_precision: float,
    b1: float = 0.9,
    b2: float = 0.999,
    eps_root: float = 1e-16,
    m_dtype: Optional[chex.ArrayDType] = None,
) -> base.GradientTransformation:
  """Rescale updates according to the variational AdaBelief algorithm.

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

  s_init = scaled_init_precision - scaled_prior_precision
  t_lam = scaled_prior_precision

  def init_fn(params):
    m = jax.tree_util.tree_map(  # First moment
        lambda t: jnp.zeros_like(t, dtype=m_dtype), params)
    s = jax.tree_util.tree_map(lambda x: s_init * jnp.ones_like(x), params)  # Second moment
    return ScaleByState(count=jnp.zeros([], jnp.int32), m=m, s=s)

  def update_fn(updates, state, params):
    m = update_moment(updates, state.m, params, b1, t_lam, 1)

    prediction_error = jax.tree_util.tree_map(lambda g, m: g - m, updates, state.m)
    s = update_moment_per_elem_norm(prediction_error, state.s, b2, 2)
    s = jax.tree_util.tree_map(lambda v: v + eps_root, s)
    count_inc = numerics.safe_int32_increment(state.count)
    m_hat = bias_correction(m, b1, count_inc)
    s_hat = bias_correction(s, b2, count_inc)
    updates = jax.tree_util.tree_map(
        lambda m, v: m / (jnp.sqrt(v) + t_lam), m_hat, s_hat)
    
    m = utils.cast_tree(m, m_dtype)
    return updates, ScaleByState(count=count_inc, m=m, s=s)

  return base.GradientTransformation(init_fn, update_fn)


def _scale_by_learning_rate(learning_rate: ScalarOrSchedule, flip_sign=True):
  m = -1 if flip_sign else 1
  if callable(learning_rate):
    return transform.scale_by_schedule(lambda count: m * learning_rate(count))
  return transform.scale(m * learning_rate)

def vadam(
    learning_rate: ScalarOrSchedule,
    t_lam: float,
    t_init: float, 
    b1: float = 0.9,
    b2: float = 0.999,
    eps_root: float = 0.0,
    m_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  r"""The variational Adam (Vadam) optimizer.

  Vadam is a stochastic natural gradient variant with gradient scaling adaptation. The scaling
  used for each parameter is computed from estimates of first and second-order
  moments of the gradients (using suitable exponential moving averages).

  Let :math:`\alpha_t` represent the learning rate and :math:`\beta_1, \beta_2`,
  :math:`\varepsilon`, :math:`\bar{\varepsilon}` represent the arguments
  ``b1``, ``b2``, ``eps`` and ``eps_root`` respectievly. The learning rate is
  indexed by :math:`t` since the learning rate may also be provided by a
  schedule function.

  The ``init`` function of this optimizer initializes an internal state
  :math:`S_0 := (m_0, s_0) = (0, (init_precision - prior_precision ) / train_set_size)`, representing initial estimates for the first and second moments. In practice these values are stored as pytrees
  , with the same shape as the model updates.
  At step :math:`t`, the ``update`` function of this optimizer takes as
  arguments the incoming gradients :math:`g_t` and optimizer state :math:`S_t`
  and computes updates :math:`u_t` and new state :math:`S_{t+1}`. Thus, for
  :math:`t > 0`, we have,

  .. math::
    \begin{align*}
      m_t &\leftarrow \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot ( g_t + \tilde{\lambda} \mu ) \\
      s_t &\leftarrow \beta_2 \cdot s_{t-1} + (1-\beta_2) \cdot {g_t}^2 \\
      \hat{m}_t &\leftarrow m_t / {(1-\beta_1^t)} \\
      \hat{s}_t &\leftarrow s_t / {(1-\beta_2^t)} \\
      u_t &\leftarrow \alpha_t \cdot \hat{m}_t / \left({\sqrt{\hat{v}_t +
      \bar{\varepsilon}} + \tilde{\lambda}} \right)\\
      S_t &\leftarrow (m_t, s_t).
    \end{align*}

  References:
    Khan et al, 2018: https://arxiv.org/abs/1806.04854

  Args:
    learning_rate: A fixed global scaling factor.
    t_lam: Prior precision devided by the number of data points in the training set.
    t_init: Initial precision of the variational posterior q devided by the number of data points in the training set.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      example when computing (meta-)gradients through Vadam.
    m_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    The corresponding `GradientTransformation`.
  """
  return combine.chain(
      scale_by_vadam(
          t_lam, t_init, b1=b1, b2=b2, eps_root=eps_root, m_dtype=m_dtype),
      _scale_by_learning_rate(learning_rate),
  )

def vadabelief(
    learning_rate: ScalarOrSchedule,
    t_lam: float,
    t_init: float, 
    b1: float = 0.9,
    b2: float = 0.999,
    eps_root: float = 1e-16,
    m_dtype: Optional[Any] = None,
) -> base.GradientTransformation:
  r"""The variational Adabelief (Vadabelief) optimizer.

  Vadabelief is a stochastic natural gradient variant with gradient scaling adaptation. The scaling
  used for each parameter is computed from estimates of first and second-order
  moments of the gradients (using suitable exponential moving averages).

  Let :math:`\alpha_t` represent the learning rate and :math:`\beta_1, \beta_2`,
  :math:`\varepsilon`, :math:`\bar{\varepsilon}` represent the arguments
  ``b1``, ``b2``, ``eps`` and ``eps_root`` respectievly. The learning rate is
  indexed by :math:`t` since the learning rate may also be provided by a
  schedule function.

  The ``init`` function of this optimizer initializes an internal state
  :math:`S_0 := (m_0, s_0) = (0, (init_precision - prior_precision ) / train_set_size)`, representing initial estimates for the first and second moments. In practice these values are stored as pytrees
  , with the same shape as the model updates.
  At step :math:`t`, the ``update`` function of this optimizer takes as
  arguments the incoming gradients :math:`g_t` and optimizer state :math:`S_t`
  and computes updates :math:`u_t` and new state :math:`S_{t+1}`. Thus, for
  :math:`t > 0`, we have,

  .. math::
    \begin{align*}
      m_t &\leftarrow \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot ( g_t + \tilde{\lambda} \mu ) \\
      s_t &\leftarrow \beta_2 \cdot s_{t-1} + (1-\beta_2) \cdot {g_t}^2 \\
      \hat{m}_t &\leftarrow m_t / {(1-\beta_1^t)} \\
      \hat{s}_t &\leftarrow s_t / {(1-\beta_2^t)} \\
      u_t &\leftarrow \alpha_t \cdot \hat{m}_t / \left({\sqrt{\hat{v}_t +
      \bar{\varepsilon}} + \tilde{\lambda}} \right)\\
      S_t &\leftarrow (m_t, s_t).
    \end{align*}

  References:
    Khan et al, 2018: https://arxiv.org/abs/1806.04854

  Args:
    learning_rate: A fixed global scaling factor.
    t_lam: Prior precision devided by the number of data points in the training set.
    t_init: Initial precision of the variational posterior q devided by the number of data points in the training set.
    b1: Exponential decay rate to track the first moment of past gradients.
    b2: Exponential decay rate to track the second moment of past gradients.
    eps_root: A small constant applied to denominator inside the square root (as
      in RMSProp), to avoid dividing by zero when rescaling. This is needed for
      example when computing (meta-)gradients through Vadam.
    m_dtype: Optional `dtype` to be used for the first order accumulator; if
      `None` then the `dtype` is inferred from `params` and `updates`.

  Returns:
    The corresponding `GradientTransformation`.
  """
  return combine.chain(
      scale_by_vadabelief(
          t_lam, t_init, b1=b1, b2=b2, eps_root=eps_root, m_dtype=m_dtype),
      _scale_by_learning_rate(learning_rate),
  )