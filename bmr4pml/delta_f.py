import jax.numpy as jnp
from jax import jit
from jax.scipy.linalg import solve_triangular

@jit
def ΔF_mn(M, a, B, gamma_sqr, sigma_sqr=1.):
    # Delta F for matrix normal posterior
    # vec(AXB) = (B^T \otimes A) vec(X)
    i, j = M.shape

    Z_trans = solve_triangular(B, M.T, trans=1) / a
    Z = Z_trans.T

    df = - jnp.trace(Z_trans @ Z) / 2
    df = df + j * i * jnp.log(sigma_sqr) / 2 # - j * jnp.log(gamma_sqr).sum() / 2

    V = B.T @ B
    u = jnp.square(a)
    _g = 1 - gamma_sqr/sigma_sqr

    lam, Q = jnp.linalg.eigh(V)
    Q_inv = Q.T
    inv_mat = jnp.clip(1 / ( jnp.kron(jnp.ones(j), gamma_sqr) + jnp.kron(lam, u * _g)), a_min=1e-16)
    D = inv_mat.reshape(i, j, order='F')

    t_M = gamma_sqr[:, None] * ( ( D * ( M @ Q ) ) @ Q_inv )
    t_S = jnp.sqrt( (u * gamma_sqr)[:, None] * (D @ (Q_inv * (Q_inv @ V ))) ) # diagonal of the covariance matrix

    df += jnp.log(inv_mat).sum() / 2

    t_Z_trans = solve_triangular(B, t_M.T, trans=1) / a
    df += jnp.trace(t_Z_trans @ Z ) / 2

    return df, t_M, t_S

@jit
def ΔF_mv(mu, P, gamma_sqr, sigma_sqr):
    # Delta F for multivariate normal posterior
    M = jnp.diag(gamma_sqr) @ P + jnp.diag(1 - gamma_sqr/sigma_sqr)

    _, logdet = jnp.linalg.slogdet(M)
    df = -logdet/2

    _, logdet = jnp.linalg.slogdet(P)
    df += logdet / 2

    df += jnp.sum(jnp.log(sigma_sqr))
    
    t_P = P + jnp.diag(1/gamma_sqr - 1/sigma_sqr)
    _mu = P @ mu
    t_mu = jnp.linalg.solve(t_P, _mu)

    df -= jnp.inner(_mu, mu) / 2
    df += jnp.inner(_mu, t_mu) / 2
    
    return df, t_mu, t_P

@jit
def ΔF_mf(mu, pi, gamma_sqr, sigma_sqr=1):
    # Delta F for normal posterior
    M = gamma_sqr * (pi - 1/sigma_sqr) + 1
    t_sig_sqr = gamma_sqr / M

    df = - jnp.log(M).sum() / 2

    df += (jnp.log(pi) + jnp.log(sigma_sqr)).sum() / 2

    _mu = pi * mu
    t_mu = t_sig_sqr * _mu 
    
    df -= jnp.inner(_mu, mu) / 2
    df += jnp.inner(_mu, t_mu) / 2
    
    return df, t_mu, jnp.sqrt(t_sig_sqr)