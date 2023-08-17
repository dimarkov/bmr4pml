import equinox as eqx
import optax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpyro.distributions as dist

from collections import defaultdict
from functools import partial
from jax import nn, random, vmap, lax, jit, grad
from jax.scipy.special import digamma
from numpyro import handlers, sample, plate, deterministic, factor, subsample, param, prng_key
from numpyro.infer import SVI, Predictive, TraceGraph_ELBO, TraceMeanField_ELBO
from numpyro.distributions import constraints
from numpyro.distributions.transforms import AffineTransform
from numpyro.infer.reparam import TransformReparam
from numpyro.infer.autoguide import (
    AutoDelta,
    AutoNormal,
    AutoMultivariateNormal,
    AutoLowRankMultivariateNormal
)
from numpyro.optim import optax_to_numpyro
from .nn.vit import Params

from .delta_f import *

def init_fn(rng_key, shape, radius=2.):
    return random.uniform(rng_key, shape=shape, minval=-radius, maxval=radius)

@jit
def trigamma(x):
    return grad(digamma)(x)

@jit
def zeta(alpha, beta):
    return digamma(alpha) - digamma(beta)

def get_linear_layers(layer):
    is_linear = lambda x: isinstance(x, eqx.nn.Linear) or isinstance(x, eqx.nn.Conv) or isinstance(x, eqx.nn.LayerNorm)
    return [x for x in jtu.tree_leaves(layer, is_leaf=is_linear) if is_linear(x)]

def get_norm_layers(layer):
    def is_norm(x):
        is_batch = isinstance(x, eqx.nn.BatchNorm)
        is_group = isinstance(x, eqx.nn.GroupNorm)

        return is_batch or is_group

    return [x for x in jtu.tree_leaves(layer, is_leaf=is_norm) if is_norm(x)]

class BayesRegression(object):
    type: str
    layers: list
    gamma: dict
    tau0: float
    
    def __init__(
        self,
        nnet,
        *,
        regtype='linear',
        tau0=1e-2,
        gamma0=1.,
        ):

        self.nnet = nnet
        self.layers = get_linear_layers(nnet)
        
        # type of the rergression problem
        self.type = regtype 

        # global scale hyperparameter
        self.tau0 = tau0
        
        # prior weight uncertanty
        self.gamma = defaultdict(lambda: gamma0)

    def likelihood(self, x, y, nnet, sigma, batch_size=None):
        N = x.shape[0]
        with plate('data', N, subsample_size=batch_size):
            batch_x = subsample(x, event_dim=3)
            key = random.PRNGKey(0) if prng_key() is None else prng_key()
            keys = random.split(key, batch_x.shape[0]) 
            mu = vmap(nnet)(batch_x, key=keys)

            batch_y = y if y is None else subsample(y, event_dim=0)

            if self.type == 'linear':
                sample('obs', dist.Normal(mu.squeeze(), sigma), obs=batch_y)
            
            elif self.type in ['logistic', 'multinomial']:
                logits = jnp.pad(mu, ((0, 0), (1, 0))) if self.type == 'logistic' else mu
                deterministic('probs', nn.softmax(logits, -1))
                sample('obs', dist.Categorical(logits=logits), obs=batch_y)

    def hyperprior(self, name, shape, layer, last=False):
        c_sqr = sample(name + '.c^2', dist.InverseGamma(2., 6.))
        i, j = shape

        if not last:
            eps = sample(name + '.eps', dist.HalfCauchy(1.)) if i > 1 else jnp.ones(1)
            tau0 = jnp.sqrt(self.tau0) * eps

            tau = sample(name + '.tau', dist.HalfCauchy(1.).expand([i]).to_event(1))            
            lam = sample(name + '.lam', dist.HalfCauchy(1.).expand([i, j]).to_event(2)) if layer == 0 else jnp.ones(j)
        
        else:
            eps = sample(name + '.eps', dist.HalfCauchy(1.))
            tau = jnp.broadcast_to(10 * jnp.sqrt(self.tau0) * eps, (i,))
            lam = jnp.ones((i, j))

        psi = jnp.expand_dims(tau, -1) * lam
        
        gamma = deterministic(name + '.gamma', jnp.sqrt(c_sqr * psi ** 2 / (c_sqr + psi ** 2)))

        return gamma

    def prior(self, name, sigma, gamma, loc=0., centered=False):
        if centered:
            aff = AffineTransform(loc, sigma)
            dstr = dist.Normal(0., gamma).to_event(2) 
        else:
            aff = AffineTransform(loc, sigma * gamma)
            dstr = dist.Normal(0., 1.).expand(list(gamma.shape)).to_event(2)

        with handlers.reparam(config={name: TransformReparam()}):
            weight = sample(
                name, 
                dist.TransformedDistribution(dstr, aff)
            )

        return weight

    def _register_network_vars(self, sigma, with_hyperprior, gammas=None):
        L = len(self.layers)
        new_layers = []
        for l, layer in enumerate(self.layers):
            weight = layer.weight
            bias = layer.bias
            shape = weight.reshape(weight.shape[0], -1).shape
            if bias is not None:
                shape = (shape[0], shape[1] + 1)
            
            # mark layer as last if l + 1 == L
            last = False if l == 0 else True
            last = False if l + 1 < L else last 
            name = f'layer{l}.weight'        
            if with_hyperprior:
                gamma = self.hyperprior(name, shape, l, last=last)
                c = 1.
            else:
                gamma = self.gamma[name] if gammas is None else gammas[name]
                
                c_inv_sqr = sample(f'{name}.c_inv_sqr', dist.Gamma(2, 2))
                c = jnp.sqrt( 1 / c_inv_sqr ) if last else jnp.sqrt( 1 / c_inv_sqr ) / 10
            
            gamma = jnp.broadcast_to(gamma, shape)
            
            if self.type == 'linear':
                scale = c if l + 1 < L else sigma
            else:
                scale = c
            
            _weights = self.prior(name, scale, gamma)
            
            weight = _weights.reshape(weight.shape) if bias is None else _weights[..., :-1].reshape(weight.shape)
            bias = _weights[..., -1].reshape(bias.shape) if bias is not None else bias

            layer = eqx.tree_at(lambda x: (x.weight, x.bias), layer, (weight, bias))
            new_layers.append(layer)
        
        nnet = eqx.tree_at(get_linear_layers, self.nnet, new_layers)

        return nnet

    def model(self, x, y=None, gammas=None, a0=2., b0=2., batch_size=None, with_hyperprior=False, **kwargs):
        if self.type == 'linear':
            sigma_sqr_inv = sample('sigma^-2', dist.Gamma(a0, b0))
            sigma = deterministic('sigma', 1/jnp.sqrt(sigma_sqr_inv))
        else:
            sigma = deterministic('sigma', jnp.ones(1))

        nnet = self._register_network_vars(sigma, with_hyperprior, gammas=gammas)

        inference = kwargs.pop('inference', None)
        if inference is not None:
            nnet = eqx.tree_inference(nnet, value=inference)

        self.likelihood(x, y, partial(nnet, **kwargs), sigma, batch_size)


class SVIRegression(BayesRegression):
    reduced: bool

    def __init__(
        self, 
        nnet, 
        *,
        reduced=False,
        regtype='linear',
        tau0=1e-2,
        gamma0=1.,
        **kwargs
    ):  
        super().__init__(
            nnet, 
            regtype=regtype,
            tau0=tau0,
            gamma0=gamma0,
        )

        self.reduced = reduced
        self.set_guide_loss_optim(**kwargs)

    def set_guide_loss_optim(self, **kwargs):
        num_particles = kwargs.pop('num_particles', 1)
        autoguide = kwargs.pop('autoguide', 'delta')
        optimizer = kwargs.pop('optimizer', optax.adabelief)
        optim_kwargs = kwargs.pop('optim_kwargs', {'learning_rate': 1e-3})
        self.optim = optax_to_numpyro(optimizer(**optim_kwargs))

        if autoguide == 'mean-field':
            self.guide = AutoNormal(self.model, **kwargs)
            self.loss = TraceGraph_ELBO(num_particles=num_particles)

        elif autoguide == 'multivariate':
            self.guide = AutoMultivariateNormal(self.model, **kwargs)
            self.loss = TraceGraph_ELBO(num_particles=num_particles)

        elif autoguide == 'lowrank-multivariate':
            self.guide = AutoLowRankMultivariateNormal(self.model, **kwargs)
            self.loss = TraceGraph_ELBO(num_particles=num_particles)

        elif autoguide == 'structured':
            self.guide = self.structured
            self.loss = TraceGraph_ELBO(num_particles=num_particles)
        else:
            self.guide = AutoDelta(self.model, **kwargs)
            self.loss = TraceMeanField_ELBO(num_particles=1)
    
    def __z(self, name):
        return sample(name + '.z', dist.Gamma(1/2, 1).expand([2]).to_event(1))
    
    def hyperprior(self, name, shape, layer, last=False):
        i, j = shape

        if not last:
            c_sqr_inv = sample(name + '.c^-2', dist.Gamma(2., 6.))
            z = self.__z(name) if i > 1 else jnp.ones(2)
            tau0_sqr = self.tau0 ** 2 * z[0]/z[1]

            u = sample(name + '.u_tau', dist.Gamma(1/2, 1).expand([i]).to_event(1))
            v = sample(name + '.v_tau', dist.Gamma(1/2, 1).expand([i]).to_event(1))

            _u = sample(name + '.u_lam', dist.Gamma(1/2, 1).expand([i, j]).to_event(1)) if not self.reduced else 1.
            _v = sample(name + '.v_lam', dist.Gamma(1/2, 1).expand([i, j]).to_event(1)) if not self.reduced else 1.
            
            psi = tau0_sqr * _v * jnp.expand_dims(v, -1)
            ksi = _u * jnp.expand_dims(u, -1)
            deterministic(name + '.tau_v', jnp.sqrt(psi / ksi))
            gamma = deterministic(name + '.gamma', jnp.sqrt(psi / (ksi + c_sqr_inv * psi)))
        else:
            z = self.__z(name)
            gamma = deterministic(name + '.gamma', jnp.sqrt( z[0]/z[1] ))

        return gamma

    def matrixnormal_weight_posterior(self, name, shape, **sample_kwargs):
        loc = param(name + '.loc', lambda rng_key: init_fn(rng_key, shape, radius=2.))
        i, j = shape

        A = param(name + '.A', jnp.eye(i)/10, constraint=constraints.softplus_lower_cholesky)
        B = param(name + '.B.T', jnp.eye(j)/10, constraint=constraints.softplus_lower_cholesky)
        
        return sample(name + '_base', dist.MatrixNormal(loc, A, B), **sample_kwargs)

    def structured(self, *args, with_hyperprior=False, **kwargs):
        L = len(self.layers)
        for l, layer in enumerate(self.layers):
            weight = layer.weight
            bias = layer.bias
            shape = weight.reshape(weight.shape[0], -1).shape
            if bias is not None:
                shape = (shape[0], shape[1] + 1)

            i, j = shape
            name = f'layer{l}.weight'
            if with_hyperprior:
                last = False if l == 0 else True
                last = False if l + 1 < L else last
                _i = i if self.reduced or l > 0 else i + 2
                if not last:
                    x = self.matrixnormal_weight_posterior(
                        f'layer{l}.aux', 
                        (_i, j + 2), 
                        infer={'is_auxiliary': True}
                    )

                    guide = AutoNormal(self.__z, prefix=f'auto.layer{l}')
                    z = guide(name)[name + '.z']

                    loc = param(name + '.c^-2.loc', jnp.zeros(1))
                    scale = param(name + '.c^-2.scale', jnp.ones(1)/10, constraint=constraints.positive)
                    sample(name + '.c^-2', dist.LogNormal(loc, scale))

                    if not self.reduced:
                        sample(name + '_base', dist.Delta(x[..., :i, :j], event_dim=2))

                        u = jnp.exp(x[..., :i, -2])
                        log_u = - x[..., :i, -2].sum(-1)
                        sample(name + '.u_tau', dist.Delta(u, log_density=log_u, event_dim=1))
                        
                        v = jnp.exp(x[..., :i, -1])
                        log_v = - x[..., :i, -1].sum(-1)
                        sample(name + '.v_tau', dist.Delta(v, log_density=log_v, event_dim=1))

                        if l == 0:
                            u = jnp.exp(x[..., -2, :j])
                            log_u = - x[..., -2, :j].sum(-1)
                            sample(name + '.u_lam', dist.Delta(u, log_density=log_u, event_dim=1))
                        
                            v = jnp.exp(x[..., -1, :j])
                            log_v = - x[..., -1, :j].sum(-1)
                            sample(name + '.v_lam', dist.Delta(v, log_density=log_v, event_dim=1))
                    
                    else:
                        sample(name + '_base', dist.Delta(x[..., :j], event_dim=2))
                        
                        u = jnp.exp(x[..., -2])
                        log_u = - x[..., -2].sum(-1)
                        sample(name + '.u_tau', dist.Delta(u, log_density=log_u, event_dim=1))
                        
                        v = jnp.exp(x[..., -1])
                        log_v = - x[..., -1].sum(-1)
                        sample(name + '.v_tau', dist.Delta(v, log_density=log_v, event_dim=1))

                else:
                    guide = AutoMultivariateNormal(self.hyperprior, prefix=f'auto.layer{l}')
                    smpl = guide(name, shape, l, last=last)
                    with handlers.block(), handlers.mask(mask=False):
                        gamma = handlers.condition(self.hyperprior, data=smpl)(name, shape, l, last=last)
                    
                    self.matrixnormal_weight_posterior(name, shape)
            else:
                gamma = self.gamma[name]                
                self.matrixnormal_weight_posterior(name, shape)


class BMRRegression(SVIRegression):
    posterior: str

    def __init__(
        self, 
        nnet, 
        *,
        regtype='linear', 
        tau0=1e-2,
        gamma0=1.,
        reduced=False,
        **kwargs
        ):
        super().__init__(
            nnet, 
            regtype=regtype,
            tau0=tau0,
            gamma0=gamma0,
            reduced=reduced
        )
        self.set_guide_loss_optim(**kwargs)

    def set_guide_loss_optim(self, **kwargs):
        num_particles = kwargs.pop('num_particles', 1)
        pruning = kwargs.pop('pruning', 'spike-and-slab')
        optimizer = kwargs.pop('optimizer', optax.adabelief)
        optim_kwargs = kwargs.pop('optim_kwargs', {'learning_rate': 1e-3})

        self.optim = optax_to_numpyro(optimizer(**optim_kwargs))
        self.posterior = kwargs.pop('posterior', 'normal')
        self.loss = TraceGraph_ELBO(num_particles=num_particles)

        if self.posterior == 'delta':
            self.guide = AutoDelta(self.model)
        if self.posterior == 'matrixnormal':
            self.reduced = True

        if pruning == 'spike-and-slab':
            self.pruning = self.prune_spike_and_slab

        else:
            self.pruning = self.prune_regularised_horseshoe
            autoguide = kwargs.pop('autoguide', 'mean-field')
            if autoguide == 'mean-field':
                self.rh_guide = AutoNormal(self.rh_model, **kwargs)

            elif autoguide == 'multivariate':
                self.rh_guide = AutoMultivariateNormal(self.rh_model, **kwargs)

            elif autoguide == 'lowrank-multivariate':
                self.rh_guide = AutoLowRankMultivariateNormal(self.rh_model, **kwargs)

            else:
                self.rh_guide = AutoDelta(self.rh_model, **kwargs)

    def normal_weight_posterior(self, name, shape):
        loc = param(
            name + '.loc', lambda rng_key: init_fn(rng_key, shape, radius=2.)
        )
        # scale = param(name + '.scale', jnp.ones(shape)/10, constraint=constraints.interval(1e-8, 1.5))
        scale = param(
            name + '.scale', jnp.full(shape, 0.1), constraint=constraints.softplus_positive
        )
        sample(name + '_base', dist.Normal(loc, scale).to_event(2))

    def multivariate_weight_posterior(self, name, shape):
        loc = param(
            name + '.loc', lambda rng_key: init_fn(rng_key, shape, radius=2.)
        )
        scale = param(
            name + '.scale', vmap(jnp.diag)(jnp.ones(shape)) / 10, constraint=constraints.softplus_lower_cholesky
        )
        sample(name + '_base', dist.MultivariateNormal(loc, scale_tril=scale).to_event(1))

    def get_weight_posterior(self, name, shape):
        if self.posterior == 'normal':
            return self.normal_weight_posterior(name, shape)

        elif self.posterior == 'multivariate':
            return self.multivariate_weight_posterior(name, shape)

        elif self.posterior == 'matrixnormal':
            return self.matrixnormal_weight_posterior(name, shape)

        else:
            raise NotImplementedError

    def __lognormal(self, name, shape):
        loc = param(name + '.loc', lambda rng_key: init_fn(rng_key, shape, radius=2.) )
        scale = param(name + '.scale', jnp.full(shape, 0.1), constraint=constraints.softplus_positive)
        return dist.LogNormal(loc, scale)

    def guide(self, *args, **kwargs):
        if self.type == 'linear':
            sigma_sqr_inv = sample('sigma^-2', self.__lognormal('sigma^-2', (1,)))
            sigma = 1/jnp.sqrt(sigma_sqr_inv)
        else:
            sigma = jnp.ones(1)

        L = len(self.layers)
        for l, layer in enumerate(self.layers):         
            s = 1. if l + 1 < L else sigma
            weight = layer.weight
            bias = layer.bias
            shape = weight.reshape(weight.shape[0], -1).shape
            if bias is not None:
                shape = (shape[0], shape[1] + 1)
            
            name = f'layer{l}.weight'
            sample(f'{name}.c_inv_sqr', self.__lognormal(f'{name}.c_inv_sqr', ()) )
            self.get_weight_posterior(name, shape)

    def ΔF(self, mu, P, gamma, sigma_sqr=1):
        if self.posterior == 'normal':
            return ΔF_mf(mu, P, gamma, sigma_sqr)
        elif self.posterior == 'multivariate':
            return ΔF_mv(mu, P, gamma, sigma_sqr)
        elif self.posterior == 'matrixnormal':
            return ΔF_mn(mu, P[0], P[1], gamma, sigma_sqr)
        else:
            raise NotImplementedError

    def sufficient_stats(self, name, params, invert=[]):
        '''Multivariate normal guide'''
        if len(invert) == 0:
            if self.posterior == 'normal':
                pi = 1 / params[name + '.scale'] ** 2
                mu = params[name + '.loc']
                return (mu, pi)

            elif self.posterior == 'multivariate':
                mu = params[name + '.loc']
                L_inv = vmap(jnp.linalg.inv)(params[name + '.scale'])
                P = jnp.matmul(L_inv.transpose((0, -1, -2)), L_inv)
                return (mu, P)

            elif self.posterior == 'matrixnormal':
                mu = params[name + '.loc']
                B = params[name + '.B.T'].T
                a = params[name + '.a']
                return (mu, (a, B))
            else:
                raise NotImplementedError
        
        else:
            if self.posterior in ['normal', 'matrixnormal']:
                deterministic(f'{name}.loc', invert[0])
                deterministic(f'{name}.scale', invert[1])

            elif self.posterior == 'multivariate':
                deterministic(f'{name}.loc', invert[0])
                deterministic(f'{name}.scale', vmap(lambda x: jnp.linalg.inv(jnp.linalg.cholesky(x)).T)(invert[1]))
            
            else:
                raise NotImplementedError

    def prune_spike_and_slab(self, rng_key, params, **kwargs):
        assert self.posterior == 'normal'

        del_f  = vmap(lambda mu, pi: ΔF_mf(mu, pi, 1e-16))
        for l in range(len(self.layers) - 1):
            name = f'layer{l}.weight'
            mu, pi = self.sufficient_stats(name, params)

            df, _, _ = del_f(mu.reshape(-1), pi.reshape(-1))
            df = df.reshape(mu.shape)
            alpha_0 = 1.
            beta_0 = 1.
            def step_fn(carry, t):
                zeta_k = carry
                q = nn.sigmoid( zeta_k - df)

                alpha = alpha_0 + q.sum()
                beta = beta_0 + (1 - q).sum()

                zeta_k = zeta(alpha, beta)

                return zeta_k, None
            
            zeta_0 = zeta(alpha_0, beta_0)

            zeta_k, _ = lax.scan(step_fn, zeta_0, jnp.arange(4))

            active_weights = df <= zeta_k  # alternatively just use df <= 0.

            self.gamma[name] = self.gamma[name] * active_weights + 1e-8 * ~active_weights
        
        return self.gamma
    
    def rh_model(self, params):
        L = len(self.layers)
        for l, layer in enumerate(self.layers):
            name = f'layer{l}.weight'
            last = False if l == 0 else True
            last = False if l + 1 < L else last
            mu_n, P_n = self.sufficient_stats(name, params)
            shape = mu_n.shape

            gamma = self.hyperprior(name, shape, l, last=last)
        
            if self.posterior == 'matrixnormal':
                gamma_sqr = jnp.broadcast_to(jnp.square(gamma.squeeze()), shape[:1])
                log_prob, t_mu, t_sig = self.ΔF(mu_n, P_n, gamma_sqr)
            else:
                gamma_sqr = jnp.broadcast_to(jnp.square(gamma), shape )
                log_prob, t_mu, t_sig = vmap(self.ΔF)(mu_n, P_n, gamma_sqr)
                log_prob = log_prob.sum()
            
            factor(f'layer{l}.weight.log_prob', log_prob)
        
    def prune_regularised_horseshoe(self, rng_key, params, num_particles=16, delta=1e-6, num_iters=5_000, **kwargs):
        
        model = self.rh_model
        guide = self.rh_guide
        loss = TraceGraph_ELBO(num_particles=num_particles)
        optim = self.optim

        svi = SVI(model, guide, optim, loss)
        rng_key, key = random.split(rng_key)
        res = svi.run(rng_key, num_iters, params, progress_bar=False, stable_update=True)
        
        pred = Predictive(model, guide=guide, params=res.params, num_samples=1000)
        rng_key, key = random.split(rng_key)
        smpl = lax.stop_gradient( pred(key, params) )

        for l, layer in enumerate(self.layers):
            name = f'layer{l}.weight'
            last = False if l == 0 else True
            last = False if l + 1 < len(self.layers) else last

            key = name + '.gamma'
            gamma = self.gamma[name] / jnp.sqrt( (1 / jnp.square(smpl[key])).mean(0) )
            if not last:
                self.gamma[name] = jnp.where(gamma < delta, 1e-16, gamma)

        return self.gamma