"""
MLX-MCMC: Bayesian Inference for Apple Silicon

A lightweight probabilistic programming library built on Apple's MLX framework,
providing native Metal GPU acceleration for MCMC sampling.

Example:
    >>> from mlx_mcmc import Normal, HalfNormal, MCMC
    >>>
    >>> def log_prob(params):
    ...     mu = params['mu']
    ...     sigma = params['sigma']
    ...     return Normal(0, 10).log_prob(mu) + HalfNormal(5).log_prob(sigma)
    >>>
    >>> mcmc = MCMC(log_prob)
    >>> samples = mcmc.run({'mu': 0.0, 'sigma': 1.0}, num_samples=1000)
"""

__version__ = "0.1.0-alpha"
__author__ = "Claude Code Community"
__license__ = "MIT"

# Import core components
from mlx_mcmc.distributions.normal import Normal
from mlx_mcmc.distributions.halfnormal import HalfNormal
from mlx_mcmc.kernels.metropolis import metropolis_hastings
from mlx_mcmc.kernels.hmc import hmc
from mlx_mcmc.inference.mcmc import MCMC

__all__ = [
    "Normal",
    "HalfNormal",
    "metropolis_hastings",
    "hmc",
    "MCMC",
]
