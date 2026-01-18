"""Gamma distribution."""

import mlx.core as mx
import numpy as np
from scipy.special import gammaln
from mlx_mcmc.distributions.base import Distribution


class Gamma(Distribution):
    """Gamma distribution.

    The Gamma distribution is a continuous probability distribution for positive real numbers.
    It's commonly used for modeling waiting times, rates, and positive continuous variables.

    Parameters
    ----------
    alpha : float or array_like
        Shape parameter, must be positive
    beta : float or array_like, optional
        Rate parameter (inverse scale), must be positive. Default: 1.0

    Notes
    -----
    This uses the shape-rate parameterization. The scale parameter is 1/beta.
    Mean = alpha / beta
    Variance = alpha / beta^2

    Examples
    --------
    >>> from mlx_mcmc import Gamma
    >>> import mlx.core as mx
    >>>
    >>> # Exponential-like (shape=1)
    >>> rate_prior = Gamma(1, 1)
    >>>
    >>> # More concentrated around mean
    >>> rate_prior = Gamma(10, 2)  # mean = 5
    """

    def __init__(self, alpha, beta=1.0):
        """Initialize Gamma distribution.

        Parameters
        ----------
        alpha : float or array_like
            Shape parameter, must be positive
        beta : float or array_like, optional
            Rate parameter (inverse scale), must be positive. Default: 1.0
        """
        self.alpha = mx.array(alpha)
        self.beta = mx.array(beta)

        # Precompute log normalizing constant
        # log Z = alpha*log(beta) - log Γ(alpha)
        alpha_val = float(self.alpha) if self.alpha.size == 1 else self.alpha
        self._log_norm = (
            self.alpha * mx.log(self.beta) -
            mx.array(gammaln(alpha_val))
        )

    def log_prob(self, value):
        """Compute log probability density.

        Parameters
        ----------
        value : array_like
            Value at which to evaluate the log probability

        Returns
        -------
        log_prob : array_like
            Log probability density
        """
        value = mx.array(value)

        # Gamma PDF: beta^alpha * x^(alpha-1) * exp(-beta*x) / Γ(alpha)
        # log PDF: alpha*log(beta) + (alpha-1)*log(x) - beta*x - log Γ(alpha)

        log_prob = (
            self._log_norm +
            (self.alpha - 1) * mx.log(value) -
            self.beta * value
        )

        # Return -inf for negative values
        log_prob = mx.where(value > 0, log_prob, mx.array(-mx.inf))

        return log_prob

    def sample(self, key, shape=()):
        """Draw samples from the distribution.

        Uses numpy's gamma sampler for now (MLX doesn't have gamma sampling yet).

        Parameters
        ----------
        key : array_like
            Random key for sampling
        shape : tuple, optional
            Shape of samples to draw

        Returns
        -------
        samples : array_like
            Samples from the distribution
        """
        # Use numpy for sampling (MLX doesn't have gamma sampling yet)
        # Convert key to numpy seed by generating a random integer
        seed = int(mx.random.randint(0, 2**31 - 1, key=key))
        rng = np.random.default_rng(seed)

        alpha_val = float(self.alpha)
        # numpy uses shape-scale parameterization, we use shape-rate
        scale = 1.0 / float(self.beta)
        samples_np = rng.gamma(alpha_val, scale=scale, size=shape)

        return mx.array(samples_np)

    def mean(self):
        """Compute the mean of the distribution.

        Returns
        -------
        mean : array_like
            Mean: alpha / beta
        """
        return self.alpha / self.beta

    def variance(self):
        """Compute the variance of the distribution.

        Returns
        -------
        variance : array_like
            Variance: alpha / beta^2
        """
        return self.alpha / (self.beta ** 2)

    def mode(self):
        """Compute the mode of the distribution.

        Returns
        -------
        mode : array_like
            Mode: (alpha - 1) / beta if alpha >= 1, else 0
        """
        # Mode only exists if alpha >= 1
        mode = (self.alpha - 1) / self.beta
        return mx.where(self.alpha >= 1, mode, mx.array(0.0))
