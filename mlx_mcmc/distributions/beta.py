"""Beta distribution."""

import mlx.core as mx
import numpy as np
from scipy.special import gammaln
from mlx_mcmc.distributions.base import Distribution


class Beta(Distribution):
    """Beta distribution.

    The Beta distribution is a continuous probability distribution on the interval (0, 1).
    It's commonly used for modeling probabilities, proportions, and rates.

    Parameters
    ----------
    alpha : float or array_like
        First shape parameter (concentration1), must be positive
    beta : float or array_like
        Second shape parameter (concentration2), must be positive

    Examples
    --------
    >>> from mlx_mcmc import Beta
    >>> import mlx.core as mx
    >>>
    >>> # Uniform prior on [0, 1]
    >>> prior = Beta(1, 1)
    >>>
    >>> # Skeptical prior (favors values near 0.5)
    >>> prior = Beta(5, 5)
    >>>
    >>> # Prior favoring high probabilities
    >>> prior = Beta(8, 2)
    """

    def __init__(self, alpha, beta):
        """Initialize Beta distribution.

        Parameters
        ----------
        alpha : float or array_like
            First shape parameter (concentration1), must be positive
        beta : float or array_like
            Second shape parameter (concentration2), must be positive
        """
        self.alpha = mx.array(alpha)
        self.beta = mx.array(beta)

        # Precompute log normalizing constant: log(B(alpha, beta))
        # log B(a, b) = log Γ(a) + log Γ(b) - log Γ(a+b)
        # Use scipy's gammaln for numerical stability
        alpha_val = float(self.alpha) if self.alpha.size == 1 else self.alpha
        beta_val = float(self.beta) if self.beta.size == 1 else self.beta
        self._log_beta_const = mx.array(
            gammaln(alpha_val) + gammaln(beta_val) - gammaln(alpha_val + beta_val)
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

        # Beta PDF: x^(alpha-1) * (1-x)^(beta-1) / B(alpha, beta)
        # log PDF: (alpha-1)*log(x) + (beta-1)*log(1-x) - log B(alpha, beta)

        # Check if value is in support [0, 1]
        log_prob = (
            (self.alpha - 1) * mx.log(value) +
            (self.beta - 1) * mx.log(1 - value) -
            self._log_beta_const
        )

        # Return -inf for values outside [0, 1]
        log_prob = mx.where(
            (value > 0) & (value < 1),
            log_prob,
            mx.array(-mx.inf)
        )

        return log_prob

    def sample(self, key, shape=()):
        """Draw samples from the distribution.

        Uses numpy's beta sampler for now (MLX doesn't have gamma sampling yet).

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
        # Use numpy for sampling (MLX doesn't have beta/gamma sampling yet)
        # Convert key to numpy seed by generating a random integer
        seed = int(mx.random.randint(0, 2**31 - 1, key=key))
        rng = np.random.default_rng(seed)

        alpha_val = float(self.alpha)
        beta_val = float(self.beta)
        samples_np = rng.beta(alpha_val, beta_val, size=shape)

        return mx.array(samples_np)

    def mean(self):
        """Compute the mean of the distribution.

        Returns
        -------
        mean : array_like
            Mean: alpha / (alpha + beta)
        """
        return self.alpha / (self.alpha + self.beta)

    def variance(self):
        """Compute the variance of the distribution.

        Returns
        -------
        variance : array_like
            Variance: (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
        """
        ab_sum = self.alpha + self.beta
        return (self.alpha * self.beta) / (ab_sum ** 2 * (ab_sum + 1))

    def mode(self):
        """Compute the mode of the distribution.

        Returns
        -------
        mode : array_like
            Mode: (alpha - 1) / (alpha + beta - 2) if alpha, beta > 1
        """
        # Mode only exists if both alpha, beta > 1
        return (self.alpha - 1) / (self.alpha + self.beta - 2)
