"""Exponential distribution."""

import mlx.core as mx
from mlx_mcmc.distributions.base import Distribution


class Exponential(Distribution):
    """Exponential distribution.

    The Exponential distribution is a continuous probability distribution for positive
    real numbers. It's commonly used for modeling waiting times, time until events,
    and rates.

    This is a special case of the Gamma distribution with shape parameter alpha = 1.

    Parameters
    ----------
    rate : float or array_like
        Rate parameter (lambda), must be positive. Higher rate = shorter wait times.

    Notes
    -----
    Mean = 1 / rate
    Variance = 1 / rate^2

    Examples
    --------
    >>> from mlx_mcmc import Exponential
    >>> import mlx.core as mx
    >>>
    >>> # Average wait time of 2 units (rate = 0.5)
    >>> wait_time = Exponential(0.5)
    >>>
    >>> # Fast events (rate = 10, mean wait = 0.1)
    >>> fast_events = Exponential(10)
    """

    def __init__(self, rate):
        """Initialize Exponential distribution.

        Parameters
        ----------
        rate : float or array_like
            Rate parameter (lambda), must be positive
        """
        self.rate = mx.array(rate)

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

        # Exponential PDF: lambda * exp(-lambda * x)
        # log PDF: log(lambda) - lambda * x

        log_prob = mx.log(self.rate) - self.rate * value

        # Return -inf for negative values
        log_prob = mx.where(value >= 0, log_prob, mx.array(-mx.inf))

        return log_prob

    def sample(self, key, shape=()):
        """Draw samples from the distribution.

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
        # Sample from Uniform(0, 1) and use inverse CDF
        # Exponential CDF^-1(u) = -log(1 - u) / lambda
        u = mx.random.uniform(shape=shape, key=key)
        return -mx.log(1 - u) / self.rate

    def mean(self):
        """Compute the mean of the distribution.

        Returns
        -------
        mean : array_like
            Mean: 1 / rate
        """
        return 1.0 / self.rate

    def variance(self):
        """Compute the variance of the distribution.

        Returns
        -------
        variance : array_like
            Variance: 1 / rate^2
        """
        return 1.0 / (self.rate ** 2)

    def mode(self):
        """Compute the mode of the distribution.

        Returns
        -------
        mode : array_like
            Mode: 0 (always)
        """
        return mx.array(0.0)

    def median(self):
        """Compute the median of the distribution.

        Returns
        -------
        median : array_like
            Median: log(2) / rate
        """
        return mx.log(mx.array(2.0)) / self.rate
