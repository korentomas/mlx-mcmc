"""Half-normal distribution (for positive parameters)."""

import mlx.core as mx
from mlx_mcmc.distributions.base import Distribution


class HalfNormal(Distribution):
    """Half-normal distribution.

    The half-normal is the distribution of |X| where X ~ Normal(0, σ).
    Useful for positive parameters like standard deviations.

    The probability density function is:
        p(x | σ) = (2 / (σ√(2π))) exp(-x² / (2σ²)) for x >= 0

    Parameters
    ----------
    scale : float or mlx.core.array
        Scale parameter, must be positive

    Examples
    --------
    >>> dist = HalfNormal(5.0)  # Half-normal with scale 5
    >>> log_p = dist.log_prob(mx.array(2.0))
    >>> samples = dist.sample(mx.random.key(0), shape=(1000,))
    """

    def __init__(self, scale):
        self.scale = mx.array(scale)
        self._log_scale = mx.log(self.scale)
        self._log_norm = -0.5 * mx.log(2 * mx.pi)
        self._log2 = mx.log(mx.array(2.0))

    def log_prob(self, value):
        """
        Compute log probability density.

        log p(x) = log(2) - 0.5*log(2π) - log(σ) - 0.5*(x/σ)²  for x >= 0
                 = -∞  for x < 0

        Parameters
        ----------
        value : float or mlx.core.array
            Value at which to evaluate log probability

        Returns
        -------
        log_prob : mlx.core.array
            Log probability density at value (−∞ for negative values)
        """
        value = mx.array(value)
        var = self.scale ** 2

        # Log probability for positive values
        log_prob_pos = (
            self._log2
            + self._log_norm
            - self._log_scale
            - 0.5 * (value ** 2) / var
        )

        # Return -inf for negative values (zero probability)
        return mx.where(value >= 0, log_prob_pos, mx.array(-mx.inf))

    def sample(self, key, shape=()):
        """
        Sample from half-normal distribution.

        Uses absolute value of normal: |Z * σ| where Z ~ N(0, 1)

        Parameters
        ----------
        key : mlx.core.array
            Random key for sampling
        shape : tuple, optional
            Shape of samples to draw

        Returns
        -------
        samples : mlx.core.array
            Samples from HalfNormal(scale)
        """
        return mx.abs(mx.random.normal(shape, key=key) * self.scale)

    def __repr__(self):
        return f"HalfNormal(scale={float(self.scale):.3f})"
