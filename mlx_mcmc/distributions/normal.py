"""Normal (Gaussian) distribution."""

import mlx.core as mx
from mlx_mcmc.distributions.base import Distribution


class Normal(Distribution):
    """Normal (Gaussian) distribution.

    The probability density function is:
        p(x | μ, σ) = (1 / (σ√(2π))) exp(-(x - μ)² / (2σ²))

    Parameters
    ----------
    loc : float or mlx.core.array
        Mean (location parameter)
    scale : float or mlx.core.array
        Standard deviation (scale parameter), must be positive

    Examples
    --------
    >>> dist = Normal(0, 1)  # Standard normal
    >>> log_p = dist.log_prob(mx.array(0.0))  # Log prob at zero
    >>> samples = dist.sample(mx.random.key(0), shape=(1000,))
    """

    def __init__(self, loc, scale):
        self.loc = mx.array(loc)
        self.scale = mx.array(scale)
        self._log_scale = mx.log(self.scale)
        self._log_norm = -0.5 * mx.log(2 * mx.pi)

    def log_prob(self, value):
        """
        Compute log probability density.

        log p(x) = -0.5 * log(2π) - log(σ) - 0.5 * ((x - μ) / σ)²

        Parameters
        ----------
        value : float or mlx.core.array
            Value at which to evaluate log probability

        Returns
        -------
        log_prob : mlx.core.array
            Log probability density at value
        """
        value = mx.array(value)
        var = self.scale ** 2
        log_prob = (
            self._log_norm
            - self._log_scale
            - 0.5 * ((value - self.loc) ** 2) / var
        )
        return log_prob

    def sample(self, key, shape=()):
        """
        Sample from normal distribution.

        Uses MLX's built-in normal sampling with transformation:
        X = μ + σZ, where Z ~ N(0, 1)

        Parameters
        ----------
        key : mlx.core.array
            Random key for sampling
        shape : tuple, optional
            Shape of samples to draw

        Returns
        -------
        samples : mlx.core.array
            Samples from Normal(loc, scale)
        """
        return mx.random.normal(shape, key=key) * self.scale + self.loc

    def __repr__(self):
        return f"Normal(loc={float(self.loc):.3f}, scale={float(self.scale):.3f})"
