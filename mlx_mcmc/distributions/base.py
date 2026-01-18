"""Base class for probability distributions."""

import mlx.core as mx


class Distribution:
    """Base class for all probability distributions.

    All distributions must implement:
    - log_prob(value): Compute log probability density/mass
    - sample(key, shape): Draw samples from the distribution
    """

    def log_prob(self, value):
        """
        Compute log probability density or mass function.

        Parameters
        ----------
        value : mlx.core.array or float
            Value(s) at which to evaluate log probability

        Returns
        -------
        log_prob : mlx.core.array
            Log probability at the given value(s)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement log_prob()"
        )

    def sample(self, key, shape=()):
        """
        Draw samples from the distribution.

        Parameters
        ----------
        key : mlx.core.array
            Random key for sampling
        shape : tuple, optional
            Shape of samples to draw

        Returns
        -------
        samples : mlx.core.array
            Samples from the distribution
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement sample()"
        )

    def __repr__(self):
        """String representation of the distribution."""
        return f"{self.__class__.__name__}()"
