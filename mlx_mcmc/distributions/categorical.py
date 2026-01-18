"""Categorical distribution."""

import mlx.core as mx
from mlx_mcmc.distributions.base import Distribution


class Categorical(Distribution):
    """Categorical distribution.

    The Categorical distribution is a discrete probability distribution for a random
    variable that can take one of K possible outcomes. It's commonly used for
    classification, choice modeling, and discrete outcomes.

    Parameters
    ----------
    probs : array_like, optional
        Probabilities for each category. Must sum to 1.
        Either probs or logits must be specified, but not both.
    logits : array_like, optional
        Unnormalized log probabilities. Will be normalized via softmax.
        Either probs or logits must be specified, but not both.

    Examples
    --------
    >>> from mlx_mcmc import Categorical
    >>> import mlx.core as mx
    >>>
    >>> # Uniform over 3 categories
    >>> cat = Categorical(probs=[1/3, 1/3, 1/3])
    >>>
    >>> # Favoring first category
    >>> cat = Categorical(probs=[0.7, 0.2, 0.1])
    >>>
    >>> # Using logits (unnormalized)
    >>> cat = Categorical(logits=[2.0, 1.0, 0.5])
    """

    def __init__(self, probs=None, logits=None):
        """Initialize Categorical distribution.

        Parameters
        ----------
        probs : array_like, optional
            Probabilities for each category. Must sum to 1.
        logits : array_like, optional
            Unnormalized log probabilities.
        """
        if probs is None and logits is None:
            raise ValueError("Either probs or logits must be specified")
        if probs is not None and logits is not None:
            raise ValueError("Only one of probs or logits can be specified")

        if probs is not None:
            self.probs = mx.array(probs)
            # Normalize to ensure sum to 1
            self.probs = self.probs / mx.sum(self.probs)
            self.logits = mx.log(self.probs)
        else:
            self.logits = mx.array(logits)
            # Compute probs via softmax
            # softmax(x) = exp(x) / sum(exp(x))
            # For numerical stability: softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
            logits_max = mx.max(self.logits)
            exp_logits = mx.exp(self.logits - logits_max)
            self.probs = exp_logits / mx.sum(exp_logits)

        self.num_categories = self.probs.shape[0]

    def log_prob(self, value):
        """Compute log probability mass.

        Parameters
        ----------
        value : array_like
            Category index (0 to K-1) at which to evaluate the log probability

        Returns
        -------
        log_prob : array_like
            Log probability mass
        """
        value = mx.array(value, dtype=mx.int32)

        # Check if value is valid category index
        valid = (value >= 0) & (value < self.num_categories)

        # Get log probability for the category
        log_p = self.logits[value]

        # Return -inf for invalid indices
        log_p = mx.where(valid, log_p, mx.array(-mx.inf))

        return log_p

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
            Category indices sampled from the distribution
        """
        # Sample from Uniform(0, 1)
        u = mx.random.uniform(shape=shape, key=key)

        # Compute cumulative probabilities
        cumsum = mx.cumsum(self.probs)

        # Find which category the uniform sample falls into
        # Use broadcasting to compare u with cumsum
        if len(shape) == 0:
            # Scalar case
            sample = mx.sum((u > cumsum).astype(mx.int32))
        else:
            # Vectorized case
            # Expand dimensions for broadcasting
            u_expanded = mx.expand_dims(u, -1)
            cumsum_expanded = mx.expand_dims(cumsum, 0)
            sample = mx.sum((u_expanded > cumsum_expanded).astype(mx.int32), axis=-1)

        return sample

    def entropy(self):
        """Compute the entropy of the distribution.

        Returns
        -------
        entropy : array_like
            Entropy: -sum(p * log(p))
        """
        # Avoid log(0) by only including non-zero probabilities
        log_probs = mx.where(self.probs > 0, mx.log(self.probs), mx.array(0.0))
        return -mx.sum(self.probs * log_probs)

    def mode(self):
        """Compute the mode of the distribution.

        Returns
        -------
        mode : array_like
            Category index with highest probability
        """
        return mx.argmax(self.probs)
