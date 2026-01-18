"""Metropolis-Hastings MCMC sampler."""

import mlx.core as mx


def metropolis_hastings(
    log_prob_fn,
    initial_params,
    num_samples=1000,
    proposal_scale=0.1,
    random_seed=0,
    verbose=False
):
    """
    Metropolis-Hastings MCMC sampler.

    Uses a Gaussian random walk proposal distribution:
        θ' = θ + ε, where ε ~ N(0, proposal_scale²I)

    Parameters
    ----------
    log_prob_fn : callable
        Function that computes log probability given parameters dict
    initial_params : dict
        Initial parameter values {name: value}
    num_samples : int, optional
        Number of samples to draw (default: 1000)
    proposal_scale : float, optional
        Scale of Gaussian proposal distribution (default: 0.1)
    random_seed : int, optional
        Random seed for reproducibility (default: 0)
    verbose : bool, optional
        If True, print progress updates (default: False)

    Returns
    -------
    samples : dict
        Dictionary of parameter samples {name: list of values}
    acceptance_rate : float
        Fraction of proposals that were accepted

    Examples
    --------
    >>> def log_prob(params):
    ...     return Normal(0, 1).log_prob(params['x'])
    >>> samples, accept_rate = metropolis_hastings(
    ...     log_prob, {'x': 0.0}, num_samples=1000
    ... )
    """
    # Initialize random key
    key = mx.random.key(random_seed)

    # Convert initial params to MLX arrays
    current_params = {k: mx.array(v) for k, v in initial_params.items()}
    current_log_prob = log_prob_fn(current_params)

    # Storage for samples
    samples = {k: [] for k in current_params.keys()}
    n_accepted = 0

    if verbose:
        print(f"Running {num_samples} Metropolis-Hastings iterations...")

    for i in range(num_samples):
        # Generate proposal by random walk
        key, *subkeys = mx.random.split(key, len(current_params) + 1)

        proposed_params = {}
        for (param_name, param_value), subkey in zip(
            current_params.items(), subkeys
        ):
            # Gaussian random walk proposal
            noise = mx.random.normal(param_value.shape, key=subkey) * proposal_scale
            proposed_params[param_name] = param_value + noise

        # Compute acceptance probability
        proposed_log_prob = log_prob_fn(proposed_params)
        log_accept_ratio = proposed_log_prob - current_log_prob

        # Accept/reject
        key, subkey = mx.random.split(key)
        log_u = mx.log(mx.random.uniform(key=subkey))

        if float(log_u) < float(log_accept_ratio):
            # Accept proposal
            current_params = proposed_params
            current_log_prob = proposed_log_prob
            n_accepted += 1

        # Store current sample (accepted or rejected proposal)
        for param_name, param_value in current_params.items():
            samples[param_name].append(float(param_value))

        # Progress indicator
        if verbose and (i + 1) % 500 == 0:
            print(f"  Iteration {i+1}/{num_samples} "
                  f"(accept rate: {n_accepted/(i+1):.2%})")

    acceptance_rate = n_accepted / num_samples

    return samples, acceptance_rate
