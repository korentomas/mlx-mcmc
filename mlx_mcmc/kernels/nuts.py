"""No-U-Turn Sampler (NUTS) implementation using MLX.

Based on Hoffman & Gelman (2014): "The No-U-Turn Sampler: Adaptively
Setting Path Lengths in Hamiltonian Monte Carlo"
"""

import mlx.core as mx
from typing import Callable, Tuple, Dict, Any
import math


# Constants
DELTA_MAX = 1000.0  # Maximum energy difference to prevent infinite trajectories


def nuts(
    log_prob_fn: Callable,
    initial_params: Dict[str, mx.array],
    num_samples: int = 1000,
    num_warmup: int = 1000,
    step_size: float = 0.1,
    max_tree_depth: int = 10,
    adapt_step_size: bool = True,
    target_accept: float = 0.65,
    key: mx.array = None,
) -> Tuple[Dict[str, mx.array], float]:
    """No-U-Turn Sampler (NUTS) for efficient HMC sampling.

    NUTS automatically tunes the trajectory length by building a binary tree
    of states and stopping when the trajectory starts to turn back on itself.
    This eliminates the need to manually specify num_leapfrog_steps.

    Args:
        log_prob_fn: Function that computes log probability given parameters.
        initial_params: Dictionary of initial parameter values.
        num_samples: Number of samples to draw after warmup.
        num_warmup: Number of warmup samples for step size adaptation.
        step_size: Initial step size for leapfrog integrator.
        max_tree_depth: Maximum tree depth (2^depth leapfrog steps).
        adapt_step_size: Whether to adapt step size during warmup.
        target_accept: Target acceptance rate for step size adaptation (0.65 typical).
        key: Random key for reproducibility.

    Returns:
        samples: Dictionary mapping parameter names to arrays of samples.
        acceptance_rate: Final acceptance rate.

    References:
        Hoffman, M. D., & Gelman, A. (2014). "The No-U-Turn Sampler".
        JMLR 15(1), 1593-1623.
    """
    if key is None:
        key = mx.random.key(0)

    # Initialize
    current_params = {k: mx.array(v) for k, v in initial_params.items()}
    param_names = list(current_params.keys())

    # Storage for samples
    all_samples = {name: [] for name in param_names}

    # Adaptation state
    mu = mx.log(10 * step_size)  # Dual averaging target
    epsilon_bar = 1.0
    H_bar = 0.0
    gamma = 0.05
    t0 = 10.0
    kappa = 0.75

    # Statistics
    n_accept = 0
    n_total = 0
    total_tree_depth = 0

    # Define gradient function
    def grad_log_prob(params_dict):
        """Compute gradients of log probability with respect to all parameters."""
        def log_prob_flat(*param_values):
            params = {name: val for name, val in zip(param_names, param_values)}
            return log_prob_fn(params)

        param_list = [params_dict[name] for name in param_names]
        grads = mx.grad(log_prob_flat, argnums=list(range(len(param_names))))(*param_list)

        if not isinstance(grads, tuple):
            grads = (grads,)
        return {name: grad for name, grad in zip(param_names, grads)}

    def leapfrog_step(params, momentum, epsilon):
        """Single leapfrog integration step."""
        # Half step for momentum
        grads = grad_log_prob(params)
        momentum_half = {
            name: momentum[name] + 0.5 * epsilon * grads[name]
            for name in param_names
        }

        # Full step for position
        params_new = {
            name: params[name] + epsilon * momentum_half[name]
            for name in param_names
        }

        # Half step for momentum
        grads_new = grad_log_prob(params_new)
        momentum_new = {
            name: momentum_half[name] + 0.5 * epsilon * grads_new[name]
            for name in param_names
        }

        return params_new, momentum_new

    def hamiltonian(params, momentum):
        """Compute Hamiltonian H(q, p) = -log p(q) + 0.5 * ||p||^2."""
        potential_energy = -log_prob_fn(params)
        kinetic_energy = 0.5 * sum(mx.sum(p ** 2) for p in momentum.values())
        return potential_energy + kinetic_energy

    def no_u_turn(theta_minus, theta_plus, r_minus, r_plus):
        """Check if trajectory has made a U-turn (generalized criterion).

        A U-turn is detected when the trajectory starts coming back toward itself.
        """
        # Compute position difference
        delta_theta = {name: theta_plus[name] - theta_minus[name] for name in param_names}

        # Check dot products
        dot_minus = sum(
            mx.sum(delta_theta[name] * r_minus[name]) for name in param_names
        )
        dot_plus = sum(
            mx.sum(delta_theta[name] * r_plus[name]) for name in param_names
        )

        return float(dot_minus) >= 0 and float(dot_plus) >= 0

    def build_tree(theta, r, u, v, j, epsilon, theta0, r0, rng_key):
        """Recursively build tree of states using doubling procedure.

        Args:
            theta: Current position
            r: Current momentum
            u: Slice variable
            v: Direction (+1 forward, -1 backward)
            j: Tree depth (0 = single step, 1 = 2 steps, 2 = 4 steps, etc.)
            epsilon: Step size
            theta0: Initial position (for U-turn check)
            r0: Initial momentum (for U-turn check)
            rng_key: Random key for sampling

        Returns:
            theta_minus, theta_plus: Left and right boundaries of tree
            r_minus, r_plus: Momentum at boundaries
            theta_prime: Candidate sample from tree
            n_prime: Number of valid states in tree
            s_prime: Is tree still valid (no U-turn, in slice)?
            alpha: Sum of acceptance probabilities (for adaptation)
            n_alpha: Number of acceptance decisions
        """
        if j == 0:
            # Base case: single leapfrog step
            theta_prime, r_prime = leapfrog_step(theta, r, v * epsilon)

            # Check if new state is in slice
            H_prime = hamiltonian(theta_prime, r_prime)
            n_prime = 1 if float(mx.log(u)) <= float(-H_prime) else 0

            # Check if trajectory diverged
            H0 = hamiltonian(theta0, r0)
            s_prime = float(mx.log(u)) < float(DELTA_MAX - H_prime)

            # Acceptance probability for adaptation
            alpha = min(1.0, float(mx.exp(-H_prime + H0)))
            n_alpha = 1

            return (theta_prime, theta_prime,  # Boundaries same for single step
                    r_prime, r_prime,
                    theta_prime, n_prime, s_prime,
                    alpha, n_alpha)
        else:
            # Recursion: build left and right subtrees
            rng_key, subkey1, subkey2 = mx.random.split(rng_key, 3)

            (theta_minus, theta_plus, r_minus, r_plus,
             theta_prime, n_prime, s_prime,
             alpha_prime, n_alpha_prime) = build_tree(theta, r, u, v, j - 1, epsilon, theta0, r0, subkey1)

            if s_prime:
                # Build adjacent subtree
                if v == -1:
                    # Build left
                    (theta_minus, _, r_minus, _,
                     theta_double_prime, n_double_prime, s_double_prime,
                     alpha_double_prime, n_alpha_double_prime) = build_tree(
                         theta_minus, r_minus, u, v, j - 1, epsilon, theta0, r0, subkey2)
                else:
                    # Build right
                    (_, theta_plus, _, r_plus,
                     theta_double_prime, n_double_prime, s_double_prime,
                     alpha_double_prime, n_alpha_double_prime) = build_tree(
                         theta_plus, r_plus, u, v, j - 1, epsilon, theta0, r0, subkey2)

                # Sample uniformly from both subtrees (Metropolis-Hastings on trees)
                rng_key, subkey3 = mx.random.split(rng_key)
                if float(mx.random.uniform(key=subkey3)) < n_double_prime / max(n_prime + n_double_prime, 1.0):
                    theta_prime = theta_double_prime

                # Update counts
                n_prime += n_double_prime
                alpha_prime += alpha_double_prime
                n_alpha_prime += n_alpha_double_prime

                # Check U-turn
                s_prime = s_double_prime and no_u_turn(theta_minus, theta_plus, r_minus, r_plus)

            return (theta_minus, theta_plus, r_minus, r_plus,
                    theta_prime, n_prime, s_prime,
                    alpha_prime, n_alpha_prime)

    def nuts_step(params, epsilon, iteration, key):
        """Single NUTS iteration."""
        # Sample momentum
        key, *subkeys = mx.random.split(key, len(param_names) + 1)
        r = {
            name: mx.random.normal(params[name].shape, key=subkey)
            for name, subkey in zip(param_names, subkeys)
        }

        # Compute initial Hamiltonian
        H0 = hamiltonian(params, r)

        # Sample slice variable u ~ Uniform(0, exp(-H0))
        # Using u ~ Uniform(0, 1) and log(u) ~ -Exp(1)
        key, subkey = mx.random.split(key)
        uniform_sample = mx.random.uniform(key=subkey)
        log_u = float(-H0) + float(mx.log(uniform_sample))
        u = mx.exp(mx.array(log_u))

        # Initialize tree
        theta_minus = theta_plus = params
        r_minus = r_plus = r
        j = 0  # Tree depth
        n = 1  # Number of valid states
        s = True  # Is tree still valid?

        theta_prime = params  # Candidate
        alpha_sum = 0.0
        n_alpha = 0

        # Build tree until U-turn or max depth
        while s and j < max_tree_depth:
            # Choose direction uniformly
            key, subkey1, subkey2, subkey3 = mx.random.split(key, 4)
            v = 1 if float(mx.random.uniform(key=subkey1)) < 0.5 else -1

            # Build subtree in chosen direction
            if v == -1:
                (theta_minus, _, r_minus, _,
                 theta_double_prime, n_prime, s_prime,
                 alpha_prime, n_alpha_prime) = build_tree(
                     theta_minus, r_minus, u, v, j, epsilon, params, r, subkey2)
            else:
                (_, theta_plus, _, r_plus,
                 theta_double_prime, n_prime, s_prime,
                 alpha_prime, n_alpha_prime) = build_tree(
                     theta_plus, r_plus, u, v, j, epsilon, params, r, subkey2)

            # Accept proposal with probability n'/n (multinomial sampling)
            if s_prime:
                accept_prob = min(1.0, n_prime / max(n, 1.0))
                if float(mx.random.uniform(key=subkey3)) < accept_prob:
                    theta_prime = theta_double_prime

            # Update tree
            n += n_prime
            s = s_prime and no_u_turn(theta_minus, theta_plus, r_minus, r_plus)
            alpha_sum += alpha_prime
            n_alpha += n_alpha_prime

            j += 1

        # Compute average acceptance probability
        accept_prob = alpha_sum / max(n_alpha, 1.0)

        return theta_prime, accept_prob, j, key

    # Warmup phase
    print(f"NUTS warmup: {num_warmup} samples")
    epsilon = step_size

    for m in range(num_warmup):
        current_params, alpha, tree_depth, key = nuts_step(current_params, epsilon, m, key)

        n_accept += int(alpha > 0.5)  # Count as accepted if accept prob > 0.5
        n_total += 1
        total_tree_depth += tree_depth

        # Dual averaging for step size adaptation
        if adapt_step_size:
            eta = 1.0 / (m + t0)
            H_bar = (1 - eta) * H_bar + eta * (target_accept - alpha)
            log_epsilon = mu - (math.sqrt(m + 1) / gamma) * H_bar

            # Clip to prevent explosion or collapse
            log_epsilon = max(min(log_epsilon, 10.0), -10.0)  # exp(-10) to exp(10)
            epsilon = float(mx.exp(mx.array(log_epsilon)))

            m_eta = float(m + 1) ** (-kappa)
            log_epsilon_bar = m_eta * math.log(epsilon) + (1 - m_eta) * math.log(epsilon_bar)
            epsilon_bar = float(mx.exp(mx.array(log_epsilon_bar)))

        if (m + 1) % 250 == 0:
            avg_depth = total_tree_depth / (m + 1)
            print(f"  Iteration {m+1}/{num_warmup} (accept: {100*alpha:.1f}%, depth: {tree_depth}, "
                  f"avg_depth: {avg_depth:.1f}, step_size: {epsilon:.4f})")

    # Use averaged step size for sampling
    if adapt_step_size:
        epsilon = epsilon_bar
        print(f"Warmup complete. Using step_size: {epsilon:.4f}")

    warmup_accept_rate = n_accept / n_total
    avg_warmup_depth = total_tree_depth / num_warmup
    print(f"Warmup statistics: accept_rate: {100*warmup_accept_rate:.2f}%, avg_tree_depth: {avg_warmup_depth:.2f}")

    # Reset counters
    n_accept = 0
    n_total = 0
    total_tree_depth = 0

    # Sampling phase
    print(f"\nNUTS sampling: {num_samples} samples")
    for m in range(num_samples):
        current_params, alpha, tree_depth, key = nuts_step(current_params, epsilon, m + num_warmup, key)

        # Store samples
        for name in param_names:
            all_samples[name].append(float(current_params[name]) if current_params[name].size == 1
                                   else current_params[name].tolist())

        n_accept += int(alpha > 0.5)
        n_total += 1
        total_tree_depth += tree_depth

        if (m + 1) % 500 == 0:
            avg_depth = total_tree_depth / (m + 1)
            print(f"  Iteration {m+1}/{num_samples} (avg_depth: {avg_depth:.2f})")

    # Convert to arrays
    for name in param_names:
        all_samples[name] = mx.array(all_samples[name])

    final_accept_rate = n_accept / n_total
    avg_tree_depth = total_tree_depth / num_samples
    print(f"\nSampling complete!")
    print(f"Final statistics: accept_rate: {100*final_accept_rate:.2f}%, avg_tree_depth: {avg_tree_depth:.2f}")

    return all_samples, final_accept_rate
