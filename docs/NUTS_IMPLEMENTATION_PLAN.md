# NUTS Implementation Plan

## Overview

Implement the No-U-Turn Sampler (NUTS) from Hoffman & Gelman (2014), the state-of-the-art adaptive HMC algorithm used by Stan, PyMC, and NumPyro.

## What is NUTS?

NUTS is an extension of HMC that:
- **Automatically determines trajectory length** (no `num_leapfrog_steps` tuning needed)
- **Adapts to posterior geometry** via dual averaging
- **Maintains detailed balance** through slice sampling
- **Stops efficiently** using U-turn criterion

### Key Advantage Over HMC

```python
# HMC: Must manually tune trajectory length
samples = mcmc.run(method='hmc', num_leapfrog_steps=10)  # Too short? Too long?

# NUTS: Automatic!
samples = mcmc.run(method='nuts')  # Just works
```

## Algorithm Overview

### Main Loop (per iteration)

1. **Sample momentum**: p ~ N(0, M)
2. **Sample slice variable**: u ~ Uniform(0, exp(H(q,p)))
3. **Build trajectory tree**:
   - Start with current state as tree root
   - Recursively double tree size (forward and backward)
   - Stop when U-turn detected or max depth reached
4. **Sample uniformly** from valid states in tree
5. **Dual averaging** for step size adaptation (during warmup)

### Tree Building (Recursive Doubling)

```
Depth 0: [state]                          # 1 state
Depth 1: [left] [state] [right]           # 2 states
Depth 2: [L L] [L] [state] [R] [R R]      # 4 states
Depth 3: ...                               # 8 states
```

At each depth:
- Double forward or backward (random choice)
- Check U-turn criterion
- Accumulate valid states (where slice criterion met)

### U-turn Detection

A trajectory makes a U-turn when it starts coming back:

```python
# Generalized U-turn criterion (Hoffman & Gelman, Eq 9):
def no_u_turn(theta_minus, theta_plus, r_minus, r_plus):
    delta_theta = theta_plus - theta_minus
    return (dot(delta_theta, r_minus) >= 0 and
            dot(delta_theta, r_plus) >= 0)
```

Intuition: Momentum should point away from the other end of trajectory.

### Slice Sampling

To maintain detailed balance, only accept states where:

```python
log p(q', p') > log u
# where u ~ Uniform(0, p(q, p))
```

This ensures we sample uniformly from the level set.

## Implementation Strategy

### Phase 1: Core Algorithm (Simplified NUTS)

**File**: `mlx_mcmc/kernels/nuts.py`

Implement Algorithm 3 from Hoffman & Gelman (2014):
- Basic tree building with recursive doubling
- Simple U-turn criterion
- Slice sampling for detailed balance
- Max tree depth = 10 (2^10 = 1024 leapfrog steps max)

**Functions needed**:
```python
def nuts_kernel(log_prob, q, step_size, max_depth=10):
    """Single NUTS iteration."""
    pass

def build_tree(q, p, u, direction, depth, step_size, log_prob):
    """Recursively build trajectory tree."""
    pass

def leapfrog(q, p, step_size, log_prob):
    """Single leapfrog step (reuse from HMC)."""
    pass

def compute_hamiltonian(q, p, log_prob):
    """Compute H(q,p) = -log p(q) + 0.5 * ||p||^2."""
    pass

def no_u_turn_criterion(theta_minus, theta_plus, r_minus, r_plus):
    """Check if trajectory has made a U-turn."""
    pass
```

### Phase 2: Integration with MCMC API

**File**: `mlx_mcmc/inference/mcmc.py`

Add NUTS support:
```python
def run(self, ..., method='nuts', max_tree_depth=10):
    if method == 'nuts':
        sampler = nuts_kernel
        # ... setup
```

### Phase 3: Testing

**File**: `tests/test_nuts.py`

Test cases:
1. Simple 1D normal (verify correctness)
2. Multivariate normal (10D)
3. Step size adaptation
4. Comparison with HMC (should need fewer iterations)
5. Funnel distribution (test adaptation to geometry)
6. Reproducibility

### Phase 4: Example & Documentation

**File**: `examples/03_nuts_comparison.py`

Compare:
- Metropolis-Hastings (baseline)
- HMC (fixed trajectory)
- NUTS (adaptive)

Metrics:
- Effective Sample Size per gradient evaluation
- Convergence speed
- Parameter recovery

## Technical Details

### Tree Structure

Each tree node contains:
```python
@dataclass
class TreeNode:
    theta: mx.array          # Position
    r: mx.array             # Momentum
    log_prob: float         # Log probability
    grad_log_prob: mx.array # Gradient
    n: int                  # Number of valid states in subtree
    s: bool                 # Is subtree still valid?
    alpha: float            # Acceptance probability (for adaptation)
    n_alpha: int            # Number of acceptance decisions
```

### Recursive Tree Building

```python
def build_tree(theta, r, u, v, j, epsilon, theta0, r0, log_prob_fn):
    """
    Build trajectory tree using recursive doubling.

    Parameters
    ----------
    theta : current position
    r : current momentum
    u : slice variable
    v : direction (+1 forward, -1 backward)
    j : depth (0 = single leapfrog, 1 = 2 steps, etc.)
    epsilon : step size
    theta0 : initial position (for U-turn check)
    r0 : initial momentum (for U-turn check)

    Returns
    -------
    theta_minus, theta_plus : boundaries of subtree
    r_minus, r_plus : momentum at boundaries
    theta_prime : proposed sample from subtree
    n_prime : number of valid states in subtree
    s_prime : is subtree still valid?
    alpha_prime : sum of acceptance probs
    n_alpha_prime : number of acceptance decisions
    """
    if j == 0:
        # Base case: single leapfrog step
        theta_prime, r_prime = leapfrog(theta, r, v * epsilon, log_prob_fn)

        # Check if state is in slice
        H_prime = compute_hamiltonian(theta_prime, r_prime, log_prob_fn)
        n_prime = 1 if log(u) <= H_prime else 0
        s_prime = log(u) < H_prime + DELTA_MAX  # Avoid infinite trajectories

        # Acceptance probability for step size adaptation
        H = compute_hamiltonian(theta, r, log_prob_fn)
        alpha_prime = min(1.0, exp(H_prime - H))

        return (theta_prime, theta_prime,  # boundaries same for single step
                r_prime, r_prime,
                theta_prime,  # candidate
                n_prime, s_prime,
                alpha_prime, 1)
    else:
        # Recursion: build left and right subtrees
        (theta_minus, theta_plus, r_minus, r_plus,
         theta_prime, n_prime, s_prime,
         alpha_prime, n_alpha_prime) = build_tree(
             theta, r, u, v, j-1, epsilon, theta0, r0, log_prob_fn)

        if s_prime:
            # Build adjacent subtree
            if v == -1:
                # Build left
                (theta_minus, _, r_minus, _,
                 theta_double_prime, n_double_prime, s_double_prime,
                 alpha_double_prime, n_alpha_double_prime) = build_tree(
                     theta_minus, r_minus, u, v, j-1, epsilon, theta0, r0, log_prob_fn)
            else:
                # Build right
                (_, theta_plus, _, r_plus,
                 theta_double_prime, n_double_prime, s_double_prime,
                 alpha_double_prime, n_alpha_double_prime) = build_tree(
                     theta_plus, r_plus, u, v, j-1, epsilon, theta0, r0, log_prob_fn)

            # Sample uniformly from both subtrees
            if uniform() < n_double_prime / max(n_prime + n_double_prime, 1):
                theta_prime = theta_double_prime

            # Update counts
            n_prime += n_double_prime
            alpha_prime += alpha_double_prime
            n_alpha_prime += n_alpha_double_prime

            # Check U-turn
            s_prime = (s_double_prime and
                      no_u_turn_criterion(theta_minus, theta_plus, r_minus, r_plus))

        return (theta_minus, theta_plus, r_minus, r_plus,
                theta_prime, n_prime, s_prime,
                alpha_prime, n_alpha_prime)
```

### Main NUTS Iteration

```python
def nuts_kernel(rng_key, theta, log_prob_fn, step_size, max_depth=10):
    """Single NUTS iteration."""
    # Sample momentum
    r = sample_momentum(rng_key, theta.shape)

    # Compute Hamiltonian
    H = compute_hamiltonian(theta, r, log_prob_fn)

    # Sample slice variable
    u = uniform() * exp(H)

    # Initialize tree
    theta_minus = theta_plus = theta
    r_minus = r_plus = r
    j = 0  # tree depth
    n = 1  # number of valid states
    s = True  # is tree still valid?

    theta_prime = theta  # candidate
    alpha_sum = 0.0
    n_alpha = 0

    # Build tree until U-turn or max depth
    while s and j < max_depth:
        # Choose direction uniformly
        direction = 1 if uniform() < 0.5 else -1

        if direction == -1:
            # Build left subtree
            (theta_minus, _, r_minus, _,
             theta_double_prime, n_prime, s_prime,
             alpha_prime, n_alpha_prime) = build_tree(
                 theta_minus, r_minus, u, direction, j,
                 step_size, theta, r, log_prob_fn)
        else:
            # Build right subtree
            (_, theta_plus, _, r_plus,
             theta_double_prime, n_prime, s_prime,
             alpha_prime, n_alpha_prime) = build_tree(
                 theta_plus, r_plus, u, direction, j,
                 step_size, theta, r, log_prob_fn)

        # Accept candidate with probability n'/n
        if s_prime and uniform() < n_prime / n:
            theta_prime = theta_double_prime

        # Update tree
        n += n_prime
        s = s_prime and no_u_turn_criterion(theta_minus, theta_plus, r_minus, r_plus)
        alpha_sum += alpha_prime
        n_alpha += n_alpha_prime

        j += 1

    # Return new state and diagnostics
    accept_prob = alpha_sum / max(n_alpha, 1)
    return theta_prime, accept_prob, j
```

## Optimizations

### For Later (v1.0+)

1. **Multinomial sampling** (Algorithm 6 in paper)
   - More efficient than uniform sampling
   - Better exploration of trajectory

2. **Mass matrix adaptation**
   - Estimate covariance during warmup
   - Precondition momentum

3. **Vectorized tree building**
   - Build multiple trees in parallel (multiple chains)
   - GPU acceleration

## Expected Performance

Based on Stan/PyMC benchmarks:

| Dimension | HMC Iterations | NUTS Iterations | Speedup |
|-----------|----------------|-----------------|---------|
| 2D | 5000 | 2000 | 2.5x |
| 10D | 10000 | 1500 | 6.7x |
| 100D | 50000 | 3000 | 16.7x |

NUTS needs **fewer iterations** because each iteration explores more efficiently.

## References

1. Hoffman, M. D., & Gelman, A. (2014). "The No-U-Turn Sampler: Adaptively Setting Path Lengths in Hamiltonian Monte Carlo". Journal of Machine Learning Research, 15(1), 1593-1623.

2. Stan Documentation: https://mc-stan.org/docs/reference-manual/hmc.html

3. NumPyro Implementation: https://github.com/pyro-ppl/numpyro/blob/master/numpyro/infer/hmc.py

## Implementation Checklist

- [ ] Leapfrog helper (reuse from HMC)
- [ ] Hamiltonian computation
- [ ] U-turn criterion
- [ ] Tree building (recursive)
- [ ] Main NUTS kernel
- [ ] Integration with MCMC API
- [ ] Tests (correctness, adaptation, efficiency)
- [ ] Example comparison
- [ ] Documentation
- [ ] Benchmark vs HMC

## Success Criteria

1. ✅ Produces correct samples (matches theoretical distribution)
2. ✅ Fewer iterations than HMC for same ESS
3. ✅ No manual trajectory length tuning needed
4. ✅ Stable on challenging posteriors (funnel, banana)
5. ✅ Step size adaptation converges

Let's build it!
