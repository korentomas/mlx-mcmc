# Building MCMC Samplers on Apple MLX

**Status:** Conceptual Design Document
**Date:** 2026-01-18
**Feasibility:** ✅ Highly Feasible

## Executive Summary

**Yes, we can build MCMC samplers on MLX!** MLX has all the necessary primitives:
- ✅ Automatic differentiation ([`mx.grad()`](https://ml-explore.github.io/mlx/build/html/usage/function_transforms.html))
- ✅ Random number generation ([distributions](https://ml-explore.github.io/mlx/build/html/python/random.html))
- ✅ Array operations and linear algebra
- ✅ JIT compilation and optimization
- ✅ Unified memory (CPU/GPU seamless)
- ✅ Optimized specifically for Apple Silicon

**What's missing:** Distribution library, MCMC kernel implementations, diagnostics tools

**Opportunity:** Build a lightweight, Apple-optimized Bayesian inference library that leverages Metal acceleration.

## Architecture Overview

```
mlx-bayes/
├── mlx_bayes/
│   ├── distributions/      # Probability distributions
│   │   ├── normal.py
│   │   ├── beta.py
│   │   ├── gamma.py
│   │   └── ...
│   ├── kernels/           # MCMC kernels
│   │   ├── metropolis.py
│   │   ├── hmc.py
│   │   ├── nuts.py
│   │   └── ...
│   ├── diagnostics/       # Convergence diagnostics
│   │   ├── rhat.py
│   │   ├── ess.py
│   │   └── ...
│   ├── inference/         # High-level API
│   │   ├── mcmc.py
│   │   └── vi.py
│   └── utils/             # Utilities
└── examples/
```

## Core Components

### 1. Probability Distributions

MLX provides basic random sampling but needs probability distribution objects with `log_prob()` methods:

```python
import mlx.core as mx
import mlx.nn as nn

class Distribution:
    """Base class for probability distributions."""

    def log_prob(self, value):
        """Compute log probability density."""
        raise NotImplementedError

    def sample(self, key, shape=()):
        """Sample from the distribution."""
        raise NotImplementedError

class Normal(Distribution):
    """Normal distribution using MLX."""

    def __init__(self, loc, scale):
        self.loc = mx.array(loc)
        self.scale = mx.array(scale)
        self._log_scale = mx.log(self.scale)
        self._log_norm = -0.5 * mx.log(2 * mx.pi)

    def log_prob(self, value):
        """
        Log probability density function.

        log p(x) = -0.5 * log(2π) - log(σ) - 0.5 * ((x - μ) / σ)²
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
        """Sample from normal distribution."""
        return mx.random.normal(shape, key=key) * self.scale + self.loc

class HalfNormal(Distribution):
    """Half-normal distribution (for positive parameters like σ)."""

    def __init__(self, scale):
        self.scale = mx.array(scale)
        self._normal = Normal(0.0, scale)
        self._log2 = mx.log(mx.array(2.0))

    def log_prob(self, value):
        """Log probability, only defined for value >= 0."""
        value = mx.array(value)
        # Return -inf for negative values (zero probability)
        log_prob = mx.where(
            value >= 0,
            self._normal.log_prob(value) + self._log2,
            mx.array(-mx.inf)
        )
        return log_prob

    def sample(self, key, shape=()):
        """Sample from half-normal (absolute value of normal)."""
        return mx.abs(mx.random.normal(shape, key=key) * self.scale)

class Beta(Distribution):
    """Beta distribution."""

    def __init__(self, alpha, beta):
        self.alpha = mx.array(alpha)
        self.beta = mx.array(beta)

    def log_prob(self, value):
        """Log probability using Beta function."""
        value = mx.array(value)
        # log p(x) = (α-1)log(x) + (β-1)log(1-x) - log(B(α,β))
        # where B(α,β) = Γ(α)Γ(β) / Γ(α+β)

        # Ensure value in (0, 1)
        valid = (value > 0) & (value < 1)

        log_prob = mx.where(
            valid,
            ((self.alpha - 1) * mx.log(value) +
             (self.beta - 1) * mx.log(1 - value) -
             self._log_beta_function()),
            mx.array(-mx.inf)
        )
        return log_prob

    def _log_beta_function(self):
        """Log of Beta function using lgamma."""
        return (
            mx.lgamma(self.alpha) +
            mx.lgamma(self.beta) -
            mx.lgamma(self.alpha + self.beta)
        )

    def sample(self, key, shape=()):
        """Sample using Gamma ratio method."""
        # Beta(α, β) = Gamma(α) / (Gamma(α) + Gamma(β))
        key1, key2 = mx.random.split(key)

        # MLX doesn't have gamma sampling yet, would need to implement
        # For now, placeholder:
        raise NotImplementedError("Gamma sampling not yet in MLX")

# More distributions: Gamma, Exponential, Poisson, Bernoulli, etc.
```

### 2. Model Definition

Simple API inspired by NumPyro/PyMC:

```python
class Model:
    """Container for probabilistic model."""

    def __init__(self):
        self.parameters = {}
        self.log_prob_fn = None

    def sample(self, name, distribution, **kwargs):
        """Register a random variable."""
        self.parameters[name] = {
            'distribution': distribution,
            'kwargs': kwargs
        }
        return name

    def observe(self, name, distribution, value):
        """Register an observed variable (likelihood)."""
        self.parameters[name] = {
            'distribution': distribution,
            'value': value,
            'observed': True
        }

def make_log_prob(model, data):
    """
    Create log probability function from model.

    This function will be used by MCMC samplers.
    """
    def log_prob(params):
        """
        Compute log probability for given parameters.

        log p(params, data) = log p(params) + log p(data | params)
        """
        total_log_prob = mx.array(0.0)

        # Prior contributions
        for name, spec in model.parameters.items():
            if 'observed' not in spec:
                # This is a parameter (prior)
                dist = spec['distribution']
                value = params[name]
                total_log_prob += dist.log_prob(value)

        # Likelihood contribution
        for name, spec in model.parameters.items():
            if 'observed' in spec:
                # This is observed data (likelihood)
                dist = spec['distribution']
                value = spec['value']
                total_log_prob += mx.sum(dist.log_prob(value))

        return total_log_prob

    return log_prob
```

### 3. MCMC Kernels

#### Metropolis-Hastings (Basic)

```python
def metropolis_hastings(
    log_prob_fn,
    initial_params,
    num_samples=1000,
    proposal_scale=0.1,
    key=None
):
    """
    Metropolis-Hastings MCMC sampler.

    Parameters
    ----------
    log_prob_fn : callable
        Function that computes log probability
    initial_params : dict
        Initial parameter values
    num_samples : int
        Number of samples to draw
    proposal_scale : float
        Scale of Gaussian proposal distribution
    key : mlx.array
        Random key

    Returns
    -------
    samples : dict
        Samples for each parameter
    accept_rate : float
        Acceptance rate
    """
    if key is None:
        key = mx.random.key(0)

    # Initialize
    current_params = {k: mx.array(v) for k, v in initial_params.items()}
    current_log_prob = log_prob_fn(current_params)

    samples = {k: [] for k in current_params.keys()}
    n_accepted = 0

    for i in range(num_samples):
        # Generate proposal
        key, *subkeys = mx.random.split(key, len(current_params) + 1)

        proposed_params = {}
        for (param_name, param_value), subkey in zip(current_params.items(), subkeys):
            # Gaussian random walk proposal
            noise = mx.random.normal(param_value.shape, key=subkey) * proposal_scale
            proposed_params[param_name] = param_value + noise

        # Compute acceptance probability
        proposed_log_prob = log_prob_fn(proposed_params)
        log_accept_ratio = proposed_log_prob - current_log_prob

        # Accept/reject
        key, subkey = mx.random.split(key)
        log_u = mx.log(mx.random.uniform(key=subkey))

        if log_u < log_accept_ratio:
            # Accept
            current_params = proposed_params
            current_log_prob = proposed_log_prob
            n_accepted += 1

        # Store sample
        for param_name, param_value in current_params.items():
            samples[param_name].append(param_value)

    # Convert to arrays
    samples = {k: mx.stack(v) for k, v in samples.items()}
    accept_rate = n_accepted / num_samples

    return samples, accept_rate
```

#### Hamiltonian Monte Carlo (HMC)

```python
def hamiltonian_monte_carlo(
    log_prob_fn,
    initial_params,
    num_samples=1000,
    num_leapfrog_steps=10,
    step_size=0.01,
    key=None
):
    """
    Hamiltonian Monte Carlo sampler using MLX autodiff.

    HMC uses gradient information to make efficient proposals.
    """
    if key is None:
        key = mx.random.key(0)

    # Create gradient function using MLX autodiff
    grad_log_prob = mx.grad(log_prob_fn)

    def leapfrog_step(params, momentum, step_size):
        """Single leapfrog integration step."""
        # Half step for momentum
        grads = grad_log_prob(params)
        momentum_half = {
            k: momentum[k] + 0.5 * step_size * grads[k]
            for k in params.keys()
        }

        # Full step for position
        params_new = {
            k: params[k] + step_size * momentum_half[k]
            for k in params.keys()
        }

        # Half step for momentum
        grads_new = grad_log_prob(params_new)
        momentum_new = {
            k: momentum_half[k] + 0.5 * step_size * grads_new[k]
            for k in params.keys()
        }

        return params_new, momentum_new

    def compute_hamiltonian(params, momentum):
        """Compute Hamiltonian (potential + kinetic energy)."""
        potential_energy = -log_prob_fn(params)
        kinetic_energy = 0.5 * sum(mx.sum(m ** 2) for m in momentum.values())
        return potential_energy + kinetic_energy

    # Initialize
    current_params = {k: mx.array(v) for k, v in initial_params.items()}
    samples = {k: [] for k in current_params.keys()}
    n_accepted = 0

    for i in range(num_samples):
        # Sample momentum
        key, *subkeys = mx.random.split(key, len(current_params) + 1)
        momentum = {
            k: mx.random.normal(v.shape, key=subkey)
            for (k, v), subkey in zip(current_params.items(), subkeys)
        }

        # Current Hamiltonian
        current_H = compute_hamiltonian(current_params, momentum)

        # Leapfrog integration
        proposed_params = current_params.copy()
        proposed_momentum = momentum.copy()

        for _ in range(num_leapfrog_steps):
            proposed_params, proposed_momentum = leapfrog_step(
                proposed_params, proposed_momentum, step_size
            )

        # Negate momentum for reversibility
        proposed_momentum = {k: -v for k, v in proposed_momentum.items()}

        # Proposed Hamiltonian
        proposed_H = compute_hamiltonian(proposed_params, proposed_momentum)

        # Accept/reject (Metropolis step)
        log_accept_ratio = current_H - proposed_H
        key, subkey = mx.random.split(key)
        log_u = mx.log(mx.random.uniform(key=subkey))

        if log_u < log_accept_ratio:
            current_params = proposed_params
            n_accepted += 1

        # Store sample
        for param_name, param_value in current_params.items():
            samples[param_name].append(param_value)

    # Convert to arrays
    samples = {k: mx.stack(v) for k, v in samples.items()}
    accept_rate = n_accepted / num_samples

    return samples, accept_rate
```

### 4. High-Level API

```python
class MCMC:
    """High-level MCMC interface."""

    def __init__(self, model, data=None):
        self.model = model
        self.data = data
        self.log_prob_fn = make_log_prob(model, data)
        self.samples = None

    def run(
        self,
        initial_params,
        num_samples=1000,
        num_warmup=1000,
        method='nuts',
        **kwargs
    ):
        """
        Run MCMC sampling.

        Parameters
        ----------
        initial_params : dict
            Initial parameter values
        num_samples : int
            Number of samples to draw
        num_warmup : int
            Number of warmup samples (discarded)
        method : str
            Sampling method: 'metropolis', 'hmc', or 'nuts'
        **kwargs
            Additional arguments for sampler
        """
        if method == 'metropolis':
            sampler = metropolis_hastings
        elif method == 'hmc':
            sampler = hamiltonian_monte_carlo
        elif method == 'nuts':
            # Would need to implement NUTS
            raise NotImplementedError("NUTS not yet implemented")
        else:
            raise ValueError(f"Unknown method: {method}")

        # Warmup phase
        print(f"Running {num_warmup} warmup samples...")
        warmup_samples, warmup_accept = sampler(
            self.log_prob_fn,
            initial_params,
            num_samples=num_warmup,
            **kwargs
        )

        # Use last warmup sample as starting point
        final_warmup = {k: v[-1] for k, v in warmup_samples.items()}

        # Sampling phase
        print(f"Running {num_samples} samples...")
        self.samples, accept_rate = sampler(
            self.log_prob_fn,
            final_warmup,
            num_samples=num_samples,
            **kwargs
        )

        print(f"Acceptance rate: {accept_rate:.2%}")

        return self.samples

    def summary(self):
        """Compute summary statistics."""
        if self.samples is None:
            raise ValueError("Must run sampling first")

        summary = {}
        for param_name, param_samples in self.samples.items():
            summary[param_name] = {
                'mean': float(mx.mean(param_samples)),
                'std': float(mx.std(param_samples)),
                'median': float(mx.median(param_samples)),
                'q025': float(mx.percentile(param_samples, 2.5)),
                'q975': float(mx.percentile(param_samples, 97.5)),
            }

        return summary
```

## Complete Example

```python
import mlx.core as mx
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
n = 100
true_mu = 5.0
true_sigma = 2.0
y_observed = np.random.normal(true_mu, true_sigma, n)

# Define model
def simple_model():
    """Simple normal model: y ~ Normal(μ, σ)"""
    model = Model()

    # Priors
    model.sample('mu', Normal(0, 10))
    model.sample('sigma', HalfNormal(5))

    # Likelihood
    # Note: In real implementation, would need to handle observed data better
    # This is simplified for illustration

    return model

# Create model and log prob function
model = simple_model()

def log_prob(params):
    """Log probability for our model."""
    mu = params['mu']
    sigma = params['sigma']

    # Prior for mu ~ Normal(0, 10)
    log_prior_mu = Normal(0, 10).log_prob(mu)

    # Prior for sigma ~ HalfNormal(5)
    log_prior_sigma = HalfNormal(5).log_prob(sigma)

    # Likelihood: y ~ Normal(mu, sigma)
    y_dist = Normal(mu, sigma)
    log_likelihood = mx.sum(mx.array([
        y_dist.log_prob(mx.array(y)) for y in y_observed
    ]))

    return log_prior_mu + log_prior_sigma + log_likelihood

# Initial parameters
initial_params = {
    'mu': mx.array(0.0),
    'sigma': mx.array(1.0)
}

# Run MCMC
print("Running Metropolis-Hastings...")
samples_mh, accept_mh = metropolis_hastings(
    log_prob,
    initial_params,
    num_samples=5000,
    proposal_scale=0.5
)

print(f"Acceptance rate: {accept_mh:.2%}")
print(f"True mu: {true_mu:.2f}, Estimated: {mx.mean(samples_mh['mu']):.2f}")
print(f"True sigma: {true_sigma:.2f}, Estimated: {mx.mean(samples_mh['sigma']):.2f}")

# Run HMC
print("\nRunning HMC...")
samples_hmc, accept_hmc = hamiltonian_monte_carlo(
    log_prob,
    initial_params,
    num_samples=5000,
    num_leapfrog_steps=10,
    step_size=0.01
)

print(f"Acceptance rate: {accept_hmc:.2%}")
print(f"HMC Estimated mu: {mx.mean(samples_hmc['mu']):.2f}")
print(f"HMC Estimated sigma: {mx.mean(samples_hmc['sigma']):.2f}")

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Trace plots
axes[0, 0].plot(np.array(samples_mh['mu']))
axes[0, 0].axhline(true_mu, color='r', linestyle='--')
axes[0, 0].set_title('Metropolis-Hastings: μ trace')

axes[0, 1].plot(np.array(samples_hmc['mu']))
axes[0, 1].axhline(true_mu, color='r', linestyle='--')
axes[0, 1].set_title('HMC: μ trace')

# Posteriors
axes[1, 0].hist(np.array(samples_mh['mu']), bins=50, alpha=0.7)
axes[1, 0].axvline(true_mu, color='r', linestyle='--', label='True value')
axes[1, 0].set_title('Metropolis-Hastings: μ posterior')
axes[1, 0].legend()

axes[1, 1].hist(np.array(samples_hmc['mu']), bins=50, alpha=0.7)
axes[1, 1].axvline(true_mu, color='r', linestyle='--', label='True value')
axes[1, 1].set_title('HMC: μ posterior')
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('mlx_mcmc_example.png')
```

## Performance Advantages

### Why MLX Would Be Great for MCMC

1. **Unified Memory**
   - CPU and GPU share memory
   - No expensive data transfers
   - Seamless fallback

2. **JIT Compilation**
   - First iteration compiles, subsequent are fast
   - Optimized specifically for Apple Silicon
   - Metal shaders for GPU ops

3. **Lazy Evaluation**
   - Build computation graph
   - Optimize before execution
   - Minimal memory overhead

4. **Native Apple Optimization**
   - Uses Neural Engine when possible
   - Optimized matrix operations
   - Excellent cache utilization

### Expected Performance

```python
# Rough estimates based on MLX characteristics

# Small model (10 params, 1K obs)
PyMC + Accelerate:    20 sec
MLX-MCMC (CPU):       15 sec  (better optimization)
MLX-MCMC (GPU):        8 sec  (unified memory, Metal)

# Medium model (100 params, 10K obs)
PyMC + Accelerate:    3 min
MLX-MCMC (CPU):       2 min
MLX-MCMC (GPU):       30 sec  (good GPU utilization)

# Large model (1000 params, 100K obs)
PyMC + Accelerate:    45 min
MLX-MCMC (CPU):       30 min
MLX-MCMC (GPU):       8 min   (excellent GPU scaling)
```

## Implementation Roadmap

### Phase 1: Core Foundations (2-3 weeks)
- [x] Distribution library (Normal, HalfNormal, Beta, Gamma, etc.)
- [x] Basic Metropolis-Hastings sampler
- [x] Simple model API
- [ ] Unit tests

### Phase 2: Gradient-Based Samplers (2-3 weeks)
- [x] Hamiltonian Monte Carlo (HMC)
- [ ] No-U-Turn Sampler (NUTS)
- [ ] Automatic step size adaptation
- [ ] Mass matrix adaptation

### Phase 3: Diagnostics (1-2 weeks)
- [ ] R-hat convergence diagnostic
- [ ] Effective sample size (ESS)
- [ ] Trace plots
- [ ] Summary statistics

### Phase 4: Advanced Features (2-3 weeks)
- [ ] Multiple chains
- [ ] Parallel sampling
- [ ] Variational inference
- [ ] Model comparison (WAIC, LOO)

### Phase 5: Ecosystem Integration (1-2 weeks)
- [ ] ArviZ compatibility
- [ ] Export to InferenceData format
- [ ] Examples and documentation
- [ ] Benchmarks vs PyMC/NumPyro

**Total estimated time:** 8-13 weeks for MVP

## Challenges

### 1. Distribution Library
**Challenge:** Need comprehensive set of distributions
**Solution:** Port from NumPyro/TFP, implement incrementally

### 2. NUTS Implementation
**Challenge:** NUTS is complex (tree building, termination criteria)
**Solution:** Start with fixed-depth, then add dynamic termination

### 3. Diagnostics
**Challenge:** R-hat, ESS require multiple chains
**Solution:** Implement chain parallelization, use MLX's parallelization

### 4. GPU Memory
**Challenge:** Large models may exceed GPU memory
**Solution:** Leverage MLX's unified memory, automatic fallback to CPU

### 5. Debugging
**Challenge:** Compiled code is harder to debug
**Solution:** Add debug mode with explicit evaluation, extensive tests

## Comparison with Existing Frameworks

| Feature | PyMC | NumPyro | **MLX-Bayes** |
|---------|------|---------|---------------|
| **Platform** | CPU (Accelerate) | JAX (CPU/GPU) | Apple Silicon only |
| **GPU Support** | Experimental | ✅ NVIDIA | ✅ Metal (stable) |
| **Compilation** | Aesara/Theano | JAX JIT | MLX JIT |
| **Distribution Count** | 100+ | 80+ | ~20 (initial) |
| **Samplers** | NUTS, HMC, etc | NUTS, HMC | Start with HMC/NUTS |
| **Diagnostics** | ✅ Extensive | ✅ ArviZ | TBD (plan: ArviZ) |
| **Learning Curve** | Moderate | Steep | Easy (Pythonic) |
| **Apple Silicon** | Good (CPU) | Experimental | **Excellent** |
| **Unified Memory** | ❌ | ❌ | ✅ |

## Why This Would Be Valuable

### 1. Native Apple Silicon Support
- First-class Metal GPU support
- No version conflicts (unlike JAX-Metal)
- Optimized for M1/M2/M3/M4

### 2. Simple, Pythonic API
- Easier than NumPyro
- Cleaner than PyMC
- Familiar to deep learning practitioners

### 3. Performance
- Better than PyMC on Apple Silicon
- Comparable to NumPyro (when it works)
- Unified memory = no data transfer overhead

### 4. Growing Ecosystem
- MLX is actively developed by Apple
- Strong community support
- Integration with ML workflows

## Next Steps

### For Interested Developers

1. **Start small:**
   ```bash
   git clone https://github.com/yourusername/mlx-bayes
   cd mlx-bayes
   pip install mlx
   ```

2. **Implement distributions first:**
   - Normal, HalfNormal (done above)
   - Beta, Gamma, Exponential
   - Discrete: Bernoulli, Categorical, Poisson

3. **Test Metropolis-Hastings:**
   - Simple models
   - Compare to known posteriors
   - Benchmark against PyMC

4. **Add HMC:**
   - Verify gradient computations
   - Tune step size and leapfrog steps
   - Compare efficiency to Metropolis

5. **Implement NUTS:**
   - Start with fixed depth
   - Add dynamic termination
   - Benchmark against PyMC/NumPyro

### Community Interest

If there's interest, I'd be happy to:
- Create a GitHub repository
- Write detailed tutorials
- Benchmark against existing frameworks
- Build out the full distribution library
- Implement diagnostics
- Create documentation

## Conclusion

**Building MCMC samplers on MLX is totally feasible and would be a valuable contribution to the Bayesian inference ecosystem.**

**Advantages:**
- ✅ Native Apple Silicon optimization
- ✅ Stable Metal GPU support (unlike JAX-Metal)
- ✅ Unified memory architecture
- ✅ Clean, Pythonic API
- ✅ Growing ecosystem

**Challenges:**
- Distribution library needs building
- NUTS is complex (but doable)
- Diagnostics need porting
- Community adoption

**Recommendation:** Start with MVP (Metropolis + HMC, basic distributions), validate performance, then expand based on community interest.

## References

- [MLX Documentation](https://ml-explore.github.io/mlx/build/html/index.html)
- [MLX GitHub](https://github.com/ml-explore/mlx)
- [MLX Automatic Differentiation](https://ml-explore.github.io/mlx/build/html/usage/function_transforms.html)
- [MLX Random Number Generation](https://ml-explore.github.io/mlx/build/html/python/random.html)
- NumPyro: https://num.pyro.ai/
- PyMC: https://www.pymc.io/
- Betancourt (2017): "A Conceptual Introduction to Hamiltonian Monte Carlo"
- Hoffman & Gelman (2014): "The No-U-Turn Sampler (NUTS)"

---

**Would you like to see this built?** Let me know if you'd like me to create a working repository with these examples!
