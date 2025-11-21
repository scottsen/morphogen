# Optimization Examples

This directory contains examples demonstrating Kairo's Optimization Domain.

## Files

### basic_optimization_demo.py

Comprehensive demonstration of all Phase 1 optimization algorithms:

1. **Differential Evolution (DE)** - On Rosenbrock function (5D)
2. **CMA-ES** - On Rosenbrock function (10D)
3. **Particle Swarm Optimization (PSO)** - On Ackley function (5D)
4. **Nelder-Mead Simplex** - On Rosenbrock function (3D)
5. **Unified Interface** - Algorithm comparison
6. **Determinism Verification** - Reproducibility testing
7. **Realistic Application** - PID controller tuning

**Run:**
```bash
python examples/optimization/basic_optimization_demo.py
```

## Quick Start

### Basic Usage

```python
from morphogen.stdlib.optimization import minimize
import numpy as np

# Define objective function
def my_function(x):
    return np.sum(x**2)

# Optimize
result = minimize(
    my_function,
    bounds=[(-5, 5)] * 10,
    method='auto',
    seed=42
)

print(f"Best solution: {result.best_solution}")
print(f"Best fitness:  {result.best_fitness}")
```

### Algorithm-Specific Usage

```python
from morphogen.stdlib.optimization import (
    differential_evolution,
    cmaes,
    particle_swarm,
    nelder_mead
)

# Differential Evolution
result_de = differential_evolution(
    my_function,
    bounds=[(-5, 5)] * 10,
    population_size=50,
    max_iterations=100,
    seed=42
)

# CMA-ES
result_cmaes = cmaes(
    my_function,
    initial_mean=np.zeros(10),
    initial_sigma=1.0,
    max_iterations=100,
    seed=42
)

# PSO
result_pso = particle_swarm(
    my_function,
    bounds=[(-5, 5)] * 10,
    n_particles=30,
    max_iterations=100,
    seed=42
)

# Nelder-Mead
result_nm = nelder_mead(
    my_function,
    initial=np.ones(10),
    max_iterations=500
)
```

## Benchmark Functions

Test your optimizers with standard benchmarks:

```python
from morphogen.stdlib.optimization import BenchmarkFunctions

# Simple functions
sphere = BenchmarkFunctions.sphere        # Unimodal, min at x=0
rosenbrock = BenchmarkFunctions.rosenbrock  # Narrow valley, min at x=[1,...]

# Multimodal functions (many local minima)
rastrigin = BenchmarkFunctions.rastrigin  # Highly multimodal
ackley = BenchmarkFunctions.ackley        # Multimodal with steep region
schwefel = BenchmarkFunctions.schwefel    # Deceptive landscape
```

## Application Examples

### PID Controller Tuning

```python
from morphogen.stdlib.optimization import differential_evolution

def pid_cost(params):
    kp, ki, kd = params
    # Simulate closed-loop response
    response = simulate_pid(kp, ki, kd)
    return response.overshoot + response.settling_time

result = differential_evolution(
    pid_cost,
    bounds=[(0, 10), (0, 5), (0, 2)],
    population_size=30,
    max_iterations=100,
    seed=42
)

print(f"Optimal: Kp={result.best_solution[0]:.2f}, "
      f"Ki={result.best_solution[1]:.2f}, "
      f"Kd={result.best_solution[2]:.2f}")
```

### Parameter Fitting

```python
from morphogen.stdlib.optimization import cmaes
import numpy as np

def fitting_error(params):
    model = simulate_model(params)
    measured = load_data()
    return np.linalg.norm(model - measured)

result = cmaes(
    fitting_error,
    initial_mean=np.array([1.0, 2.0, 3.0]),
    initial_sigma=0.5,
    max_iterations=200,
    seed=42
)
```

### Geometric Optimization

```python
from morphogen.stdlib.optimization import particle_swarm

def geometry_cost(params):
    length, width, height = params
    volume = length * width * height
    surface_area = 2 * (length*width + width*height + height*length)

    # Minimize surface area for given volume
    target_volume = 1000.0
    return surface_area + 100 * abs(volume - target_volume)

result = particle_swarm(
    geometry_cost,
    bounds=[(1, 20), (1, 20), (1, 20)],
    n_particles=25,
    max_iterations=100,
    seed=42
)
```

## Algorithm Selection Guide

| Problem Type | Recommended Algorithm | Rationale |
|--------------|----------------------|-----------|
| Low-dim continuous (≤5D) | Nelder-Mead | Fast, no gradient needed |
| Mid-dim continuous (6-20D) | Differential Evolution | Best general-purpose |
| High-dim continuous (>20D) | CMA-ES | Handles curse of dimensionality |
| Smooth landscape | PSO or Nelder-Mead | Exploits smoothness |
| Multimodal | DE or PSO | Good exploration |
| Noisy objectives | CMA-ES or DE | Robust to noise |
| Expensive functions | (Phase 2: Bayesian Opt) | Sample-efficient |

## Performance Tips

1. **Start small:** Begin with small population/iterations to test
2. **Use deterministic seeds:** For debugging and reproducibility
3. **Monitor convergence:** Check `fitness_history` for improvement
4. **Tune parameters:** Adjust population size, mutation rates, etc.
5. **Try multiple algorithms:** Different problems favor different methods
6. **Use callbacks:** Monitor progress during long runs

## Example Callback

```python
def progress_callback(iteration, best_solution, best_fitness):
    if iteration % 10 == 0:
        print(f"Iteration {iteration}: fitness = {best_fitness:.6e}")

result = differential_evolution(
    my_function,
    bounds,
    callback=progress_callback,
    seed=42
)
```

## Expected Output

From `basic_optimization_demo.py`:

```
======================================================================
  Kairo Optimization Domain - Phase 1 Demonstration
======================================================================

1. Differential Evolution (DE)
   - Rosenbrock 5D: converges to ~0.01
   - ~2,500 evaluations

2. CMA-ES
   - Rosenbrock 10D: converges to ~1.0
   - ~3,000 evaluations
   - Shows covariance adaptation

3. PSO
   - Ackley 5D: converges to ~1e-5
   - ~3,000 evaluations
   - Shows swarm dynamics

4. Nelder-Mead
   - Rosenbrock 3D: converges to <1e-10
   - ~400 evaluations
   - Very precise on smooth problems

5. Unified Interface
   - Auto-selects DE for 5D problem
   - Compares all algorithms

6. Determinism
   - Bit-exact reproduction with same seed

7. PID Tuning
   - Finds optimal gains: Kp=2.0, Ki=1.0, Kd=0.5
```

## Resources

- **Module:** `morphogen/stdlib/optimization.py`
- **Tests:** `tests/test_optimization_operations.py`
- **Docs:** `docs/optimization-domain-implementation.md`
- **Planning:** `docs/reference/optimization-algorithms.md`

## Contributing

To add new examples:

1. Follow existing code style
2. Include docstrings
3. Add to this README
4. Test with `python examples/optimization/your_example.py`

## Phase 2 Preview

Coming in Phase 2:

- **Multi-objective optimization** (NSGA-II)
- **Bayesian Optimization** (for expensive simulations)
- **Gradient-based methods** (L-BFGS)
- **Constraint handling**
- **Parallel evaluation**
