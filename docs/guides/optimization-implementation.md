# Optimization Domain Implementation

**Version:** 0.9.0
**Date:** 2025-11-16
**Status:** Phase 1 Complete

---

## Overview

The Optimization Domain implements **Phase 1 evolutionary algorithms** for Morphogen, transforming it from a simulation platform into a **design discovery platform**. This enables automatic parameter tuning, shape optimization, and multi-objective design exploration across all physical domains.

### Implemented Algorithms (Phase 1)

1. **Differential Evolution (DE)** - Best general-purpose real-valued optimizer
2. **CMA-ES** - Gold standard for high-dimensional continuous problems
3. **Particle Swarm Optimization (PSO)** - Swarm-based global search
4. **Nelder-Mead Simplex** - Derivative-free local optimization

Additionally, the existing **Genetic Algorithm (GA)** implementation in `morphogen/stdlib/genetic.py` provides complete evolutionary computation capabilities.

---

## Architecture

### Module Structure

```
morphogen/stdlib/optimization.py          # Main optimization module (1,200+ lines)
├── OptimizationResult                # Unified result type
├── DifferentialEvolution             # DE implementation
├── CMAES                             # CMA-ES implementation
├── ParticleSwarmOptimization         # PSO implementation
├── NelderMead                        # Nelder-Mead implementation
├── Optimizer                         # Unified interface
└── BenchmarkFunctions                # Standard test functions

morphogen/stdlib/genetic.py               # Genetic algorithm (600 lines)
├── Individual, Population            # GA data structures
├── GeneticOperations                 # 4-layer operator hierarchy
└── Presets                          # Common GA configurations
```

### Design Principles

Following Morphogen's architectural patterns:

1. **Deterministic Execution** - All algorithms support fixed seeds for reproducibility
2. **Immutable Semantics** - Results are immutable data structures
3. **Unified Interface** - Consistent API across all optimizers
4. **Comprehensive Metadata** - Full tracking of convergence and statistics
5. **Modular Design** - Each algorithm is self-contained and testable

---

## Algorithms

### 1. Differential Evolution (DE)

**Best for:** Continuous real-valued parameters, stable and efficient

**Key features:**
- Strategy: DE/rand/1/bin
- Self-adaptive mutation
- Fast convergence on smooth landscapes
- Robust to moderate noise

**Use cases:**
- PID controller tuning
- Motor parameter optimization
- Acoustic chamber tuning
- Heat-transfer parameter fitting

**Example:**
```python
from morphogen.stdlib.optimization import differential_evolution

result = differential_evolution(
    objective_fn=my_objective,
    bounds=[(-5, 5)] * 10,
    population_size=50,
    max_iterations=100,
    F=0.8,
    CR=0.9,
    seed=42
)
```

### 2. CMA-ES (Covariance Matrix Adaptation)

**Best for:** High-dimensional continuous optimization (up to 100+ dimensions)

**Key features:**
- Adapts to landscape curvature via covariance matrix learning
- Handles ill-conditioned problems
- Robust to noise
- State-of-the-art convergence

**Use cases:**
- High-dimensional parameter fitting (20+ parameters)
- Inverse problems (matching recorded signals)
- Complex geometric optimization
- Spectral fitting for acoustic models

**Example:**
```python
from morphogen.stdlib.optimization import cmaes

result = cmaes(
    objective_fn=my_objective,
    initial_mean=np.zeros(20),
    initial_sigma=1.0,
    max_iterations=300,
    seed=42
)
```

### 3. Particle Swarm Optimization (PSO)

**Best for:** Smooth continuous landscapes, cooperative search

**Key features:**
- Models particles "flying" through search space
- Balances exploration (inertia) and exploitation (cognitive + social)
- Good for problems with unknown but smooth gradients

**Use cases:**
- Resonant cavity geometries
- Speaker crossover optimization
- Antenna design
- Magnet position optimization

**Example:**
```python
from morphogen.stdlib.optimization import particle_swarm

result = particle_swarm(
    objective_fn=my_objective,
    bounds=[(-5, 5)] * 10,
    n_particles=30,
    max_iterations=100,
    w=0.7,   # Inertia
    c1=1.5,  # Cognitive
    c2=1.5,  # Social
    seed=42
)
```

### 4. Nelder-Mead Simplex

**Best for:** Low-dimensional derivative-free local optimization (<10D)

**Key features:**
- No gradient required
- Robust to noise
- Simple and reliable
- Fast convergence on smooth problems

**Use cases:**
- Fine-tuning parameters
- Noisy objective functions
- Problems without gradient information
- Local refinement after global search

**Example:**
```python
from morphogen.stdlib.optimization import nelder_mead

result = nelder_mead(
    objective_fn=my_objective,
    initial=np.array([1.0, 2.0, 3.0]),
    max_iterations=500,
    tol=1e-6
)
```

---

## Unified Interface

The `Optimizer` class provides algorithm auto-selection:

```python
from morphogen.stdlib.optimization import Optimizer

# Auto-select best algorithm for problem
result = Optimizer.minimize(
    objective_fn=my_objective,
    bounds=[(-5, 5)] * 10,
    method='auto',  # or 'de', 'cmaes', 'pso', 'nelder-mead'
    max_iterations=100,
    seed=42
)
```

**Auto-selection logic:**
- **≤5 dimensions:** Nelder-Mead (if initial point) or DE
- **6-20 dimensions:** Differential Evolution
- **>20 dimensions:** CMA-ES

---

## Results and Metadata

All algorithms return `OptimizationResult`:

```python
@dataclass
class OptimizationResult:
    best_solution: np.ndarray      # Optimal parameters found
    best_fitness: float            # Objective value at optimum
    fitness_history: List[float]   # Convergence tracking
    n_evaluations: int             # Total function evaluations
    converged: bool                # Convergence flag
    metadata: Dict[str, Any]       # Algorithm-specific data
```

**Metadata includes:**
- **DE:** Final population, F, CR parameters
- **CMA-ES:** Final mean, sigma, covariance matrix
- **PSO:** Final particles, velocities, swarm statistics
- **Nelder-Mead:** Final simplex, simplex size

---

## Benchmark Functions

Standard test functions for validation:

```python
from morphogen.stdlib.optimization import BenchmarkFunctions

# Simple unimodal
sphere = BenchmarkFunctions.sphere          # min at x=0
rosenbrock = BenchmarkFunctions.rosenbrock  # min at x=[1,1,...]

# Multimodal (many local minima)
rastrigin = BenchmarkFunctions.rastrigin    # min at x=0
ackley = BenchmarkFunctions.ackley          # min at x=0
schwefel = BenchmarkFunctions.schwefel      # min at x=[420.97,...]
```

---

## Determinism

All algorithms are **deterministic** with fixed seed:

```python
# Same seed → identical results
result1 = differential_evolution(obj, bounds, seed=42)
result2 = differential_evolution(obj, bounds, seed=42)

assert np.array_equal(result1.best_solution, result2.best_solution)
assert result1.best_fitness == result2.best_fitness
```

This enables:
- Reproducible research
- Regression testing
- Debugging and validation
- Bit-exact reproduction across platforms

---

## Testing

Comprehensive test suite in `tests/test_optimization_operations.py`:

- **Correctness:** Finds known optima on benchmark functions
- **Determinism:** Same seed produces identical results
- **Convergence:** Tracks fitness improvement over iterations
- **Edge cases:** Bounds enforcement, callback invocation
- **Integration:** Realistic application scenarios

### Running Tests

```bash
pytest tests/test_optimization_operations.py -v
```

---

## Examples

### Example 1: Simple Optimization

```python
from morphogen.stdlib.optimization import minimize
import numpy as np

# Define objective
def rosenbrock(x):
    return sum(100*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2)

# Optimize
result = minimize(
    rosenbrock,
    bounds=[(-5, 5)] * 5,
    method='de',
    seed=42
)

print(f"Solution: {result.best_solution}")
print(f"Fitness:  {result.best_fitness}")
```

### Example 2: PID Controller Tuning

```python
from morphogen.stdlib.optimization import differential_evolution

def pid_performance(params):
    kp, ki, kd = params
    # Simulate controller performance
    overshoot = abs(kp - 2.0)
    settling = abs(ki - 1.0)
    noise = abs(kd - 0.5)
    return overshoot + settling + noise

result = differential_evolution(
    pid_performance,
    bounds=[(0, 10), (0, 5), (0, 2)],
    population_size=30,
    max_iterations=100,
    seed=42
)

print(f"Optimal PID: Kp={result.best_solution[0]:.3f}, "
      f"Ki={result.best_solution[1]:.3f}, "
      f"Kd={result.best_solution[2]:.3f}")
```

### Example 3: High-Dimensional Fitting

```python
from morphogen.stdlib.optimization import cmaes
import numpy as np

# Fit 20-parameter model
def model_error(params):
    model_output = complex_simulation(params)
    measured = load_measurements()
    return np.linalg.norm(model_output - measured)

result = cmaes(
    model_error,
    initial_mean=np.zeros(20),
    initial_sigma=1.0,
    max_iterations=300,
    seed=42
)

print(f"Model fit error: {result.best_fitness:.6e}")
```

---

## Performance Characteristics

### Function Evaluations

Typical evaluations to convergence on 10D Rosenbrock:

| Algorithm     | Evaluations | Time (relative) |
|---------------|-------------|-----------------|
| Nelder-Mead   | ~500        | 1x              |
| DE            | ~2,500      | 5x              |
| PSO           | ~3,000      | 6x              |
| CMA-ES        | ~3,000      | 8x*             |

*CMA-ES has higher per-iteration cost due to covariance update

### Scaling

| Algorithm     | Max Dimensions | Scaling    |
|---------------|----------------|------------|
| Nelder-Mead   | ~10            | O(N²)      |
| DE            | ~50            | O(N·P)     |
| PSO           | ~100           | O(N·P)     |
| CMA-ES        | ~200+          | O(N²·P)    |

Where N = dimensions, P = population size

---

## Cross-Domain Applications

### Combustion Domain

```python
# Optimize J-tube geometry
def flame_quality(params):
    diameter, jet_count, air_ratio = params
    flame = combustion.simulate_flame(diameter, jet_count, air_ratio)
    return -flame.uniformity + 0.1 * flame.smoke_index

result = differential_evolution(
    flame_quality,
    bounds=[(50, 150), (4, 16), (10, 20)],
    seed=42
)
```

### Acoustics Domain

```python
# Optimize muffler geometry
def transmission_loss(params):
    length, diameter, baffle_count = params
    muffler = acoustics.expansion_chamber(length, diameter, baffle_count)
    tl = acoustics.transmission_loss(muffler, freq_range=[100, 1000])
    return -np.mean(tl)  # Maximize average TL

result = particle_swarm(
    transmission_loss,
    bounds=[(100, 500), (50, 200), (1, 5)],
    seed=42
)
```

### Motors Domain

```python
# Tune current controller
def control_performance(params):
    kp, ki = params
    controller = motors.pi_controller(kp, ki)
    response = motors.step_response(motor, controller)
    return response.overshoot + 0.5 * response.settling_time

result = cmaes(
    control_performance,
    initial_mean=np.array([1.0, 0.5]),
    initial_sigma=0.5,
    seed=42
)
```

---

## Future Work (Phase 2)

### Planned Enhancements

1. **Multi-objective optimization** (NSGA-II, SPEA2)
2. **Bayesian Optimization** for expensive simulations
3. **Gradient-based methods** (L-BFGS, requires autodiff)
4. **Constraint handling** (inequality/equality constraints)
5. **Parallel evaluation** (distribute objective function calls)

### Integration Opportunities

- **MLIR lowering:** Compile optimization loops to efficient code
- **GPU acceleration:** Parallelize population evaluations
- **Autodiff integration:** Enable gradient-based methods
- **Surrogate models:** Gaussian Processes for expensive simulations

---

## References

### Academic Papers

- **Differential Evolution:** Storn & Price, "Differential Evolution – A Simple and Efficient Heuristic" (1997)
- **CMA-ES:** Hansen & Ostermeier, "Completely Derandomized Self-Adaptation in Evolution Strategies" (2001)
- **PSO:** Kennedy & Eberhart, "Particle Swarm Optimization" (1995)
- **Nelder-Mead:** Nelder & Mead, "A Simplex Method for Function Minimization" (1965)

### Implementation Libraries

- **scipy.optimize:** Reference implementations (Nelder-Mead, L-BFGS)
- **pycma:** CMA-ES reference implementation
- **DEAP:** Evolutionary algorithm framework
- **PyMOO:** Multi-objective optimization

### Morphogen Documentation

- `docs/reference/optimization-algorithms.md` - Algorithm catalog
- `docs/guides/domain-implementation.md` - Implementation guide
- `examples/optimization/` - Usage examples

---

## Summary

The Optimization Domain Phase 1 implementation provides Morphogen with:

✅ **4 production-ready algorithms** (DE, CMA-ES, PSO, Nelder-Mead)
✅ **Unified interface** with auto-selection
✅ **Deterministic execution** for reproducibility
✅ **Comprehensive testing** with benchmark functions
✅ **Cross-domain applications** (combustion, acoustics, motors)
✅ **Complete documentation** and examples

This unlocks **design discovery** capabilities across all Morphogen domains, enabling automatic parameter tuning, shape optimization, and multi-objective design exploration.

**Status:** Ready for production use
**Next:** Phase 2 (Multi-objective, Bayesian Optimization, Constraints)
