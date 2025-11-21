# Morphogen Optimization Domain: Complete Algorithm Catalog

**Version:** 1.0
**Date:** 2025-11-15
**Status:** Planning Document
**Related:** ../architecture/domain-architecture.md, ADR-002

---

## Overview

This document catalogs the **complete set of optimization algorithms** that Morphogen's Optimization Domain should eventually support. These algorithms transform Morphogen from a simulation platform into a **design discovery platform**, enabling automatic parameter tuning, shape optimization, and multi-objective design exploration across all physical domains.

### Why Optimization Matters for Morphogen

Different optimization problems require different solvers based on:
- **Continuity** — Smooth vs. discontinuous objective functions
- **Smoothness** — Gradient availability and reliability
- **Dimensionality** — Number of parameters (1D, 10D, 100D+)
- **Noise** — Simulation noise, numerical noise
- **Computational Cost** — Fast analytical vs. expensive CFD/FEM
- **Multi-Objective** — Single goal vs. competing tradeoffs

Morphogen's physical domains (combustion, acoustics, circuits, motors, geometry, etc.) span all these problem types, so the Optimization Domain must be **comprehensive** and **modular**.

---

## Document Structure

Algorithms are grouped into five categories:

1. **Evolutionary / Population-Based** — Global search for messy, nonlinear, noisy problems
2. **Local Numerical Optimization** — Fast gradient-based methods for smooth problems
3. **Surrogate / Model-Based** — Intelligent sampling for expensive simulations
4. **Combinatorial / Discrete** — Optimization over discrete parameter spaces
5. **Multi-Objective** — Pareto-optimal tradeoff exploration

For each algorithm, we specify:
- **Best For** — Problem characteristics where it excels
- **Use Cases** — Concrete Morphogen domain applications
- **Operator Signature** — How it integrates into the operator registry
- **Dependencies** — Required Morphogen subsystems
- **Implementation Priority** — Phase 1, Phase 2, or Future

---

## Algorithm Categories

---

## 1. Evolutionary / Population-Based (Global Search)

**Purpose**: Broad exploration of rough, nonlinear, noisy, or discontinuous landscapes.

**Why Essential**: Common in Morphogen's physical domains (combustion, acoustics, circuits, motors) where objective functions are:
- Non-differentiable (geometric constraints, discrete choices)
- Noisy (stochastic simulations, CFD turbulence)
- Multi-modal (many local optima)
- Black-box (no gradient information)

**Strategy**: Maintain a population of candidate solutions, evolve over generations.

---

### 1.1 Genetic Algorithm (GA)

**Best For**: Broad search of rough landscapes, mixed continuous/discrete parameters.

**Use Cases**:
- LC filter optimization (discrete component values + continuous parameters)
- J-tube geometry (flame shape, jet positions)
- Muffler shapes (baffle counts, chamber dimensions)
- Speaker EQ / crossover optimization (filter parameters)
- Noisy CFD-like problems (exhaust flow, combustion)

**Operator Signature**:
```morphogen
opt.ga<T>(
    genome: GenomeSpec<T>,
    fitness: (T) -> f64,
    population_size: int = 100,
    generations: int = 100,
    mutation_rate: f64 = 0.01,
    crossover_rate: f64 = 0.7,
    selection: "tournament" | "roulette" | "rank" = "tournament",
    elitism: int = 2,
    seed: int
) -> OptResult<T>
```

**Genome Specification**:
```morphogen
# Example: LC filter optimization
let genome = GenomeSpec({
    L: RangeParam(1.0 μH, 100.0 μH),
    C: RangeParam(1.0 μF, 100.0 μF),
    topology: DiscreteParam(["T", "Pi", "L"])
})
```

**Output**:
```morphogen
struct OptResult<T> {
    best: T,
    best_fitness: f64,
    population: Array<T>,
    fitness_history: Array<f64>,
    convergence_plot: Plot
}
```

**Dependencies**: Stochastic (for mutation/crossover), Serialization (for genome encoding)

**Implementation Priority**: **Phase 1** (baseline evolutionary optimizer)

**Determinism**: DETERMINISTIC (with fixed seed)

---

### 1.2 Differential Evolution (DE)

**Best For**: Continuous real-valued parameters, more efficient and stable than GA.

**Use Cases**:
- PID controller tuning (proportional, integral, derivative gains)
- Motor torque ripple minimization (magnet shapes, winding patterns)
- Heat-transfer parameter fitting (thermal conductivity, convection coefficients)
- Acoustic chamber tuning (Helmholtz resonator dimensions)
- Combustion parameter optimization (fuel/air ratio, injection timing)

**Why Superior to GA**:
- Faster convergence on continuous problems
- Self-adaptive mutation
- More reliable for real-parameter optimization

**Operator Signature**:
```morphogen
opt.de<T>(
    bounds: Array<(f64, f64)>,  # Min/max for each parameter
    fitness: (Array<f64>) -> f64,
    population_size: int = 50,
    generations: int = 100,
    F: f64 = 0.8,              # Mutation factor
    CR: f64 = 0.9,             # Crossover probability
    strategy: "rand/1" | "best/1" | "current-to-best/1" = "rand/1",
    seed: int
) -> OptResult<Array<f64>>
```

**Example**:
```morphogen
# Optimize PID controller
let bounds = [(0.0, 10.0), (0.0, 5.0), (0.0, 1.0)]  # Kp, Ki, Kd
let result = opt.de(
    bounds,
    fitness = |params| -> f64 {
        let [Kp, Ki, Kd] = params
        let controller = control.pid(Kp, Ki, Kd)
        let response = simulate_step_response(controller)
        return -response.overshoot  # Minimize overshoot
    },
    population_size = 30,
    generations = 50
)
```

**Dependencies**: Stochastic (for mutation), Linear Algebra (for vector operations)

**Implementation Priority**: **Phase 1** (best general-purpose real-valued optimizer)

**Determinism**: DETERMINISTIC (with fixed seed)

---

### 1.3 CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

**Best For**: High-dimensional continuous optimization, noisy gradients, ill-conditioned problems.

**Use Cases**:
- Matching recorded signals (spectral fitting, inverse problems)
- Tuning dozens of acoustic filter parameters (parametric EQ, room correction)
- Optimizing 2-stroke expansion chamber (complex geometry, many dimensions)
- Inverse modeling of flame shape (10+ geometric parameters)
- Identifying mechanical system parameters (multi-body dynamics, friction models)

**Why Gold Standard**:
- Handles up to 100+ dimensions
- Adapts to landscape curvature (covariance matrix learning)
- Robust to noise
- State-of-the-art for black-box optimization

**Operator Signature**:
```morphogen
opt.cmaes<T>(
    initial_mean: Array<f64>,
    initial_sigma: f64,
    bounds: Array<(f64, f64)>,
    fitness: (Array<f64>) -> f64,
    population_size: int = auto,  # 4 + floor(3*log(N))
    max_generations: int = 1000,
    tol_fun: f64 = 1e-12,
    seed: int
) -> OptResult<Array<f64>> {
    covariance_matrix: Matrix<f64>,
    evolution_path: Array<Array<f64>>
}
```

**Example**:
```morphogen
# Fit 20-parameter acoustic model to measured spectrum
let result = opt.cmaes(
    initial_mean = [0.0; 20],
    initial_sigma = 1.0,
    bounds = [(−5.0, 5.0); 20],
    fitness = |params| {
        let model = acoustics.parametric_chamber(params)
        let spectrum = acoustics.frequency_response(model, freq_range)
        let measured = io.load("measured_spectrum.wav")
        return -spectral_distance(spectrum, measured)
    }
)
```

**Dependencies**: Linear Algebra (for covariance matrix), Stochastic

**Implementation Priority**: **Phase 1** (world-class for hard continuous problems)

**Determinism**: DETERMINISTIC (with fixed seed)

---

### 1.4 Particle Swarm Optimization (PSO)

**Best For**: Smooth-ish continuous landscapes, cooperative search behavior.

**Use Cases**:
- Finding resonant cavity geometries (acoustics, electromagnetics)
- Motor controller tuning (current control loops)
- Optimizing magnet positions (motor design)
- Antenna design (electromagnetics domain)
- Speaker crossover network optimization

**Why Different from GA/DE**:
- Models particles "flying" through search space
- Influenced by personal best + global best
- Good for problems with smooth gradients but unknown analytically

**Operator Signature**:
```morphogen
opt.pso<T>(
    bounds: Array<(f64, f64)>,
    fitness: (Array<f64>) -> f64,
    n_particles: int = 30,
    max_iterations: int = 100,
    w: f64 = 0.7,          # Inertia weight
    c1: f64 = 1.5,         # Cognitive coefficient
    c2: f64 = 1.5,         # Social coefficient
    seed: int
) -> OptResult<Array<f64>> {
    particle_positions: Array<Array<f64>>,
    particle_velocities: Array<Array<f64>>,
    personal_bests: Array<Array<f64>>,
    global_best: Array<f64>
}
```

**Example**:
```morphogen
# Optimize Helmholtz resonator dimensions
let result = opt.pso(
    bounds = [(10mm, 100mm), (5mm, 50mm), (20mm, 200mm)],  # neck_dia, neck_len, volume
    fitness = |params| {
        let [neck_dia, neck_len, volume] = params
        let resonator = acoustics.helmholtz(neck_dia, neck_len, volume)
        let f_res = acoustics.resonance_frequency(resonator)
        return -abs(f_res - 440Hz)  # Target A4 note
    },
    n_particles = 20,
    max_iterations = 50
)
```

**Dependencies**: Stochastic (for initialization), Linear Algebra

**Implementation Priority**: **Phase 2** (excellent for smooth landscapes)

**Determinism**: DETERMINISTIC (with fixed seed)

---

## 2. Local Numerical Optimization (Smooth Problems)

**Purpose**: Fast optimization when gradients exist (or can be approximated).

**Why Needed**: When the simulation graph is differentiable (or nearly so), gradient-based methods are **orders of magnitude faster** than evolutionary methods.

**Strategy**: Iteratively move in the direction of steepest descent (or approximation thereof).

---

### 2.1 Gradient Descent

**Best For**: Smooth, differentiable objectives with reliable gradients.

**Use Cases**:
- Tuning LTI filter coefficients (linear time-invariant systems)
- Control stability optimization (gradient of Lyapunov function)
- Fitting mathematical models to measured data (least-squares)
- Neural network training (if Morphogen adds autodiff)

**Operator Signature**:
```morphogen
opt.gradient_descent<T>(
    initial: Array<f64>,
    gradient: (Array<f64>) -> Array<f64>,  # Or auto-computed via autodiff
    learning_rate: f64 = 0.01,
    max_iterations: int = 1000,
    tol: f64 = 1e-6,
    momentum: f64 = 0.0
) -> OptResult<Array<f64>>
```

**Example (with Autodiff)**:
```morphogen
# Fit filter to target frequency response
let target_response = [...]
let result = opt.gradient_descent(
    initial = [1.0, 0.5, 0.1],
    gradient = autodiff.grad(|params| {
        let filter = audio.biquad(params[0], params[1], params[2])
        let response = audio.frequency_response(filter)
        return l2_distance(response, target_response)
    }),
    learning_rate = 0.01,
    max_iterations = 500
)
```

**Dependencies**: Autodiff (for automatic gradients), Linear Algebra

**Implementation Priority**: **Phase 2** (requires autodiff infrastructure)

**Determinism**: DETERMINISTIC

---

### 2.2 Quasi-Newton (BFGS / L-BFGS)

**Best For**: Moderate dimensions, smooth gradients, faster convergence than gradient descent.

**Use Cases**:
- Thermodynamic equilibrium solving (chemical reactions, phase transitions)
- Estimating heat-transfer coefficients (inverse heat conduction)
- Fitting experimental curves (material properties, stress-strain)
- Matching impedance curves (speaker impedance models)

**Why Better than Gradient Descent**:
- Approximates Hessian (second-order information)
- Superlinear convergence near optimum
- L-BFGS variant handles high dimensions efficiently

**Operator Signature**:
```morphogen
opt.lbfgs<T>(
    initial: Array<f64>,
    objective: (Array<f64>) -> f64,
    gradient: (Array<f64>) -> Array<f64>,
    m: int = 10,           # Number of correction pairs (L-BFGS)
    max_iterations: int = 1000,
    tol: f64 = 1e-6,
    line_search: "backtracking" | "wolfe" = "wolfe"
) -> OptResult<Array<f64>> {
    hessian_approx: Matrix<f64>
}
```

**Example**:
```morphogen
# Estimate thermal conductivity from temperature measurements
let result = opt.lbfgs(
    initial = [10.0],  # Initial guess for k (W/m·K)
    objective = |params| {
        let k = params[0]
        let sim_temps = simulate_heat_transfer(k, boundary_conditions)
        let measured_temps = io.load("temperature_data.csv")
        return l2_distance(sim_temps, measured_temps)
    },
    gradient = autodiff.grad(objective),
    m = 5,
    max_iterations = 100
)
```

**Dependencies**: Autodiff, Linear Algebra

**Implementation Priority**: **Phase 2** (excellent for smooth problems)

**Determinism**: DETERMINISTIC

---

### 2.3 Nelder-Mead (Simplex)

**Best For**: Derivative-free local optimization, unreliable gradients.

**Use Cases**:
- Muffler frequency matching (noisy simulations)
- Flame index minimization (discontinuous due to ignition thresholds)
- Matching impedance or transfer functions (measurement noise)
- Tuning control loops without gradient information

**Why Useful**:
- No gradient required (geometry-based search)
- Robust to noise
- Simple and reliable for low-dimensional problems (<10 dimensions)

**Operator Signature**:
```morphogen
opt.nelder_mead<T>(
    initial: Array<f64>,
    objective: (Array<f64>) -> f64,
    max_iterations: int = 1000,
    tol: f64 = 1e-6,
    alpha: f64 = 1.0,      # Reflection
    gamma: f64 = 2.0,      # Expansion
    rho: f64 = 0.5,        # Contraction
    sigma: f64 = 0.5       # Shrinkage
) -> OptResult<Array<f64>> {
    simplex_history: Array<Array<Array<f64>>>
}
```

**Example**:
```morphogen
# Optimize muffler geometry to match target frequency response
let result = opt.nelder_mead(
    initial = [50mm, 100mm, 30mm],  # chamber_length, chamber_diameter, baffle_thickness
    objective = |params| {
        let [L, D, t] = params
        let muffler = acoustics.expansion_chamber(L, D, t)
        let transmission_loss = acoustics.transmission_loss(muffler, freq_range)
        let target = [20dB @ 100Hz, 30dB @ 500Hz, 25dB @ 1000Hz]
        return -correlation(transmission_loss, target)
    },
    max_iterations = 200
)
```

**Dependencies**: None (pure geometry-based search)

**Implementation Priority**: **Phase 1** (simple, reliable, derivative-free)

**Determinism**: DETERMINISTIC

---

## 3. Surrogate / Model-Based Optimization

**Purpose**: Optimize **expensive** simulations by building a lightweight model (surrogate) and exploring intelligently.

**Why Critical for Morphogen**:
- CFD simulations (combustion, fluid flow) may take minutes per evaluation
- FEM simulations (structural, thermal) can be very slow
- Complex multi-domain simulations (motor + thermal + control) are expensive

**Strategy**:
1. Sample objective function at a few points
2. Build surrogate model (Gaussian Process, polynomial, RBF)
3. Use acquisition function to choose next sample point
4. Iterate until convergence

---

### 3.1 Bayesian Optimization (BO)

**Best For**: Expensive black-box functions, limited budget (10-100 evaluations).

**Use Cases**:
- Expensive CFD simulations (combustion chamber geometry)
- Optimizing geometry where each run involves heavy finite-element analysis
- Minimizing noise from exhaust with few samples
- Gross tuning of multi-domain simulations (motor + thermal + acoustics)

**Why Different**:
- Builds probabilistic model (Gaussian Process)
- Balances exploration vs. exploitation (acquisition function)
- Sample-efficient (finds good solutions with minimal evaluations)

**Operator Signature**:
```morphogen
opt.bayesian<T>(
    bounds: Array<(f64, f64)>,
    objective: (Array<f64>) -> f64,
    n_initial: int = 5,
    n_iterations: int = 50,
    acquisition: "ei" | "ucb" | "pi" = "ei",  # Expected Improvement, UCB, Prob. Improvement
    kernel: "rbf" | "matern" = "matern",
    seed: int
) -> OptResult<Array<f64>> {
    gp_model: GaussianProcess,
    sampled_points: Array<Array<f64>>,
    sampled_values: Array<f64>,
    acquisition_history: Array<f64>,
    landscape_plot: Plot  # GP mean + uncertainty
}
```

**Example**:
```morphogen
# Optimize combustion chamber geometry (expensive CFD)
let result = opt.bayesian(
    bounds = [(50mm, 200mm), (10mm, 50mm), (100mm, 500mm)],  # diameter, height, cone_angle
    objective = |params| {
        let [D, H, angle] = params
        let chamber = combustion.chamber(D, H, angle)
        let cfd_result = cfd.simulate(chamber, fuel_rate, airflow)  # SLOW!
        return -cfd_result.efficiency  # Maximize efficiency
    },
    n_initial = 10,    # Random initial samples
    n_iterations = 40,  # Only 50 total CFD runs!
    acquisition = "ei"
)

# Morphogen stores GP model for reuse / visualization
viz.plot_landscape(result.gp_model, bounds)
```

**Dependencies**: Gaussian Processes (GP library), Stochastic, Optimization (for acquisition optimization)

**Implementation Priority**: **Phase 2** (critical for expensive simulations)

**Determinism**: DETERMINISTIC (with fixed seed for GP sampling)

**MLIR Integration**: GP inference can be lowered to linalg ops

---

### 3.2 Response Surface Modeling

**Best For**: Approximating smooth surfaces with polynomials/splines.

**Use Cases**:
- Approximating airflow vs. jet parameters (combustion)
- Fitting chassis resonance (mechanical vibrations)
- Reducing motor torque ripple models (surrogate for FEM)

**Operator Signature**:
```morphogen
opt.response_surface<T>(
    design_points: Array<Array<f64>>,
    objective: (Array<f64>) -> f64,
    model_type: "polynomial" | "spline" | "kriging" = "polynomial",
    polynomial_order: int = 2,
    n_refinement: int = 10  # Adaptive refinement iterations
) -> OptResult<Array<f64>> {
    surrogate_model: SurrogateModel,
    sampled_points: Array<Array<f64>>,
    prediction_error: f64
}
```

**Dependencies**: Linear Algebra (for polynomial fitting)

**Implementation Priority**: **Phase 3** (useful but less critical than Bayesian Optimization)

**Determinism**: DETERMINISTIC

---

### 3.3 Kriging / RBF Surrogates

**Best For**: Non-smooth, high-dimensional problems with limited samples.

**Use Cases**:
- Complex acoustics (multi-chamber mufflers)
- Flame shape prediction (combustion)
- Exhaust tuning (2-stroke engines)

**Operator Signature**:
```morphogen
opt.kriging<T>(
    bounds: Array<(f64, f64)>,
    objective: (Array<f64>) -> f64,
    n_initial: int = 20,
    kernel: "rbf" | "cubic" | "thinplate" = "rbf",
    nugget: f64 = 1e-6  # Regularization for noisy data
) -> OptResult<Array<f64>> {
    kriging_model: KrigingModel
}
```

**Dependencies**: Linear Algebra (for RBF matrix solve)

**Implementation Priority**: **Phase 3** (specialized use cases)

**Determinism**: DETERMINISTIC

---

## 4. Combinatorial / Discrete Optimization

**Purpose**: Optimize when parameters are **discrete** (hole counts, jet patterns, winding numbers, switching frequencies).

**Why Needed**: Many engineering problems have discrete choices:
- Number of baffle holes in a muffler
- Jet arrangement pattern in a fire pit
- Winding pattern in a stator
- PCB trace routing
- Discrete capacitor/inductor values (E12/E24 series)

**Strategy**: Search discrete state spaces using annealing, tabu search, or beam search.

---

### 4.1 Simulated Annealing (SA)

**Best For**: Rugged, discrete landscapes with many local optima.

**Use Cases**:
- Jet-hole pattern optimization (combustion)
- PCB trace routing with constraints (electromagnetics)
- Selecting discrete capacitor/inductor values (E12/E24 series in filters)
- Choosing muffler baffle counts (integer parameters)

**Operator Signature**:
```morphogen
opt.simulated_annealing<T>(
    initial: T,
    energy: (T) -> f64,        # Objective function
    neighbor: (T) -> T,         # Generate neighbor state
    temperature_schedule: (int) -> f64,  # T(iteration)
    max_iterations: int = 10000,
    seed: int
) -> OptResult<T> {
    temperature_history: Array<f64>,
    energy_history: Array<f64>,
    acceptance_rate: f64
}
```

**Example**:
```morphogen
# Optimize jet hole pattern (discrete positions on a grid)
struct JetPattern {
    holes: Array<(int, int)>  # Grid positions
}

let result = opt.simulated_annealing(
    initial = JetPattern { holes: random_pattern(10) },
    energy = |pattern| {
        let flame = combustion.simulate_flame(pattern)
        return -flame.uniformity  # Maximize uniformity
    },
    neighbor = |pattern| {
        # Move one hole to adjacent position
        let mut new_pattern = pattern.clone()
        let idx = random_int(0, pattern.holes.len())
        new_pattern.holes[idx] = random_adjacent(pattern.holes[idx])
        return new_pattern
    },
    temperature_schedule = |i| 100.0 / (1.0 + i as f64),
    max_iterations = 5000
)
```

**Dependencies**: Stochastic (for acceptance probability)

**Implementation Priority**: **Phase 1** (essential for discrete problems)

**Determinism**: DETERMINISTIC (with fixed seed)

---

### 4.2 Tabu Search

**Best For**: Avoiding revisiting poor regions in discrete search.

**Use Cases**:
- Optimizing logical layouts (control flow, signal routing)
- Combining multiple discrete constraints (manufacturing, assembly)

**Operator Signature**:
```morphogen
opt.tabu_search<T>(
    initial: T,
    objective: (T) -> f64,
    neighbor_generator: (T) -> Array<T>,
    tabu_tenure: int = 10,  # How long to remember tabu moves
    max_iterations: int = 1000,
    aspiration_criterion: bool = true  # Allow tabu if better than best
) -> OptResult<T> {
    tabu_list: Array<T>,
    objective_history: Array<f64>
}
```

**Dependencies**: None

**Implementation Priority**: **Phase 3** (specialized use cases)

**Determinism**: DETERMINISTIC

---

### 4.3 Beam Search / A* Variants

**Best For**: State-space exploration with constraints.

**Use Cases**:
- Multi-stage muffler design (sequential chamber optimization)
- Multi-zone jet design for fire pits (sequential placement)
- Optimizing winding patterns in stators (layer-by-layer)

**Operator Signature**:
```morphogen
opt.beam_search<T>(
    initial_states: Array<T>,
    expand: (T) -> Array<T>,   # Generate successors
    heuristic: (T) -> f64,     # Estimate cost-to-goal
    beam_width: int = 10,
    max_depth: int = 100
) -> OptResult<T> {
    search_tree: Tree<T>,
    nodes_expanded: int
}
```

**Dependencies**: None

**Implementation Priority**: **Phase 3** (specialized use cases)

**Determinism**: DETERMINISTIC

---

## 5. Multi-Objective Optimization

**Purpose**: Optimize **competing objectives** (Pareto frontiers).

**Why Critical**: Most real Morphogen problems have tradeoffs:
- **Minimize smoke AND maximize flame beauty** (combustion)
- **Maximize torque AND minimize current ripple** (motors)
- **Maximize muffler quietness AND maintain power** (acoustics)
- **Minimize cost AND maximize efficiency** (engineering design)

**Strategy**: Find Pareto-optimal solutions (no improvement in one objective without worsening another).

---

### 5.1 NSGA-II (Non-dominated Sorting Genetic Algorithm II)

**Best For**: Multi-objective evolutionary optimization (2-5 objectives).

**Use Cases**:
- Motor design (torque vs. efficiency vs. cost)
- Muffler design (noise reduction vs. backpressure vs. size)
- Speaker design (frequency response flatness vs. efficiency vs. size)
- Combustion chamber (efficiency vs. emissions vs. noise)

**Operator Signature**:
```morphogen
opt.nsga2<T>(
    genome: GenomeSpec<T>,
    objectives: Array<(T) -> f64>,  # Multiple objectives
    population_size: int = 100,
    generations: int = 100,
    mutation_rate: f64 = 0.01,
    crossover_rate: f64 = 0.9,
    seed: int
) -> MultiObjectiveResult<T> {
    pareto_front: Array<T>,
    pareto_values: Array<Array<f64>>,  # [obj1, obj2, ...] for each solution
    hypervolume: f64,
    pareto_plot: Plot
}
```

**Example**:
```morphogen
# Optimize muffler: quietness vs. backpressure
let genome = GenomeSpec({
    chamber_length: RangeParam(100mm, 500mm),
    chamber_diameter: RangeParam(50mm, 200mm),
    baffle_count: DiscreteParam([1, 2, 3, 4])
})

let result = opt.nsga2(
    genome,
    objectives = [
        |params| {  # Maximize noise reduction
            let muffler = acoustics.expansion_chamber(params)
            return acoustics.transmission_loss(muffler, 500Hz)
        },
        |params| {  # Minimize backpressure
            let muffler = acoustics.expansion_chamber(params)
            return -fluid.backpressure(muffler, flow_rate)
        }
    ],
    population_size = 100,
    generations = 50
)

# Result contains Pareto front (50+ solutions)
viz.plot_pareto_front(result.pareto_values, labels=["Noise Reduction", "Backpressure"])
```

**Dependencies**: Stochastic (for GA operations), Geometry (for Pareto dominance)

**Implementation Priority**: **Phase 2** (critical for real engineering design)

**Determinism**: DETERMINISTIC (with fixed seed)

---

### 5.2 SPEA2 (Strength Pareto Evolutionary Algorithm 2)

**Best For**: Complex tradeoff surfaces, archive-based diversity.

**Use Cases**:
- High-dimensional multi-objective problems (5+ objectives)
- Problems requiring fine-grained Pareto front resolution

**Operator Signature**:
```morphogen
opt.spea2<T>(
    genome: GenomeSpec<T>,
    objectives: Array<(T) -> f64>,
    population_size: int = 100,
    archive_size: int = 100,
    generations: int = 100,
    k: int = 1,  # k-th nearest neighbor for density
    seed: int
) -> MultiObjectiveResult<T>
```

**Dependencies**: Stochastic, Geometry (for k-NN distance)

**Implementation Priority**: **Phase 3** (advanced multi-objective)

**Determinism**: DETERMINISTIC (with fixed seed)

---

### 5.3 Multi-Objective Particle Swarm (MOPSO)

**Best For**: Continuous multi-objective problems with smooth landscapes.

**Use Cases**:
- Speaker crossover tuning (frequency response vs. phase)
- Control loop tuning (settling time vs. overshoot)

**Operator Signature**:
```morphogen
opt.mopso<T>(
    bounds: Array<(f64, f64)>,
    objectives: Array<(Array<f64>) -> f64>,
    n_particles: int = 30,
    max_iterations: int = 100,
    mutation_rate: f64 = 0.1,
    seed: int
) -> MultiObjectiveResult<Array<f64>>
```

**Dependencies**: Stochastic

**Implementation Priority**: **Phase 3** (specialized use case)

**Determinism**: DETERMINISTIC (with fixed seed)

---

## Integration with Morphogen Architecture

---

### Operator Registry

Each optimizer is registered as an operator:

```yaml
# opt.ga
operator:
  name: opt.ga
  layer: 3_construct
  domain: optimization
  signature: "<T>(GenomeSpec<T>, (T)->f64, ...) -> OptResult<T>"
  determinism: DETERMINISTIC
  dependencies: [stochastic]

# opt.bayesian
operator:
  name: opt.bayesian
  layer: 3_construct
  domain: optimization
  signature: "(Array<(f64,f64)>, (Array<f64>)->f64, ...) -> OptResult<Array<f64>>"
  determinism: DETERMINISTIC
  dependencies: [gaussian_process, stochastic, linalg]
```

---

### Operator Contract

All optimizers follow a unified contract:

**Inputs**:
1. **Parameter Space**
   - Continuous: `bounds: Array<(f64, f64)>`
   - Discrete: `genome: GenomeSpec<T>`
   - Mixed: `GenomeSpec` with both range and discrete params

2. **Objective Function(s)**
   - Single-objective: `(T) -> f64`
   - Multi-objective: `Array<(T) -> f64>`

3. **Hyperparameters**
   - Population size, iterations, learning rates, etc.
   - Sensible defaults for each algorithm

4. **Stopping Criteria**
   - Max iterations / evaluations
   - Tolerance (function value change)
   - Time budget

5. **Seed** (for deterministic RNG)

**Outputs**:
- `best: T` — Best solution found
- `best_fitness: f64` (or `Array<f64>` for multi-objective)
- `history: OptHistory` — Convergence tracking
- Algorithm-specific metadata (population, GP model, Pareto front, etc.)

---

### Simulation Subgraph Integration

Optimizers accept **Morphogen subgraphs** as objective functions:

```morphogen
# Define a simulation subgraph
scene MotorTorqueRipple(winding_pattern: Array<int>) {
    let motor = motors.pmsm(winding_pattern)
    let torque = motors.compute_torque(motor, current_profile)
    out ripple = stdev(torque)
}

# Optimize winding pattern
let result = opt.de(
    bounds = [(0, 100); 12],  # 12 winding slot positions
    fitness = |pattern| {
        let sim_result = simulate(MotorTorqueRipple(pattern))
        return -sim_result.ripple  # Minimize ripple
    },
    population_size = 30,
    generations = 50
)
```

**Key**: The subgraph is **compiled once**, then evaluated many times with different parameters.

---

### Surrogate Model Storage

Bayesian Optimization and Response Surface models are **first-class objects**:

```morphogen
# Train surrogate
let result = opt.bayesian(bounds, expensive_cfd_simulation, n_iterations=50)

# Save GP model
io.save(result.gp_model, "chamber_efficiency_surrogate.gp")

# Reuse in later session
let gp_model = io.load<GaussianProcess>("chamber_efficiency_surrogate.gp")
let predicted_efficiency = gp_model.predict([150mm, 30mm, 250mm])

# Visualize landscape
viz.plot_surface_3d(gp_model, bounds, title="Predicted Efficiency")
```

---

### Metrics Extraction

Optimizers integrate with Morphogen's metrics system:

```morphogen
metrics OptimizationMetrics {
    best_fitness: f64
    n_evaluations: int
    convergence_rate: f64
    population_diversity: f64  # For evolutionary algorithms
    acquisition_values: Array<f64>  # For Bayesian Optimization
}

scene OptimizeFilter {
    let result = opt.cmaes(bounds, fitness, max_generations=100)

    out metrics = OptimizationMetrics {
        best_fitness = result.best_fitness,
        n_evaluations = result.n_evaluations,
        convergence_rate = convergence_rate(result.fitness_history),
        population_diversity = stdev(result.population)
    }
}
```

---

### Visualization Support

Morphogen provides built-in visualization for optimization results:

```morphogen
# Convergence plot
viz.plot_convergence(result.fitness_history)

# Pareto front (multi-objective)
viz.plot_pareto_front(result.pareto_values, labels=["Obj1", "Obj2"])

# Parameter correlation
viz.plot_correlation_matrix(result.sampled_points, result.sampled_values)

# Surrogate landscape (Bayesian Optimization)
viz.plot_landscape_2d(result.gp_model, bounds, resolution=100)

# Population evolution (GA/PSO)
viz.animate_population(result.population_history)
```

---

## Implementation Roadmap

---

### Phase 1: Core Optimizers (v0.10)

**Target**: Establish OptimizationDomain with essential algorithms.

**Algorithms**:
1. **Genetic Algorithm (GA)** — Baseline evolutionary optimizer
2. **Differential Evolution (DE)** — Best general-purpose real-valued optimizer
3. **CMA-ES** — Gold standard for hard continuous problems
4. **Nelder-Mead** — Simple local optimizer, derivative-free
5. **Simulated Annealing (SA)** — Discrete + rugged landscapes

**Infrastructure**:
- `GenomeSpec<T>` for parameter encoding
- `OptResult<T>` output type
- Deterministic RNG integration
- Basic visualization (convergence plots)

**Testing**:
- Benchmark functions (Rosenbrock, Rastrigin, Ackley)
- Determinism validation (fixed seed → identical results)
- Cross-domain examples (audio filter tuning, geometry optimization)

**Deliverables**:
- `morphogen/stdlib/optimization.py` (Python reference)
- `morphogen.opt` MLIR dialect (lowering plan)
- `docs/specifications/optimization.md` (domain specification - to be created)

---

### Phase 2: Advanced & Surrogate Methods (v1.0)

**Target**: Enable expensive simulation optimization and multi-objective design.

**Algorithms**:
6. **Bayesian Optimization (BO)** — Sample-efficient for expensive simulations
7. **NSGA-II** — Multi-objective Pareto optimization
8. **L-BFGS** — Quasi-Newton for smooth problems
9. **Gradient Descent** — Requires autodiff integration
10. **Particle Swarm Optimization (PSO)** — Swarm-based global search

**Infrastructure**:
- Gaussian Process library (for Bayesian Optimization)
- Autodiff integration (for gradient-based methods)
- Multi-objective result types (`MultiObjectiveResult<T>`)
- Pareto front visualization

**Testing**:
- Multi-objective benchmarks (ZDT, DTLZ test suites)
- Expensive simulation mocks (artificial delays)
- Gradient correctness (finite differences vs. autodiff)

**Deliverables**:
- Surrogate model serialization (`io.save/load<GaussianProcess>`)
- Advanced visualization (Pareto fronts, GP landscapes)

---

### Phase 3: Specialized Algorithms (v1.1+)

**Target**: Complete algorithm catalog for niche use cases.

**Algorithms**:
11. **SPEA2** — Advanced multi-objective
12. **Response Surface Modeling** — Polynomial/spline surrogates
13. **Kriging / RBF Surrogates** — Non-smooth high-dim problems
14. **Tabu Search** — Discrete optimization with memory
15. **Beam Search** — State-space exploration
16. **Multi-Objective PSO (MOPSO)** — Swarm for multi-objective

**Infrastructure**:
- Constraint handling (inequality/equality constraints)
- Multi-fidelity optimization (cheap + expensive objectives)
- Parallel evaluation (distribute objective evaluations)

**Testing**:
- Constrained optimization benchmarks
- High-dimensional problems (50+ parameters)

---

## Cross-Domain Use Cases

---

### 1. Combustion: J-Tube Geometry Optimization

**Problem**: Optimize J-tube geometry for uniform flame, minimal smoke.

**Algorithm**: Genetic Algorithm (mixed continuous/discrete parameters)

**Parameters**:
- Jet hole positions (discrete grid)
- Pipe diameter (continuous)
- Air/fuel ratio (continuous)

```morphogen
let genome = GenomeSpec({
    jet_positions: GridPattern(10, 10),  # 10x10 grid
    pipe_diameter: RangeParam(50mm, 150mm),
    air_fuel_ratio: RangeParam(10.0, 20.0)
})

let result = opt.ga(
    genome,
    fitness = |params| {
        let jtube = combustion.jtube(params)
        let flame = combustion.simulate_flame(jtube)
        return flame.uniformity - 0.5 * flame.smoke_index
    },
    population_size = 50,
    generations = 100
)
```

---

### 2. Acoustics: Muffler Multi-Objective Design

**Problem**: Maximize noise reduction, minimize backpressure.

**Algorithm**: NSGA-II (multi-objective)

**Parameters**:
- Chamber dimensions
- Baffle count
- Perforate hole size

```morphogen
let result = opt.nsga2(
    genome = GenomeSpec({
        chamber_length: RangeParam(100mm, 500mm),
        chamber_diameter: RangeParam(50mm, 200mm),
        baffle_count: DiscreteParam([1, 2, 3, 4, 5]),
        hole_diameter: RangeParam(5mm, 20mm)
    }),
    objectives = [
        |p| acoustics.transmission_loss(muffler(p), 500Hz),  # Maximize
        |p| -fluid.backpressure(muffler(p), 100 L/min)      # Minimize (negate)
    ],
    population_size = 100,
    generations = 50
)

# Visualize Pareto front
viz.plot_pareto_front(result.pareto_values)
```

---

### 3. Motors: PID Tuning with Differential Evolution

**Problem**: Tune PID controller for minimal overshoot and settling time.

**Algorithm**: Differential Evolution (continuous parameters)

**Parameters**: Kp, Ki, Kd

```morphogen
let result = opt.de(
    bounds = [(0.0, 10.0), (0.0, 5.0), (0.0, 1.0)],  # Kp, Ki, Kd
    fitness = |params| {
        let [Kp, Ki, Kd] = params
        let controller = control.pid(Kp, Ki, Kd)
        let response = motors.step_response(motor, controller)
        return -(response.overshoot + 0.5 * response.settling_time)
    },
    population_size = 30,
    generations = 50
)
```

---

### 4. Geometry: CAD Parameter Fitting with CMA-ES

**Problem**: Fit 20-parameter geometric model to scanned point cloud.

**Algorithm**: CMA-ES (high-dimensional continuous)

**Parameters**: 20 control point positions

```morphogen
let result = opt.cmaes(
    initial_mean = [0.0; 20],
    initial_sigma = 5.0,
    bounds = [(−50mm, 50mm); 20],
    fitness = |params| {
        let model = geometry.parametric_surface(params)
        let point_cloud = io.load("scan.ply")
        return -geometry.hausdorff_distance(model, point_cloud)
    },
    max_generations = 200
)
```

---

### 5. Expensive Simulation: Bayesian Optimization for CFD

**Problem**: Optimize combustion chamber with only 50 CFD evaluations.

**Algorithm**: Bayesian Optimization (sample-efficient)

**Parameters**: Chamber geometry (3D)

```morphogen
let result = opt.bayesian(
    bounds = [(100mm, 300mm), (50mm, 150mm), (10deg, 45deg)],
    objective = |params| {
        let [D, H, angle] = params
        let chamber = combustion.chamber(D, H, angle)
        let cfd = cfd.simulate(chamber, dt=0.001s, steps=10000)  # EXPENSIVE!
        return cfd.efficiency
    },
    n_initial = 10,
    n_iterations = 40  # Only 50 total CFD runs
)

# Surrogate model captures landscape
viz.plot_landscape_3d(result.gp_model, bounds)
```

---

## What Morphogen Gains

By adding comprehensive optimization support, Morphogen transforms from **"simulate physics"** to **"discover new designs"**:

1. **Automatic Motor Tuning** — Optimize winding patterns, magnet shapes, control loops
2. **Muffler Shape Evolution** — Multi-objective noise vs. backpressure
3. **Flame Shape Discovery** — J-tube geometry, jet patterns
4. **Speaker + Room Tuning** — EQ, crossover, placement
5. **Acoustic Material Discovery** — Perforate patterns, chamber dimensions
6. **Optimal Tables for LC Filters** — Component value selection
7. **2-Stroke Expansion Chamber Design** — Length, diameter, taper
8. **Parametric CAD → Simulation → Optimization Loops** — TiaCAD integration
9. **GA-Tuned Control Loops** — PID, MPC, LQR optimization
10. **Optimization-Guided Learning from Recorded Audio** — Inverse problems

---

## Determinism Guarantees

All optimization algorithms support **deterministic execution**:

```morphogen
# Same seed → identical results
let result1 = opt.ga(genome, fitness, seed=42)
let result2 = opt.ga(genome, fitness, seed=42)
assert_eq!(result1.best, result2.best)
assert_eq!(result1.best_fitness, result2.best_fitness)
```

**Determinism Tier**: DETERMINISTIC (with fixed seed)
- Bit-exact reproduction across platforms
- Enables regression testing
- Supports reproducible research

---

## MLIR Lowering Strategy

Optimization operators are **high-level constructs** that lower to:

1. **Control Flow** (scf dialect) — For/while loops over generations/iterations
2. **Linear Algebra** (linalg dialect) — For surrogate models (GP, RBF)
3. **Stochastic** (morphogen.stochastic dialect) — For mutation, crossover, sampling
4. **Function Calls** (func dialect) — For user-provided objective functions

**Example (Genetic Algorithm)**:
```mlir
// High-level
%result = morphogen.opt.ga %genome, %fitness, %config

// Lowers to:
%population = morphogen.stochastic.init_population %genome, %pop_size, %seed
scf.for %gen = 0 to %n_generations {
    %fitness_vals = scf.for %i = 0 to %pop_size {
        %individual = tensor.extract %population[%i]
        %fit = func.call @fitness(%individual)
        scf.yield %fit
    }
    %selected = morphogen.opt.tournament_select %population, %fitness_vals
    %offspring = morphogen.opt.crossover %selected
    %mutated = morphogen.opt.mutate %offspring, %mutation_rate, %seed
    %population = %mutated
}
```

---

## Testing Strategy

### 1. Benchmark Functions

Standard optimization test functions:

| Function | Type | Difficulty | Global Optimum |
|----------|------|------------|----------------|
| **Sphere** | Unimodal, convex | Easy | f(0) = 0 |
| **Rosenbrock** | Unimodal, narrow valley | Medium | f(1,1) = 0 |
| **Rastrigin** | Multimodal, many local optima | Hard | f(0) = 0 |
| **Ackley** | Multimodal, steep | Hard | f(0) = 0 |
| **Schwefel** | Multimodal, deceptive | Very Hard | f(420.97) = 0 |

### 2. Determinism Tests

```morphogen
@test("GA determinism")
fn test_ga_determinism() {
    let genome = GenomeSpec({ x: RangeParam(−5.0, 5.0) })
    let result1 = opt.ga(genome, rosenbrock, seed=42)
    let result2 = opt.ga(genome, rosenbrock, seed=42)
    assert_eq!(result1.best, result2.best)
}
```

### 3. Convergence Tests

```morphogen
@test("CMA-ES converges on Rosenbrock")
fn test_cmaes_convergence() {
    let result = opt.cmaes(
        initial_mean = [−3.0, −3.0],
        initial_sigma = 1.0,
        bounds = [(−5.0, 5.0), (−5.0, 5.0)],
        fitness = rosenbrock,
        max_generations = 500
    )
    assert!(result.best_fitness < 1e-6)  # Near-optimal
    assert_approx_eq!(result.best, [1.0, 1.0], tol=1e-3)
}
```

### 4. Multi-Objective Tests

```morphogen
@test("NSGA-II finds Pareto front")
fn test_nsga2_pareto() {
    let result = opt.nsga2(
        genome = GenomeSpec({ x: RangeParam(0.0, 1.0) }),
        objectives = [
            |x| x,      # Minimize x
            |x| 1.0 - x  # Minimize (1-x)
        ],
        population_size = 100,
        generations = 50
    )
    # Pareto front should span [0, 1]
    assert!(result.pareto_front.len() > 10)
    let x_values = result.pareto_front.map(|p| p.x)
    assert!(x_values.min() < 0.1)
    assert!(x_values.max() > 0.9)
}
```

---

## Performance Considerations

### Parallelization

Objective function evaluations are **embarrassingly parallel**:

```morphogen
# Automatic parallelization
@parallel(strategy="rayon")
let fitness_values = population.map(|individual| fitness(individual))
```

**MLIR Integration**:
- `scf.parallel` for population evaluation
- GPU offload for batch simulations

### Incremental Compilation

Optimizers reuse compiled simulation subgraphs:

```morphogen
# Compile once
let compiled_sim = compile(MotorTorqueRipple)

# Evaluate many times
let fitness = |params| {
    return compiled_sim.run(params).ripple
}

opt.de(bounds, fitness, ...)  # Fast!
```

### Surrogate Model Caching

Bayesian Optimization caches GP models:

```morphogen
# First run (50 evaluations)
let result1 = opt.bayesian(bounds, expensive_cfd, n_iterations=50)
io.save(result1.gp_model, "model.gp")

# Continue optimization (another 50 evaluations)
let gp_model = io.load<GaussianProcess>("model.gp")
let result2 = opt.bayesian(
    bounds,
    expensive_cfd,
    initial_model = gp_model,  # Warm start!
    n_iterations = 50
)
```

---

## Documentation Requirements

### User Guide

- **Choosing an Algorithm** — Decision tree based on problem type
- **Hyperparameter Tuning** — Guidance for each algorithm
- **Troubleshooting** — Common issues (slow convergence, premature convergence)

### API Reference

- Operator signatures
- Parameter descriptions
- Return types
- Examples for each algorithm

### Domain Examples

- **Combustion**: J-tube optimization (GA)
- **Acoustics**: Muffler multi-objective (NSGA-II)
- **Motors**: PID tuning (DE)
- **Geometry**: Parameter fitting (CMA-ES)
- **Expensive Sim**: CFD optimization (Bayesian)

---

## Summary: Priority Algorithm List

### Phase 1 (v0.10) — Must-Have

1. **Genetic Algorithm (GA)** — Baseline evolutionary
2. **Differential Evolution (DE)** — Best general-purpose real-valued
3. **CMA-ES** — Gold standard for hard continuous
4. **Nelder-Mead** — Simple local, derivative-free
5. **Simulated Annealing (SA)** — Discrete + rugged

### Phase 2 (v1.0) — High-Value

6. **Bayesian Optimization** — Sample-efficient for expensive sims
7. **NSGA-II** — Multi-objective Pareto
8. **L-BFGS** — Quasi-Newton for smooth problems
9. **Gradient Descent** — Autodiff integration
10. **Particle Swarm (PSO)** — Swarm-based global search

### Phase 3 (v1.1+) — Complete Catalog

11. **SPEA2** — Advanced multi-objective
12. **Response Surface** — Polynomial/spline surrogates
13. **Kriging / RBF** — Non-smooth high-dim
14. **Tabu Search** — Discrete with memory
15. **Beam Search** — State-space exploration
16. **MOPSO** — Multi-objective PSO

---

## References

### Optimization Theory
- **Genetic Algorithms**: Goldberg, "Genetic Algorithms in Search, Optimization, and Machine Learning" (1989)
- **Differential Evolution**: Storn & Price, "Differential Evolution – A Simple and Efficient Heuristic" (1997)
- **CMA-ES**: Hansen & Ostermeier, "Completely Derandomized Self-Adaptation in Evolution Strategies" (2001)
- **Bayesian Optimization**: Brochu et al., "A Tutorial on Bayesian Optimization" (2010)
- **NSGA-II**: Deb et al., "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II" (2002)

### Morphogen Architecture
- **../architecture/domain-architecture.md** — Section 2.3 (Optimization domain overview)
- **ADR-002** — Cross-domain architectural patterns
- **../specifications/operator-registry.md** — Operator registration
- **../specifications/type-system.md** — GenomeSpec type definitions
- **../architecture/gpu-mlir-principles.md** — Parallel evaluation strategies

### Implementation Libraries (Python Reference)
- **SciPy**: `scipy.optimize` (Nelder-Mead, L-BFGS)
- **DEAP**: Genetic algorithms, NSGA-II
- **PyMoo**: Multi-objective optimization
- **scikit-optimize**: Bayesian Optimization (Gaussian Processes)
- **CMA**: CMA-ES reference implementation

---

**End of Document**
