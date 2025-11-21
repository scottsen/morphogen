# Genetic Algorithm Operators Catalog

**Domain:** Genetic Algorithm (GA) & Evolutionary Computation
**Related Examples:** [Racing AI Pipeline](../EXAMPLES/RACING-AI-PIPELINE.md)
**Status:** Operator Catalog & Design Specification

---

## 1. Overview

This document catalogs the genetic algorithm operators needed for evolutionary computation in Morphogen, with a focus on **neural network evolution** (neuroevolution) but generalizable to any optimization problem.

**Why GA operators in Morphogen?**

Genetic algorithms are a natural fit for Morphogen's operator paradigm:
- **Deterministic** (with fixed RNG seed)
- **Parallelizable** (population-level operations)
- **Composable** (mix mutation, crossover, selection strategies)
- **GPU-accelerated** (batch tensor operations)

---

## 2. Core GA Operator Categories

```
Genetic Algorithm Operators
├── Selection (choose parents)
├── Crossover (combine genomes)
├── Mutation (random variation)
├── Evaluation (fitness calculation)
├── Replacement (next generation)
└── Advanced (speciation, adaptation, multi-objective)
```

---

## 3. Selection Operators

### 3.1 Tournament Selection

**Description:** Randomly sample k individuals, return the fittest.

**Operator Spec:**
```yaml
ga.tournament_select:
  inputs:
    - population: [[tensor]]  # List of genomes
    - fitness: [f32]          # Fitness scores
    - tournament_size: i32    # Number of competitors
    - rng: RNGState           # Random state
  outputs:
    - selected: [tensor]      # Selected genome
    - new_rng: RNGState

  properties:
    deterministic: true
    parallel: true
    complexity: O(k)

  hyperparameters:
    tournament_size:
      typical: 4
      range: [2, 10]
      effect: "Higher = more selection pressure"
```

**Algorithm:**
```python
def tournament_select(population, fitness, k, rng):
    indices = random_sample(rng, range(len(population)), k)
    tournament_fitness = [fitness[i] for i in indices]
    winner_idx = indices[argmax(tournament_fitness)]
    return population[winner_idx]
```

**Pros:**
- Simple, efficient
- Tunable selection pressure
- Works with negative fitness

**Cons:**
- Can lose diversity quickly with large k

---

### 3.2 Fitness-Proportionate Selection (Roulette Wheel)

**Description:** Probability of selection proportional to fitness.

**Operator Spec:**
```yaml
ga.roulette_select:
  inputs:
    - population: [[tensor]]
    - fitness: [f32]
    - rng: RNGState
  outputs:
    - selected: [tensor]
    - new_rng: RNGState

  constraints:
    - fitness must be non-negative
```

**Algorithm:**
```python
def roulette_select(population, fitness, rng):
    total_fitness = sum(fitness)
    probabilities = [f / total_fitness for f in fitness]
    index = random_choice(rng, probabilities)
    return population[index]
```

**Pros:**
- Natural interpretation (fitness → selection probability)

**Cons:**
- Requires non-negative fitness
- Premature convergence if one individual dominates

---

### 3.3 Rank Selection

**Description:** Selection based on fitness rank, not absolute value.

**Operator Spec:**
```yaml
ga.rank_select:
  inputs:
    - population: [[tensor]]
    - fitness: [f32]
    - selection_pressure: f32  # 1.0 = uniform, 2.0 = strong
    - rng: RNGState
  outputs:
    - selected: [tensor]
    - new_rng: RNGState
```

**Algorithm:**
```python
def rank_select(population, fitness, pressure, rng):
    ranks = argsort(fitness)  # 0 = worst, N-1 = best
    probabilities = [(pressure - 1) * r / (N-1) + 1 for r in ranks]
    probabilities /= sum(probabilities)
    index = random_choice(rng, probabilities)
    return population[index]
```

**Pros:**
- Robust to fitness scaling
- Prevents premature convergence

**Cons:**
- Slower than tournament
- Requires sorting

---

### 3.4 Stochastic Universal Sampling (SUS)

**Description:** Low-variance sampling for selecting multiple individuals at once.

**Operator Spec:**
```yaml
ga.sus_select:
  inputs:
    - population: [[tensor]]
    - fitness: [f32]
    - num_select: i32  # How many to select
    - rng: RNGState
  outputs:
    - selected: [[tensor]]
    - new_rng: RNGState

  properties:
    parallel: true
    low_variance: true
```

**Algorithm:**
```python
def sus_select(population, fitness, n, rng):
    total = sum(fitness)
    step = total / n
    start = random_uniform(rng, 0, step)

    pointers = [start + i * step for i in range(n)]
    selected = []

    cumulative = 0
    for i, f in enumerate(fitness):
        cumulative += f
        while pointers and pointers[0] < cumulative:
            selected.append(population[i])
            pointers.pop(0)

    return selected
```

**Pros:**
- Minimal selection variance
- Preserves diversity better than roulette

**Cons:**
- More complex implementation

---

## 4. Crossover Operators

### 4.1 Uniform Crossover

**Description:** Each gene independently chosen from parent1 or parent2.

**Operator Spec:**
```yaml
ga.uniform_crossover:
  inputs:
    - parent1: [tensor]
    - parent2: [tensor]
    - crossover_rate: f32  # Probability per gene
    - rng: RNGState
  outputs:
    - offspring: [tensor]
    - new_rng: RNGState

  properties:
    deterministic: true
    parallel: true
    gpu_friendly: true
```

**Algorithm:**
```python
def uniform_crossover(p1, p2, rate, rng):
    offspring = copy(p1)
    for i in range(len(offspring)):
        if random(rng) < rate:
            offspring[i] = p2[i]
    return offspring
```

**GPU-optimized version:**
```python
def uniform_crossover_gpu(p1, p2, rate, rng):
    mask = random_uniform(rng, shape=p1.shape) < rate
    return where(mask, p2, p1)  # Element-wise select
```

**Pros:**
- Maximum mixing of genes
- GPU-friendly (vectorized)

**Cons:**
- Can break co-adapted gene groups

---

### 4.2 Single-Point Crossover

**Description:** Split genomes at random point, swap tails.

**Operator Spec:**
```yaml
ga.single_point_crossover:
  inputs:
    - parent1: [tensor]
    - parent2: [tensor]
    - rng: RNGState
  outputs:
    - offspring1: [tensor]
    - offspring2: [tensor]
    - new_rng: RNGState
```

**Algorithm:**
```python
def single_point_crossover(p1, p2, rng):
    point = random_int(rng, 1, len(p1) - 1)
    offspring1 = concatenate(p1[:point], p2[point:])
    offspring2 = concatenate(p2[:point], p1[point:])
    return offspring1, offspring2
```

**Pros:**
- Preserves gene linkage better than uniform

**Cons:**
- Position-dependent (bias toward head/tail)

---

### 4.3 Layer-Wise Crossover (Neural Networks)

**Description:** Swap entire layers between parent networks.

**Operator Spec:**
```yaml
ga.layer_crossover:
  inputs:
    - parent1: NeuralNetwork  # Structured genome
    - parent2: NeuralNetwork
    - rng: RNGState
  outputs:
    - offspring: NeuralNetwork
    - new_rng: RNGState

  constraints:
    - Networks must have same architecture
```

**Algorithm:**
```python
def layer_crossover(p1, p2, rng):
    offspring = empty_network(p1.architecture)

    for layer_idx in range(len(p1.layers)):
        if random(rng) < 0.5:
            offspring.layers[layer_idx] = copy(p1.layers[layer_idx])
        else:
            offspring.layers[layer_idx] = copy(p2.layers[layer_idx])

    return offspring
```

**Pros:**
- Respects layer structure
- Can transfer learned features

**Cons:**
- Only works for fixed architectures

---

### 4.4 Blend Crossover (BLX-α)

**Description:** Offspring values are weighted averages of parents.

**Operator Spec:**
```yaml
ga.blend_crossover:
  inputs:
    - parent1: [tensor]
    - parent2: [tensor]
    - alpha: f32  # Blend factor (typical: 0.5)
    - rng: RNGState
  outputs:
    - offspring: [tensor]
    - new_rng: RNGState
```

**Algorithm:**
```python
def blend_crossover(p1, p2, alpha, rng):
    # For each gene:
    offspring = []
    for g1, g2 in zip(p1, p2):
        min_val = min(g1, g2) - alpha * abs(g1 - g2)
        max_val = max(g1, g2) + alpha * abs(g1 - g2)
        offspring.append(random_uniform(rng, min_val, max_val))
    return offspring
```

**Pros:**
- Smooth exploration
- Works well for real-valued genes (NN weights)

**Cons:**
- Can drift outside parent range (exploration vs exploitation trade-off)

---

## 5. Mutation Operators

### 5.1 Gaussian Mutation

**Description:** Add Gaussian noise to genes.

**Operator Spec:**
```yaml
ga.gaussian_mutate:
  inputs:
    - genome: [tensor]
    - mutation_rate: f32  # Probability per gene
    - std: f32            # Mutation strength
    - rng: RNGState
  outputs:
    - mutated: [tensor]
    - new_rng: RNGState

  properties:
    deterministic: true
    gpu_friendly: true

  hyperparameters:
    mutation_rate:
      typical: 0.01 - 0.15
      effect: "Higher = more exploration"
    std:
      typical: 0.05
      adaptive: "Can decay over generations"
```

**Algorithm:**
```python
def gaussian_mutate(genome, rate, std, rng):
    mutated = copy(genome)
    for i in range(len(mutated)):
        if random(rng) < rate:
            mutated[i] += gaussian(rng, 0, std)
    return mutated
```

**GPU-optimized:**
```python
def gaussian_mutate_gpu(genome, rate, std, rng):
    mask = random_uniform(rng, shape=genome.shape) < rate
    noise = gaussian(rng, 0, std, shape=genome.shape)
    return genome + where(mask, noise, 0)
```

**Pros:**
- Simple, effective
- GPU-friendly
- Continuous exploration

**Cons:**
- Can produce invalid values (need clamping)

---

### 5.2 Uniform Mutation

**Description:** Replace gene with random value from range.

**Operator Spec:**
```yaml
ga.uniform_mutate:
  inputs:
    - genome: [tensor]
    - mutation_rate: f32
    - value_range: [f32, f32]  # [min, max]
    - rng: RNGState
  outputs:
    - mutated: [tensor]
    - new_rng: RNGState
```

**Algorithm:**
```python
def uniform_mutate(genome, rate, value_range, rng):
    mutated = copy(genome)
    for i in range(len(mutated)):
        if random(rng) < rate:
            mutated[i] = random_uniform(rng, *value_range)
    return mutated
```

**Pros:**
- Large jumps (exploration)
- Bounded values

**Cons:**
- Can destroy good genes

---

### 5.3 Adaptive Mutation

**Description:** Mutation strength adapts based on fitness landscape.

**Operator Spec:**
```yaml
ga.adaptive_mutate:
  inputs:
    - genome: [tensor]
    - mutation_rate: f32
    - fitness_history: [f32]  # Recent fitness values
    - rng: RNGState
  outputs:
    - mutated: [tensor]
    - new_mutation_rate: f32
    - new_rng: RNGState

  adaptation_strategy:
    - If fitness improving: decrease mutation (exploitation)
    - If fitness stagnant: increase mutation (exploration)
```

**Algorithm:**
```python
def adaptive_mutate(genome, rate, fitness_history, rng):
    # Check if fitness is improving
    if is_improving(fitness_history):
        new_rate = rate * 0.9  # Decrease mutation
    elif is_stagnant(fitness_history):
        new_rate = rate * 1.1  # Increase mutation
    else:
        new_rate = rate

    new_rate = clamp(new_rate, 0.01, 0.5)

    # Apply Gaussian mutation with adapted rate
    mutated = gaussian_mutate(genome, new_rate, std=0.05, rng)

    return mutated, new_rate
```

**Pros:**
- Self-tuning
- Balances exploration/exploitation

**Cons:**
- Requires fitness tracking
- More complex

---

### 5.4 Neuron Reinitialize Mutation

**Description:** Completely reset a random neuron's weights.

**Operator Spec:**
```yaml
ga.neuron_reinit_mutate:
  inputs:
    - network: NeuralNetwork
    - mutation_rate: f32
    - init_method: str  # "he", "xavier", "uniform"
    - rng: RNGState
  outputs:
    - mutated_network: NeuralNetwork
    - new_rng: RNGState
```

**Algorithm:**
```python
def neuron_reinit_mutate(network, rate, init_method, rng):
    for layer in network.layers:
        for neuron_idx in range(layer.size):
            if random(rng) < rate:
                # Reinitialize all incoming weights to this neuron
                layer.weights[:, neuron_idx] = initialize_weights(
                    layer.input_size,
                    method=init_method,
                    rng=rng
                )
    return network
```

**Pros:**
- Large structural change
- Can escape local optima

**Cons:**
- Disruptive (use low rate)

---

## 6. Advanced Operators

### 6.1 NEAT-Style Topology Mutation

**Description:** Add/remove neurons or connections (evolving network structure).

**Operator Spec:**
```yaml
ga.neat_mutate_topology:
  inputs:
    - genome: NeuralGraph  # Variable topology
    - innovation_db: InnovationDB  # Tracks historical mutations
    - mutation_probs:
        add_node: 0.03
        add_connection: 0.05
        remove_connection: 0.01
    - rng: RNGState
  outputs:
    - mutated_genome: NeuralGraph
    - updated_innovation_db: InnovationDB
    - new_rng: RNGState

  properties:
    variable_topology: true
    historical_tracking: true
```

**Algorithm:**
```python
def neat_mutate_topology(genome, innovation_db, probs, rng):
    # Add node: split an existing connection
    if random(rng) < probs.add_node:
        connection = random_choice(genome.connections)
        new_node = create_node()
        genome.add_node(new_node)
        genome.remove_connection(connection)
        genome.add_connection(connection.src, new_node, weight=1.0)
        genome.add_connection(new_node, connection.dst, weight=connection.weight)
        innovation_db.record(new_node)

    # Add connection: link two unconnected nodes
    if random(rng) < probs.add_connection:
        src, dst = random_unconnected_pair(genome)
        genome.add_connection(src, dst, weight=random_weight())
        innovation_db.record((src, dst))

    # Remove connection
    if random(rng) < probs.remove_connection and len(genome.connections) > 0:
        connection = random_choice(genome.connections)
        genome.remove_connection(connection)

    return genome, innovation_db
```

**Pros:**
- Evolves architecture
- Can discover novel structures

**Cons:**
- Complex implementation
- Requires speciation (see below)

---

### 6.2 Speciation

**Description:** Divide population into species to preserve diversity.

**Operator Spec:**
```yaml
ga.speciate:
  inputs:
    - population: [Genome]
    - compatibility_threshold: f32
    - distance_metric: str  # "genomic", "behavioral"
  outputs:
    - species: [[Genome]]  # List of species
    - species_representatives: [Genome]

  properties:
    preserves_diversity: true
```

**Algorithm:**
```python
def speciate(population, threshold, distance_metric):
    species = []
    representatives = []

    for genome in population:
        # Try to match to existing species
        matched = False
        for i, rep in enumerate(representatives):
            if distance(genome, rep, distance_metric) < threshold:
                species[i].append(genome)
                matched = True
                break

        # Create new species if no match
        if not matched:
            species.append([genome])
            representatives.append(genome)

    return species, representatives
```

**Distance Metrics:**

**Genomic distance (for NEAT):**
```python
def genomic_distance(g1, g2):
    excess = count_non_matching_genes(g1, g2)
    disjoint = count_disjoint_genes(g1, g2)
    weight_diff = mean_weight_difference(g1, g2)

    c1, c2, c3 = 1.0, 1.0, 0.4  # Coefficients
    N = max(len(g1.genes), len(g2.genes))

    return (c1 * excess + c2 * disjoint) / N + c3 * weight_diff
```

**Behavioral distance (for any genome):**
```python
def behavioral_distance(g1, g2, test_cases):
    # Run both genomes on test cases, compare outputs
    outputs1 = [evaluate(g1, test) for test in test_cases]
    outputs2 = [evaluate(g2, test) for test in test_cases]

    return mean([euclidean(o1, o2) for o1, o2 in zip(outputs1, outputs2)])
```

**Pros:**
- Preserves diversity
- Prevents premature convergence

**Cons:**
- Computational overhead
- Requires tuning threshold

---

### 6.3 Fitness Sharing

**Description:** Penalize fitness of individuals in crowded niches.

**Operator Spec:**
```yaml
ga.fitness_sharing:
  inputs:
    - population: [Genome]
    - raw_fitness: [f32]
    - niche_radius: f32
    - distance_metric: str
  outputs:
    - shared_fitness: [f32]
```

**Algorithm:**
```python
def fitness_sharing(population, raw_fitness, radius, distance_metric):
    shared_fitness = []

    for i, genome_i in enumerate(population):
        niche_count = 0

        for genome_j in population:
            d = distance(genome_i, genome_j, distance_metric)
            if d < radius:
                niche_count += 1 - (d / radius)  # Sharing function

        shared_fitness.append(raw_fitness[i] / niche_count)

    return shared_fitness
```

**Pros:**
- Maintains diversity
- Encourages exploration of multiple niches

**Cons:**
- O(N²) complexity
- Requires distance metric

---

### 6.4 Multi-Objective Optimization (Pareto)

**Description:** Optimize multiple conflicting objectives simultaneously.

**Operator Spec:**
```yaml
ga.pareto_rank:
  inputs:
    - objectives: tensor<NxMxf32>  # N individuals, M objectives
  outputs:
    - pareto_ranks: [i32]          # Rank (0 = Pareto front)
    - crowding_distance: [f32]      # Diversity measure

  properties:
    multi_objective: true
    preserves_diversity: true
```

**Algorithm:**
```python
def pareto_rank(objectives):
    N, M = objectives.shape
    ranks = [0] * N
    dominated_by = [[] for _ in range(N)]
    dominates_count = [0] * N

    # Compute domination relationships
    for i in range(N):
        for j in range(i + 1, N):
            if dominates(objectives[i], objectives[j]):
                dominated_by[i].append(j)
                dominates_count[j] += 1
            elif dominates(objectives[j], objectives[i]):
                dominated_by[j].append(i)
                dominates_count[i] += 1

    # Assign ranks
    current_front = [i for i in range(N) if dominates_count[i] == 0]
    rank = 0

    while current_front:
        for i in current_front:
            ranks[i] = rank

        next_front = []
        for i in current_front:
            for j in dominated_by[i]:
                dominates_count[j] -= 1
                if dominates_count[j] == 0:
                    next_front.append(j)

        current_front = next_front
        rank += 1

    # Compute crowding distance
    crowding = crowding_distance(objectives, ranks)

    return ranks, crowding

def dominates(obj1, obj2):
    # obj1 dominates obj2 if it's better in all objectives
    return all(o1 <= o2 for o1, o2 in zip(obj1, obj2)) and \
           any(o1 < o2 for o1, o2 in zip(obj1, obj2))
```

**Crowding Distance:**
```python
def crowding_distance(objectives, ranks):
    N, M = objectives.shape
    crowding = [0.0] * N

    # For each rank level
    for rank in set(ranks):
        front = [i for i, r in enumerate(ranks) if r == rank]

        if len(front) <= 2:
            for i in front:
                crowding[i] = float('inf')
            continue

        # For each objective
        for m in range(M):
            # Sort by objective m
            sorted_indices = sorted(front, key=lambda i: objectives[i, m])

            # Boundary points get infinite distance
            crowding[sorted_indices[0]] = float('inf')
            crowding[sorted_indices[-1]] = float('inf')

            # Range of objective m
            obj_range = objectives[sorted_indices[-1], m] - objectives[sorted_indices[0], m]
            if obj_range == 0:
                continue

            # Middle points
            for i in range(1, len(sorted_indices) - 1):
                idx = sorted_indices[i]
                crowding[idx] += (objectives[sorted_indices[i+1], m] -
                                  objectives[sorted_indices[i-1], m]) / obj_range

    return crowding
```

**Usage in Selection:**
```python
def nsga2_select(population, objectives):
    ranks, crowding = pareto_rank(objectives)

    # Select based on:
    # 1. Lower rank (closer to Pareto front)
    # 2. Higher crowding distance (more diverse)

    selected = sorted(
        range(len(population)),
        key=lambda i: (ranks[i], -crowding[i])
    )

    return [population[i] for i in selected[:N//2]]
```

**Pros:**
- Finds diverse set of solutions
- No need to weight objectives

**Cons:**
- Complex implementation
- Slower than single-objective

---

## 7. Evaluation & Replacement Operators

### 7.1 Parallel Population Evaluation

**Operator Spec:**
```yaml
ga.evaluate_population:
  inputs:
    - population: [[tensor]]
    - evaluator: Graph  # Morphogen graph for fitness evaluation
    - parallel: bool
  outputs:
    - fitness_scores: [f32]

  properties:
    parallel: true
    gpu_accelerated: true
    batch_friendly: true
```

**Algorithm (GPU-optimized):**
```python
def evaluate_population_gpu(population, evaluator):
    # Stack population into batch tensor
    batch = stack(population)  # shape: [N, genome_size]

    # Run evaluator graph in batch mode
    # (e.g., batch neural network inference + physics simulation)
    batch_fitness = evaluator.run(batch)  # shape: [N]

    return batch_fitness
```

**Example for Racing AI:**
```python
evaluator_graph:
  - sensors = raycast_batch(car_states, track)
  - actions = nn_batch(population, sensors)
  - new_states = physics_batch(car_states, actions)
  - fitness = compute_fitness(new_states)
```

---

### 7.2 Generational Replacement

**Operator Spec:**
```yaml
ga.generational_replacement:
  inputs:
    - old_population: [[tensor]]
    - new_population: [[tensor]]
    - old_fitness: [f32]
    - new_fitness: [f32]
    - elitism_count: i32
  outputs:
    - next_population: [[tensor]]
```

**Algorithm:**
```python
def generational_replacement(old_pop, new_pop, old_fit, new_fit, elitism):
    # Keep top elites from old population
    elite_indices = argsort(old_fit)[-elitism:]
    elites = [old_pop[i] for i in elite_indices]

    # Fill rest with new population
    next_pop = elites + new_pop[:len(new_pop) - elitism]

    return next_pop
```

---

### 7.3 Steady-State Replacement

**Operator Spec:**
```yaml
ga.steady_state_replacement:
  inputs:
    - population: [[tensor]]
    - fitness: [f32]
    - offspring: [tensor]
    - offspring_fitness: f32
  outputs:
    - new_population: [[tensor]]
    - new_fitness: [f32]
```

**Algorithm:**
```python
def steady_state_replacement(population, fitness, offspring, off_fit):
    # Replace worst individual if offspring is better
    worst_idx = argmin(fitness)

    if off_fit > fitness[worst_idx]:
        population[worst_idx] = offspring
        fitness[worst_idx] = off_fit

    return population, fitness
```

---

## 8. GPU Optimization Strategies

### 8.1 Vectorized Mutation (GPU Kernel)

**MLIR Lowering:**
```mlir
func @gaussian_mutate_batch(
    %population: tensor<NxDxf32>,
    %rate: f32,
    %std: f32,
    %rng_state: !rng.state
) -> tensor<NxDxf32> {
    // Generate random mask [N, D]
    %mask = rng.uniform(%rng_state, shape=[N, D]) : tensor<NxDxf32>
    %should_mutate = arith.cmpf "olt", %mask, %rate : tensor<NxDxf32>

    // Generate Gaussian noise [N, D]
    %noise = rng.normal(%rng_state, mean=0.0, std=%std, shape=[N, D])

    // Apply mutation
    %mutated = arith.select %should_mutate, %noise, 0.0 : tensor<NxDxf32>
    %result = arith.addf %population, %mutated : tensor<NxDxf32>

    return %result : tensor<NxDxf32>
}
```

**GPU Kernel (pseudocode):**
```cuda
__global__ void gaussian_mutate_kernel(
    float* population,  // [N * D]
    float rate,
    float std,
    uint64_t* rng_state,
    int N,
    int D
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N * D) return;

    // Per-thread RNG
    uint64_t local_rng = rng_state[tid];

    if (random_uniform(&local_rng) < rate) {
        population[tid] += random_normal(&local_rng, 0.0, std);
    }

    rng_state[tid] = local_rng;
}
```

---

### 8.2 Batch Fitness Evaluation

**Key insight:** Evaluate all agents in parallel using batch operations.

**Example: Racing AI**
```python
# CPU (slow): evaluate one agent at a time
for agent in population:
    fitness[i] = simulate_race(agent, track)  # 100ms per agent
# Total: 64 agents × 100ms = 6.4 seconds

# GPU (fast): evaluate all agents in parallel
fitness = simulate_race_batch(population, track)  # 100ms total
# Total: 100ms (64× speedup!)
```

**MLIR Batch Evaluation Graph:**
```mlir
func @evaluate_racing_population(
    %nn_weights: tensor<64x600xf32>,  // 64 agents, 600 weights each
    %track: !physics.track
) -> tensor<64xf32> {
    %max_steps = arith.constant 1000 : i64

    // Initialize car states for all agents
    %car_states = physics.init_cars_batch(%track, count=64)

    // Simulation loop
    %final_states = scf.for %step = 0 to %max_steps step 1
        iter_args(%states = %car_states) -> (tensor<64xCarState>) {

        // Batch raycast for all agents
        %sensors = physics.raycast_batch(%states, %track)
        // shape: [64, 5]

        // Batch neural network inference
        %actions = nn.mlp_batch(%nn_weights, %sensors)
        // shape: [64, 3]

        // Batch physics update
        %new_states = physics.car_update_batch(%states, %actions, dt=0.01)

        scf.yield %new_states : tensor<64xCarState>
    }

    // Compute fitness for all agents
    %fitness = physics.compute_fitness_batch(%final_states)
    return %fitness : tensor<64xf32>
}
```

---

## 9. Hyperparameter Tuning Guide

### Recommended Ranges (Neural Network Evolution)

| Parameter | Typical Range | Effect |
|-----------|---------------|--------|
| **Population size** | 32 - 256 | Larger = more diversity, slower |
| **Mutation rate** | 0.01 - 0.15 | Higher = more exploration |
| **Mutation std** | 0.01 - 0.1 | Higher = larger jumps |
| **Crossover rate** | 0.5 - 0.9 | Higher = more recombination |
| **Tournament size** | 2 - 8 | Higher = more selection pressure |
| **Elitism** | 2 - 10 | Higher = preserves best agents |

### Adaptive Schedules

**Mutation decay:**
```python
mutation_rate(gen) = initial_rate * decay^gen
# e.g., 0.15 * 0.99^gen
```

**Adaptive based on fitness variance:**
```python
if std(fitness) < threshold:
    mutation_rate *= 1.2  # Increase exploration
else:
    mutation_rate *= 0.95  # Decrease exploration
```

---

## 10. Example: Complete GA Loop (Racing AI)

```yaml
# Morphogen GA configuration for racing AI

ga_config:
  population_size: 64
  generations: 200

  selection:
    method: tournament
    tournament_size: 4

  crossover:
    method: uniform
    rate: 0.7

  mutation:
    method: gaussian
    rate: 0.12
    std: 0.05
    adaptive: true

  replacement:
    method: generational
    elitism: 4

  evaluation:
    parallel: true
    gpu: true
    early_termination: true  # Stop bad agents early

  logging:
    log_best_fitness: true
    log_mean_fitness: true
    log_diversity: true
    checkpoint_interval: 10
```

**Morphogen Operator Graph:**
```
[Initialize Population] (64 random NNs)
        ↓
  ┌─────────────┐
  │             │
  │  [Evaluate] │ ← Batch NN inference + physics
  │             │
  └──────┬──────┘
         ↓
    [Fitness]
         ↓
  ┌──────────────────┐
  │ [Tournament      │
  │  Selection]      │
  └──────┬───────────┘
         ↓
  ┌──────────────────┐
  │ [Uniform         │
  │  Crossover]      │
  └──────┬───────────┘
         ↓
  ┌──────────────────┐
  │ [Gaussian        │
  │  Mutation]       │
  └──────┬───────────┘
         ↓
  [Generational Replacement]
         ↓
    [Next Gen] ──┐
         ↑       │
         └───────┘ (repeat for 200 generations)
```

---

## 11. Implementation Checklist

### Phase 1: Core Operators
- [ ] `ga.tournament_select`
- [ ] `ga.uniform_crossover`
- [ ] `ga.gaussian_mutate`
- [ ] `ga.evaluate_population` (batch)
- [ ] `ga.generational_replacement`

### Phase 2: Advanced Selection
- [ ] `ga.rank_select`
- [ ] `ga.sus_select`
- [ ] `ga.roulette_select`

### Phase 3: Advanced Crossover
- [ ] `ga.single_point_crossover`
- [ ] `ga.layer_crossover`
- [ ] `ga.blend_crossover`

### Phase 4: Advanced Mutation
- [ ] `ga.uniform_mutate`
- [ ] `ga.adaptive_mutate`
- [ ] `ga.neuron_reinit_mutate`

### Phase 5: Neuroevolution
- [ ] `ga.neat_mutate_topology`
- [ ] `ga.speciate`
- [ ] `ga.fitness_sharing`

### Phase 6: Multi-Objective
- [ ] `ga.pareto_rank`
- [ ] `ga.nsga2_select`
- [ ] `ga.crowding_distance`

### Phase 7: GPU Optimization
- [ ] Vectorized mutation kernels
- [ ] Batch crossover
- [ ] Parallel evaluation
- [ ] Fused GA operations

---

## 12. References & Further Reading

### Classic GA Papers
- Holland (1975) — *Adaptation in Natural and Artificial Systems*
- Goldberg (1989) — *Genetic Algorithms in Search, Optimization, and Machine Learning*

### Neuroevolution
- Stanley & Miikkulainen (2002) — *NEAT: Evolving Neural Networks through Augmenting Topologies*
- Such et al. (2017) — *Deep Neuroevolution: Genetic Algorithms Are a Competitive Alternative*

### Multi-Objective
- Deb et al. (2002) — *NSGA-II: A Fast and Elitist Multiobjective Genetic Algorithm*

### GPU Acceleration
- Pospichal et al. (2010) — *Parallel Genetic Algorithm on GPU*

---

## 13. Morphogen-Specific Advantages

### Why GA in Morphogen is Better

1. **Unified with other domains**
   - GA ops compose with physics, NN, rendering
   - Single computation graph

2. **GPU-accelerated**
   - Batch operations
   - Vectorized mutations
   - Parallel evaluation

3. **Deterministic**
   - Fixed RNG seed → reproducible results
   - Version control for exact experiments

4. **Introspectable**
   - Visualize gene distributions
   - Track diversity metrics
   - Replay evolution

5. **MLIR-lowered**
   - Optimized kernels
   - Kernel fusion
   - Memory optimization

6. **Composable**
   - Mix GA with gradient descent
   - Hybrid evolution strategies
   - Multi-objective + speciation + adaptation

**Morphogen turns genetic algorithms from custom scripts into first-class, optimized, composable operators.**

---

**See also:**
- [Racing AI Pipeline Example](../EXAMPLES/RACING-AI-PIPELINE.md)
- [../specifications/operator-registry.md](../../specifications/operator-registry.md)
- [ADR-002: Cross-Domain Patterns](../../adr/002-cross-domain-architectural-patterns.md)
