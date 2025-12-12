# Racing AI Training Pipeline: Neural Evolution Example

**Status:** Design Document
**Domains:** Physics (Racing), Neural Network, Genetic Algorithm, Rendering, Telemetry, Recording
**Complexity:** Advanced
**Hardware:** GPU-optimized (RTX 3060 12GB target)

---

## 1. Overview

A complete racing AI training pipeline demonstrating how Morphogen's unified operator model enables seamless integration of physics simulation, neural network inference, genetic algorithms, and real-time visualization — all in one composable graph.

**This is one of Morphogen's strongest use cases** because it combines multiple traditionally-separate technologies into a single, clean pipeline:

- **Game engine** → Physics domain
- **Python ML frameworks** → Neural Network domain
- **Custom GA code** → Genetic Algorithm domain
- **Visualization** → Render domain
- **Data collection** → Telemetry & Recording domains

---

## 2. The Challenge: Why Traditional Approaches Struggle

Most racing AI implementations require duct-taping together:

```
Unity/Unreal (physics)
  ↓ IPC/networking ↓
Python (TensorFlow/PyTorch for NN)
  ↓ custom bridge ↓
C++/Python (custom GA code)
  ↓ file I/O ↓
Plotting tools (matplotlib/TensorBoard)
```

**Problems:**
- Multiple languages and runtimes
- Serialization overhead between systems
- Complex state synchronization
- Difficult to parallelize
- Hard to reproduce
- Debugging requires multiple tools

**Morphogen's Solution:**
A single unified computation graph with all domains composed together.

---

## 3. High-Level Vision

### What You Want

1. **Racing simulation** — Cars with physics (arcade or realistic)
2. **Sensors** — Raycasts, LIDAR, speedometer, track curvature
3. **Neural network controller** — MLP/CNN/RNN that outputs steering/throttle/brake
4. **Genetic algorithm** — Evolve better drivers over generations
5. **Optional gradient-based refinement** — Local search for fine-tuning
6. **Visualization** — 2D/3D track view, sensor rays, telemetry graphs
7. **Replay system** — Record and playback best runs
8. **Telemetry** — Track lap times, crashes, fitness metrics
9. **Export** — Save trained networks for deployment

### Morphogen Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP DOMAIN                     │
│  ┌───────────┐    ┌──────────┐    ┌─────────┐             │
│  │ Population│ -> │ Evaluate │ -> │ Fitness │ -> Evolve   │
│  └───────────┘    └──────────┘    └─────────┘             │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────┐
         │      GENETIC ALGORITHM DOMAIN      │
         │  Mutation, Crossover, Selection    │
         └────────────────────────────────────┘
                          │
                          ▼
         ┌────────────────────────────────────┐
         │    AGENT SIMULATION (Parallel)     │
         │  ┌──────────┐                      │
         │  │ Sensors  │ (rays, speed, angle) │
         │  └────┬─────┘                      │
         │       ▼                             │
         │  ┌──────────┐                      │
         │  │ Neural   │ (MLP/CNN/RNN)        │
         │  │ Network  │                      │
         │  └────┬─────┘                      │
         │       ▼                             │
         │  ┌──────────┐                      │
         │  │ Actions  │ (steer,throttle,brake)│
         │  └────┬─────┘                      │
         │       ▼                             │
         │  ┌──────────┐                      │
         │  │ Physics  │ (car dynamics)       │
         │  └────┬─────┘                      │
         │       ▼                             │
         │   New State                         │
         └────────────────────────────────────┘
                          │
                ┌─────────┴─────────┐
                ▼                   ▼
         ┌──────────┐        ┌──────────┐
         │ Render   │        │Telemetry │
         │ Domain   │        │& Record  │
         └──────────┘        └──────────┘
```

---

## 4. Morphogen Configuration Example

### Complete Pipeline Specification (YAML)

```yaml
# racing_ai_pipeline.morph

simulation:
  physics: race_car
  track:
    source: silverstone.yaml
    type: spline_track
  timestep: 0.01
  max_steps: 10000
  parallel_agents: 64

physics:
  model: arcade  # or 'realistic' for Pacejka tire model
  properties:
    mass: 1200.0  # kg
    max_acceleration: 15.0  # m/s²
    max_brake: 20.0  # m/s²
    max_steering_angle: 35.0  # degrees
    drag_coefficient: 0.3
    friction_coefficient: 0.9

agents:
  controller:
    type: neural_network
    architecture: feedforward
    layers: [6, 16, 16, 3]  # input → hidden → hidden → output
    activations:
      hidden: relu
      output: [tanh, sigmoid, sigmoid]  # steering, throttle, brake
    initialization: he_normal

training:
  method: genetic_algorithm
  population: 64
  generations: 200
  mutation:
    rate: 0.12
    type: gaussian
    std: 0.05
  crossover:
    type: uniform
    rate: 0.7
  selection:
    method: tournament
    size: 4
  elitism: 4  # preserve top 4 agents

fitness:
  objectives:
    - name: lap_time
      weight: 1.0
      minimize: true
    - name: track_adherence
      weight: 0.8
      minimize: false
    - name: smoothness
      weight: 0.2
      minimize: false
    - name: crash_penalty
      weight: -2.0
      minimize: true

  calculation:
    lap_time: completed_distance / elapsed_time
    track_adherence: 1.0 - (off_track_time / total_time)
    smoothness: 1.0 / (1.0 + jerk_integral)
    crash_penalty: collision_count

sensors:
  raycasts:
    - angle: -90  # left perpendicular
      range: 50.0
    - angle: -45  # left diagonal
      range: 50.0
    - angle: 0    # forward
      range: 50.0
    - angle: 45   # right diagonal
      range: 50.0
    - angle: 90   # right perpendicular
      range: 50.0

  state:
    - speed  # current velocity magnitude
    - angle_to_track  # deviation from track direction
    - track_curvature  # upcoming turn sharpness

visualization:
  enabled: true
  mode: 2d_topdown  # or '3d_follow'
  fps: 30
  show_sensors: true
  show_best_agent: true
  show_telemetry: true
  record_replay: true

telemetry:
  log_interval: 10  # timesteps
  metrics:
    - position
    - velocity
    - steering_angle
    - throttle
    - brake
    - sensor_readings
    - fitness_components
  export_format: hdf5

output:
  checkpoint_interval: 10  # generations
  save_best_network: true
  export_format: onnx
```

---

## 5. Domain Integration Deep Dive

### 5.1 Physics Domain: Racing Car Model

#### Arcade Physics (Simple & Fast)

**Operators:**
```
car_update(
  state: CarState,
  actions: [steering, throttle, brake],
  dt: float
) -> CarState

inputs:
  - state.position: vec2
  - state.velocity: vec2
  - state.heading: float
  - actions[0]: steering ∈ [-1, 1]
  - actions[1]: throttle ∈ [0, 1]
  - actions[2]: brake ∈ [0, 1]

outputs:
  - new_position: vec2
  - new_velocity: vec2
  - new_heading: float
  - collision: bool
  - off_track: bool

compute:
  # Steering
  heading_rate = steering * max_steering_angle * (speed / max_speed)
  new_heading = heading + heading_rate * dt

  # Acceleration
  accel = throttle * max_acceleration - brake * max_brake - drag * speed²

  # Velocity update
  speed = clamp(speed + accel * dt, 0, max_speed)
  velocity = speed * vec2(cos(heading), sin(heading))

  # Position update
  position += velocity * dt

  # Collision detection
  collision = check_wall_collision(position, track)
  off_track = !is_on_track(position, track)
```

#### Realistic Physics (Advanced)

Uses **Pacejka tire model** with:
- Weight transfer (longitudinal and lateral)
- Slip angles
- Traction circle
- Tire saturation
- Aerodynamic downforce

**Additional state:**
```
- wheel_speeds: [FL, FR, RL, RR]
- slip_angles: [FL, FR, RL, RR]
- weight_distribution: [FL, FR, RL, RR]
- yaw_rate: float
```

### 5.2 Neural Network Domain

**Architecture Flexibility:**

#### Feedforward MLP (Basic)
```yaml
layers: [6, 16, 16, 3]
activations: [relu, relu, [tanh, sigmoid, sigmoid]]
```

#### CNN for Visual Input (Advanced)
```yaml
input: 64x64x1 grayscale top-down view
architecture:
  - conv2d: [32, kernel=3, stride=2]
  - relu
  - conv2d: [64, kernel=3, stride=2]
  - relu
  - flatten
  - dense: [128, activation=relu]
  - dense: [3, activation=[tanh, sigmoid, sigmoid]]
```

#### RNN with Memory (Expert)
```yaml
# For temporal reasoning (upcoming turns)
architecture:
  - input: [6]
  - lstm: [32, return_sequences=false]
  - dense: [16, activation=relu]
  - dense: [3, activation=[tanh, sigmoid, sigmoid]]
```

**Morphogen NN Operators:**
```
mlp_forward(weights, biases, input) -> output
conv2d(weights, input, stride, padding) -> output
lstm_cell(input, hidden, cell, weights) -> (new_hidden, new_cell)
relu(x) -> max(0, x)
tanh(x) -> (e^x - e^-x) / (e^x + e^-x)
sigmoid(x) -> 1 / (1 + e^-x)
```

### 5.3 Genetic Algorithm Domain

**Core Operators:**

#### Mutation
```
gaussian_mutate(
  genome: NNWeights,
  rate: float,
  std: float
) -> NNWeights

for each weight in genome:
  if random() < rate:
    weight += gaussian(0, std)
```

#### Crossover
```
uniform_crossover(
  parent1: NNWeights,
  parent2: NNWeights,
  rate: float
) -> NNWeights

offspring = copy(parent1)
for each weight in offspring:
  if random() < rate:
    offspring.weight = parent2.weight
```

#### Selection
```
tournament_selection(
  population: [Agent],
  fitness: [float],
  tournament_size: int
) -> Agent

tournament = random_sample(population, tournament_size)
return max(tournament, key=fitness)
```

**Advanced Operators:**

- **Neuroevolution** — Add/remove neurons or connections (NEAT-style)
- **Layer-wise crossover** — Swap entire layers between parents
- **Adaptive mutation** — Increase mutation when stuck in local optimum
- **Speciation** — Preserve diversity via niche formation

### 5.4 Telemetry & Recording Domain

**Operators:**

```
log_scalar(name: str, value: float, step: int)
log_vector(name: str, value: [float], step: int)
log_trajectory(positions: [vec2], generation: int, agent_id: int)

save_replay(
  states: [CarState],
  actions: [vec3],
  sensors: [[float]],
  metadata: dict
) -> ReplayHandle

plot_fitness_over_time(generations: [int], best_fitness: [float])
plot_lap_times(agents: [Agent], lap_times: [float])
```

**Storage Format:**

HDF5 structure:
```
racing_ai_training.h5
├── /metadata
│   ├── track_name
│   ├── population_size
│   └── total_generations
├── /generations
│   ├── /gen_000
│   │   ├── fitness: [64]
│   │   ├── lap_times: [64]
│   │   └── trajectories: [64, timesteps, 2]
│   ├── /gen_001
│   └── ...
└── /best_agent
    ├── network_weights
    ├── replay_states
    └── performance_metrics
```

---

## 6. Training Pipeline: Step-by-Step

### 6.1 Initialization

```python
# Pseudocode showing Morphogen graph construction

# 1. Load track
track = load_operator("geometry.spline_track", "silverstone.yaml")

# 2. Initialize population
population = []
for i in range(64):
    genome = random_weights([6, 16, 16, 3], init="he_normal")
    population.append(genome)

# 3. Create physics simulators (parallel)
simulators = [
    create_operator("physics.race_car", track=track)
    for _ in range(64)
]

# 4. Create neural networks (parallel)
controllers = [
    create_operator("nn.mlp", genome=g)
    for g in population
]

# 5. Create sensor arrays
sensors = [
    create_operator("physics.raycast_array", angles=[-90,-45,0,45,90])
    for _ in range(64)
]
```

### 6.2 Evaluation Loop (Per Generation)

```python
def evaluate_generation(population, track):
    fitness_scores = []

    # Parallel evaluation across all agents
    for agent_id, genome in enumerate(population):
        # Reset car to start position
        state = reset_car(track.start_position, track.start_heading)

        trajectory = []
        total_fitness = 0.0

        for step in range(max_steps):
            # 1. Sense environment
            sensor_readings = cast_rays(state, track, angles)
            extra_state = [state.speed, state.angle_to_track, track.curvature]
            nn_input = concatenate(sensor_readings, extra_state)

            # 2. Neural network inference
            actions = mlp_forward(genome, nn_input)
            # actions = [steering, throttle, brake]

            # 3. Physics update
            new_state = car_update(state, actions, dt=0.01)

            # 4. Check termination
            if new_state.crashed or new_state.finished_lap:
                break

            # 5. Log telemetry
            trajectory.append((state.position, actions))

            state = new_state

        # 6. Calculate fitness
        fitness = compute_fitness(
            lap_time=step * 0.01,
            distance=state.distance_traveled,
            crashes=int(state.crashed),
            off_track_time=state.off_track_time,
            smoothness=compute_smoothness(trajectory)
        )

        fitness_scores.append(fitness)

        # 7. Save best agent replay
        if fitness > global_best_fitness:
            save_replay(trajectory, genome, fitness)

    return fitness_scores
```

### 6.3 Evolution Loop

```python
for generation in range(200):
    # 1. Evaluate all agents
    fitness = evaluate_generation(population, track)

    # 2. Log metrics
    log_scalar("best_fitness", max(fitness), generation)
    log_scalar("mean_fitness", mean(fitness), generation)
    log_scalar("std_fitness", std(fitness), generation)

    # 3. Selection & Reproduction
    new_population = []

    # Elitism: keep top performers
    elite_indices = argsort(fitness)[-4:]
    for idx in elite_indices:
        new_population.append(population[idx])

    # Generate offspring
    while len(new_population) < 64:
        # Tournament selection
        parent1 = tournament_select(population, fitness, k=4)
        parent2 = tournament_select(population, fitness, k=4)

        # Crossover
        if random() < 0.7:
            offspring = uniform_crossover(parent1, parent2)
        else:
            offspring = parent1

        # Mutation
        offspring = gaussian_mutate(offspring, rate=0.12, std=0.05)

        new_population.append(offspring)

    population = new_population

    # 4. Checkpoint
    if generation % 10 == 0:
        save_checkpoint(population, fitness, generation)
```

---

## 7. Morphogen Computation Graph Lowering

### High-Level Graph

```
[Track Geometry] ──┐
                   │
[Population] ─────→ [Parallel Evaluator] ──→ [Fitness]
                   │                             │
[NN Weights] ──────┘                             │
                                                 ▼
                                          [GA Operators]
                                          (Select, Cross, Mutate)
                                                 │
                                                 ▼
                                          [New Population]
```

### MLIR Lowering Strategy

#### 1. Vectorization (Parallel Agents)

```mlir
// Instead of sequential loop over 64 agents:
for i in range(64):
    fitness[i] = evaluate(agent[i])

// Vectorize as batch operations:
%sensors = physics.raycast_batch(%states, %track) : tensor<64x5xf32>
%actions = nn.mlp_batch(%weights, %sensors) : tensor<64x3xf32>
%new_states = physics.car_update_batch(%states, %actions) : tensor<64xCarState>
```

#### 2. GPU Kernel Fusion

```mlir
// Fuse sensor reading + NN forward pass into single kernel
gpu.kernel @sense_and_act(%states, %track, %weights) {
    %tid = gpu.thread_id
    %sensors = raycast_kernel(%states[%tid], %track)
    %actions = mlp_kernel(%weights[%tid], %sensors)
    store %actions[%tid]
}
```

#### 3. Memory Optimization

- **Double buffering** for population (old gen / new gen)
- **Persistent NN weights** on GPU across generations
- **Track geometry** loaded once, shared across all agents
- **Replay buffer** written asynchronously to disk

---

## 8. Performance Expectations

### Target Hardware: RTX 3060 (12GB VRAM)

**Baseline Configuration:**
- 64 agents in parallel
- MLP: [6, 16, 16, 3] ≈ 600 weights per agent
- Arcade physics
- 1000 timesteps per evaluation
- 5 raycasts per agent

**Expected Throughput:**

| Component | Time per Generation | Bottleneck |
|-----------|---------------------|------------|
| Physics (64 agents × 1000 steps) | ~50ms | CPU (serial) |
| NN Inference (64 × 1000 forward passes) | ~20ms | GPU (batch) |
| Raycast (64 × 5 × 1000) | ~30ms | GPU (parallel) |
| GA Operations | ~5ms | CPU |
| **Total** | **~105ms** | Physics |

**Training Time Estimates:**

- **Simple track** (arcade physics, basic MLP): 5-20 minutes
- **Complex track** (realistic physics, CNN): 20-120 minutes

**Optimizations:**

1. **GPU-accelerated physics** → 10× speedup
   Expected: ~10ms per generation

2. **No visualization during training** → 30% speedup
   Train headless, visualize best replays later

3. **Adaptive simulation length** → 2× speedup
   Terminate bad agents early if clearly failing

4. **Multi-GPU scaling** → Linear speedup
   32 agents per GPU

**With all optimizations:**
```
Generations per second: ~50-100
200 generations: 2-4 minutes
```

---

## 9. Visualization & Debugging

### 9.1 Real-Time Visualization

**2D Top-Down View:**
```
┌─────────────────────────────────────┐
│  Track (gray)                       │
│                                     │
│    ╭─────╮                         │
│   ╱       ╲    🏎️ <- Best Agent   │
│  │    S    │      (green)          │
│  │    │    │                        │
│   ╲   │   ╱   🏎️🏎️🏎️ <- Others   │
│    ╰──┴──╯      (blue/red)         │
│                                     │
│  Sensor rays shown as lines        │
│  Speed: 145 km/h                   │
│  Lap: 0:42.3                       │
│  Gen: 42/200                       │
└─────────────────────────────────────┘
```

**3D Follow Camera:**
- Chase cam behind best agent
- Sensors visualized as colored rays
- Track boundaries highlighted
- Speedometer & inputs overlay

### 9.2 Telemetry Dashboards

**Fitness Evolution:**
```
Fitness over Generations
 ↑
 │     ╱────
 │    ╱
 │   ╱      <- Mean fitness
 │  │
 │ ╱
 │╱           <- Best fitness
 └─────────────────────────→
  0   50  100  150  200
```

**Action Distribution:**
```
Steering Histogram (Generation 100)
 ↑
 │     ██
 │   ████████
 │ ██████████████
 └─────────────────────────→
  -1.0   0.0   1.0
```

**Lap Time Progression:**
```
Best Lap Time
 ↑
 │ 2:30
 │      ╲
 │ 1:30   ╲___
 │              ╲____
 │ 0:30               ╲╲___
 └─────────────────────────────→
  0   50  100  150  200
```

### 9.3 Debugging Tools

**Morphogen Operators:**
```
debug.pause_on_crash(agent_id)
debug.visualize_nn_activations(agent_id, layer_idx)
debug.plot_sensor_readings(agent_id, timestep)
debug.compare_agents(agent_a, agent_b)
```

**Replay Analysis:**
```
replay = load_replay("best_agent_gen_120.krp")
replay.seek(timestep=500)
replay.show_sensors()
replay.show_nn_weights()
replay.export_video("best_run.mp4")
```

---

## 10. Advanced Extensions

### 10.1 Hybrid Training: GA + Gradient Descent

```yaml
training:
  method: hybrid

  # Phase 1: Genetic algorithm (exploration)
  ga:
    generations: 100
    population: 64
    mutation_rate: 0.12

  # Phase 2: Gradient-based refinement (exploitation)
  gradient:
    method: PPO  # Proximal Policy Optimization
    episodes: 1000
    learning_rate: 0.0003
    batch_size: 256
```

**Why this works:**
- GA finds good behavioral patterns (exploration)
- Gradient descent fine-tunes precise control (exploitation)

### 10.2 Multi-Objective Optimization

```yaml
fitness:
  method: pareto_front
  objectives:
    - lap_time: minimize
    - fuel_consumption: minimize
    - tire_wear: minimize
    - spectator_excitement: maximize
```

Results in a **Pareto frontier** of diverse strategies:
- Fast but wasteful
- Efficient but slow
- Balanced
- Showboating (drifts, but finishes)

### 10.3 Curriculum Learning

```yaml
training:
  curriculum:
    - stage: straight_track
      generations: 20
      track: simple_oval.yaml

    - stage: gentle_turns
      generations: 30
      track: easy_circuit.yaml
      start_from: previous

    - stage: full_circuit
      generations: 150
      track: silverstone.yaml
      start_from: previous
```

### 10.4 Transfer Learning

Train on one track, transfer to another:

```python
# Train on Silverstone
population_silverstone = train(track="silverstone", generations=200)

# Transfer to Monaco (only retrain output layer)
population_monaco = transfer_learn(
    base_population=population_silverstone,
    new_track="monaco",
    freeze_layers=[0, 1],  # freeze hidden layers
    retrain_layers=[2],    # retrain output layer
    generations=50
)
```

### 10.5 Multi-Agent Racing

```yaml
simulation:
  multi_agent: true
  agents_per_race: 4

fitness:
  objectives:
    - finish_position: minimize
    - overtakes: maximize
    - collisions: minimize
```

Agents must learn:
- Spatial awareness (avoid other cars)
- Strategic overtaking
- Defensive driving

---

## 11. Domain Requirements & Operator Catalog

### 11.1 Physics Domain

**New Operators Needed:**

```yaml
# Arcade racing physics
physics.car_update_arcade:
  inputs: [state: CarState, actions: vec3, dt: f32]
  outputs: [new_state: CarState]

# Realistic racing physics (Pacejka)
physics.car_update_realistic:
  inputs: [state: CarStateExtended, actions: vec3, dt: f32]
  outputs: [new_state: CarStateExtended]

# Raycast sensors
physics.raycast:
  inputs: [origin: vec2, direction: vec2, max_distance: f32, scene: Track]
  outputs: [hit_distance: f32, hit_normal: vec2]

physics.raycast_batch:
  inputs: [origins: tensor<Nx2xf32>, directions: tensor<Nx2xf32>, scene: Track]
  outputs: [distances: tensor<Nxf32>]

# Track queries
physics.is_on_track:
  inputs: [position: vec2, track: Track]
  outputs: [on_track: bool]

physics.nearest_track_point:
  inputs: [position: vec2, track: Track]
  outputs: [nearest: vec2, distance: f32, curvature: f32]
```

### 11.2 Neural Network Domain

**Existing + New Operators:**

```yaml
# Standard MLP
nn.mlp_forward:
  inputs: [weights: [tensor], biases: [tensor], input: tensor]
  outputs: [output: tensor]

nn.mlp_batch:
  inputs: [weights: [tensor], biases: [tensor], inputs: tensor<NxDxf32>]
  outputs: [outputs: tensor<NxDxf32>]

# Activations
nn.relu, nn.tanh, nn.sigmoid, nn.elu

# For CNN extensions
nn.conv2d:
  inputs: [weights: tensor, input: tensor, stride: i32, padding: i32]
  outputs: [output: tensor]

# For RNN extensions
nn.lstm_cell:
  inputs: [input: tensor, hidden: tensor, cell: tensor, weights: [tensor]]
  outputs: [new_hidden: tensor, new_cell: tensor]
```

### 11.3 Genetic Algorithm Domain (NEW)

**Core GA Operators:**

```yaml
ga.gaussian_mutate:
  inputs: [genome: [tensor], rate: f32, std: f32, rng: RNGState]
  outputs: [mutated_genome: [tensor], new_rng: RNGState]
  properties:
    deterministic: true  # with fixed seed
    parallel: true

ga.uniform_crossover:
  inputs: [parent1: [tensor], parent2: [tensor], rate: f32, rng: RNGState]
  outputs: [offspring: [tensor], new_rng: RNGState]
  properties:
    deterministic: true
    parallel: true

ga.tournament_select:
  inputs: [population: [[tensor]], fitness: [f32], k: i32, rng: RNGState]
  outputs: [selected: [tensor], new_rng: RNGState]
  properties:
    deterministic: true

ga.evaluate_population:
  inputs: [population: [[tensor]], evaluator: Graph, parallel: bool]
  outputs: [fitness_scores: [f32]]
  properties:
    parallel: true
    gpu_accelerated: true

ga.pareto_rank:
  inputs: [objectives: tensor<NxMxf32>]  # N agents, M objectives
  outputs: [ranks: [i32], crowding_distance: [f32]]
```

**Advanced Neuroevolution:**

```yaml
ga.neat_mutate_topology:
  inputs: [genome: NeuralGraph, innovation_db: InnovationDB]
  outputs: [new_genome: NeuralGraph]
  mutations:
    - add_node
    - add_connection
    - change_weight

ga.speciate:
  inputs: [population: [Genome], compatibility_threshold: f32]
  outputs: [species: [[Genome]]]
```

### 11.4 Telemetry Domain

```yaml
telemetry.log_scalar:
  inputs: [name: str, value: f32, step: i64]
  outputs: []
  side_effects: [write_to_buffer]

telemetry.log_vector:
  inputs: [name: str, values: [f32], step: i64]
  outputs: []

telemetry.log_trajectory:
  inputs: [positions: tensor<Tx2xf32>, metadata: dict]
  outputs: [trajectory_id: i64]

telemetry.plot:
  inputs: [x: [f32], y: [f32], title: str, labels: [str]]
  outputs: [plot_handle: PlotHandle]

telemetry.export_hdf5:
  inputs: [data: dict, filename: str]
  outputs: [success: bool]
```

### 11.5 Recording Domain

```yaml
recording.start_replay:
  inputs: [capacity: i64]
  outputs: [recorder: RecorderHandle]

recording.record_frame:
  inputs: [recorder: RecorderHandle, state: dict]
  outputs: []

recording.save_replay:
  inputs: [recorder: RecorderHandle, filename: str]
  outputs: [success: bool]

recording.load_replay:
  inputs: [filename: str]
  outputs: [replay: ReplayHandle]

recording.replay_seek:
  inputs: [replay: ReplayHandle, frame: i64]
  outputs: [state: dict]

recording.replay_export_video:
  inputs: [replay: ReplayHandle, renderer: Graph, output: str]
  outputs: [success: bool]
```

---

## 12. Implementation Roadmap

### Phase 1: Core Racing Simulation (Week 1-2)

- [ ] Implement arcade physics operators (`car_update_arcade`)
- [ ] Implement spline-based track geometry
- [ ] Implement raycast sensors
- [ ] Basic 2D visualization
- [ ] Manual control test (keyboard input)

**Deliverable:** Drivable car on a simple track

### Phase 2: Neural Network Integration (Week 2-3)

- [ ] Implement batch MLP operators
- [ ] Weight initialization (He, Xavier)
- [ ] Sensor → NN → Action pipeline
- [ ] Random agent baseline

**Deliverable:** Random NN-controlled agent

### Phase 3: Genetic Algorithm (Week 3-4)

- [ ] Implement GA operators (mutate, crossover, select)
- [ ] Population evaluation loop
- [ ] Fitness calculation
- [ ] Elitism & reproduction

**Deliverable:** Evolving population that improves over time

### Phase 4: Telemetry & Debugging (Week 4-5)

- [ ] Implement telemetry operators
- [ ] Fitness tracking over generations
- [ ] Lap time logging
- [ ] Trajectory visualization
- [ ] Replay system

**Deliverable:** Full observability of training process

### Phase 5: Optimization (Week 5-6)

- [ ] GPU-accelerated physics (if feasible)
- [ ] Vectorized operations
- [ ] Memory optimization
- [ ] Adaptive simulation length
- [ ] Parallel evaluation

**Deliverable:** 10-100× speedup

### Phase 6: Advanced Features (Week 6+)

- [ ] Realistic physics (Pacejka model)
- [ ] Hybrid GA + gradient descent
- [ ] Multi-objective optimization
- [ ] Curriculum learning
- [ ] Transfer learning
- [ ] Multi-agent racing

**Deliverable:** Research-grade racing AI system

---

## 13. Validation & Testing

### 13.1 Physics Validation

**Test: Conservation of Energy**
```python
def test_energy_conservation():
    state = CarState(position=(0,0), velocity=(10,0), mass=1200)

    # No input, just friction
    for _ in range(1000):
        state = car_update(state, actions=[0, 0, 0], dt=0.01)

    # Energy should decay monotonically
    assert state.speed < 10.0
    assert state.speed >= 0.0
```

**Test: Steering Behavior**
```python
def test_steering():
    state = CarState(position=(0,0), velocity=(10,0), heading=0)

    # Full left steering for 1 second
    for _ in range(100):
        state = car_update(state, actions=[-1, 1, 0], dt=0.01)

    # Should have turned left (heading < 0)
    assert state.heading < -0.1
```

### 13.2 Neural Network Validation

**Test: Gradient Flow**
```python
def test_mlp_forward():
    weights = initialize_weights([6, 16, 16, 3], method="he")
    input = torch.randn(6)

    output = mlp_forward(weights, input)

    assert output.shape == (3,)
    assert torch.isfinite(output).all()
```

### 13.3 Genetic Algorithm Validation

**Test: Fitness Improvement**
```python
def test_ga_improvement():
    population = initialize_population(size=64)

    fitness_gen_0 = evaluate(population)

    for gen in range(50):
        population = evolve(population)

    fitness_gen_50 = evaluate(population)

    # Best fitness should improve
    assert max(fitness_gen_50) > max(fitness_gen_0)
    # Mean fitness should improve
    assert mean(fitness_gen_50) > mean(fitness_gen_0)
```

**Test: Elitism Preservation**
```python
def test_elitism():
    population = initialize_population(size=64)
    fitness = evaluate(population)

    best_agent = population[argmax(fitness)]

    new_population = evolve(population, elitism=4)

    # Best agent should still be in population
    assert best_agent in new_population
```

### 13.4 End-to-End Integration Test

```python
def test_full_training_pipeline():
    config = load_config("racing_ai_pipeline.yaml")

    # Run for 10 generations
    result = train(config, generations=10)

    assert result.best_fitness > result.initial_fitness
    assert result.generations_completed == 10
    assert result.best_agent is not None

    # Test best agent
    lap_time = evaluate_single_agent(result.best_agent, config.track)
    assert lap_time > 0  # Completed at least part of the lap
    assert lap_time < float('inf')  # Didn't crash immediately
```

---

## 14. Comparison: Morphogen vs Traditional Approaches

| Aspect | Traditional (Unity + Python) | Morphogen |
|--------|------------------------------|-------|
| **Languages** | C#, Python, YAML | Morphogen YAML/Python |
| **Architecture** | Game engine + ML framework + custom GA | Unified computation graph |
| **IPC Overhead** | High (serialization, networking) | Zero (shared memory graph) |
| **Parallelization** | Manual (complex) | Automatic (graph analysis) |
| **GPU Utilization** | Partial (only NN) | Full (physics, NN, GA) |
| **Debugging** | Multiple tools | Single introspection system |
| **Reproducibility** | Difficult (random seeds across systems) | Perfect (deterministic graph) |
| **Iteration Speed** | Slow (recompile, restart) | Fast (hot-reload graph) |
| **State Management** | Complex (sync issues) | Automatic (graph dependencies) |
| **Visualization** | Tightly coupled | Decoupled (optional render graph) |
| **Export** | Custom code | Built-in (ONNX, HDF5) |

---

## 15. Why This Showcases Morphogen's Strengths

### 1. **Multi-Domain Composition**

This example requires **6+ domains**:
- Physics (car dynamics, collision)
- Neural Networks (inference)
- Genetic Algorithms (evolution)
- Rendering (visualization)
- Telemetry (logging)
- Recording (replay)

Traditional approaches require duct-taping these together. **Morphogen composes them natively.**

### 2. **Unified Operator Model**

Every component is an operator:
- `car_update` is an operator
- `mlp_forward` is an operator
- `gaussian_mutate` is an operator
- `raycast` is an operator

This means:
- **Same optimization pipeline** (MLIR lowering)
- **Same debugging tools**
- **Same serialization format**
- **Same parallelization strategy**

### 3. **Determinism & Reproducibility**

Morphogen graphs are **perfectly deterministic**:
- Fixed RNG seed → identical results
- Version control for exact graph structure
- Snapshot entire training state
- Resume from any generation

This is critical for:
- Scientific reproducibility
- A/B testing hyperparameters
- Debugging rare failure modes

### 4. **GPU-Accelerated Everything**

Not just neural networks — **all operators can run on GPU**:
- Physics (vectorized car updates)
- Raycasts (parallel BVH traversal)
- GA operations (batch mutations)
- Fitness calculations (reduce operations)

Result: **10-100× faster than CPU-only approaches.**

### 5. **Clean Architecture**

The entire system is specified in ~200 lines of YAML. Compare to:
- Unity project: thousands of files
- Python ML repo: complex package dependencies
- Custom GA code: brittle glue code

**Morphogen is maintainable, readable, and modifiable.**

---

## 16. Future Directions

### 16.1 Real-World Deployment

Train in Morphogen, deploy to:
- **Embedded systems** (export to C/ONNX)
- **Web browsers** (export to WebAssembly)
- **Mobile devices** (export to CoreML/TFLite)
- **Game engines** (export to Unity/Unreal via plugins)

### 16.2 Sim-to-Real Transfer

Use Morphogen to:
1. Train in idealized simulation
2. Add noise/disturbances progressively (domain randomization)
3. Fine-tune with real-world data (if available)
4. Deploy to physical RC car or autonomous vehicle

### 16.3 Research Applications

- **Benchmarking evolutionary algorithms** (compare GA, CMA-ES, NEAT)
- **Neural architecture search** (evolve network topology)
- **Interpretability** (visualize learned behaviors)
- **Transfer learning studies** (track → track generalization)

### 16.4 Community Contributions

Open opportunities:
- New track designs (procedural generation)
- Alternative physics models (bike, boat, drone)
- Different NN architectures (attention, transformers)
- Multi-agent competitive racing
- Human-vs-AI racing modes

---

## 17. Conclusion

**The racing AI training pipeline is one of Morphogen's best examples** because it demonstrates:

✅ **Multi-domain integration** — Physics + NN + GA + Rendering + Telemetry
✅ **GPU acceleration** — Parallel evaluation, batch inference
✅ **Clean architecture** — Single YAML spec, unified operators
✅ **Determinism** — Perfect reproducibility
✅ **Performance** — 10-100× faster than traditional approaches
✅ **Extensibility** — Easy to add new features (multi-agent, hybrid training)
✅ **Real-world applicability** — Export to deployment targets

**This is exactly the kind of "multi-domain composable simulation" Morphogen was built for.**

If someone asks: *"Can Morphogen do X?"*

Point them to this example and say:

> **"If Morphogen can train a racing AI by composing physics, neural networks, and genetic algorithms into a single graph that runs 100× faster than Unity + Python... yes, it can do X."**

---

## 18. Getting Started

### Quick Start

```bash
# Clone Morphogen repo
git clone https://github.com/your-org/morphogen.git
cd kairo

# Install dependencies
pip install -r requirements.txt

# Run racing AI example
morphogen run examples/racing_ai/simple_track.yaml

# View training progress
kairo tensorboard --logdir=runs/racing_ai_001
```

### Modify the Example

1. **Change track:** Edit `track.yaml` to create new circuits
2. **Tune hyperparameters:** Adjust `mutation_rate`, `population`, `layers`
3. **Add sensors:** Extend `sensors` array with new raycasts
4. **Change fitness:** Modify `fitness.objectives` weights
5. **Visualize:** Toggle `visualization.mode` between 2d/3d

### Learn More

- [specifications/physics-domains.md](../specifications/physics-domains.md) — Physics operators
- [specifications/operator-registry.md](../specifications/operator-registry.md) — Operator metadata
- [ADR-002: Cross-Domain Patterns](../adr/002-cross-domain-architectural-patterns.md) — Architecture
- [architecture/domain-architecture.md](../architecture/domain-architecture.md) — Full domain vision

---

**Morphogen is not a library. Morphogen is a platform.**

🏁
