# Emergence Domain: Complete Operator Catalog

**Version:** 1.0
**Status:** Proposed
**Last Updated:** 2025-11-15
**Related:** ../specifications/emergence.md, ADR-004

---

## Overview

This document provides a **complete, implementation-ready catalog** of all operators in Morphogen's EmergenceDomain. For each operator, we specify:

- **Signature** — Type-safe function signature
- **Parameters** — All tunable parameters with defaults and ranges
- **Semantics** — What the operator does algorithmically
- **Use Cases** — Where this operator should be used
- **Examples** — Concrete Morphogen code snippets
- **Cross-Domain Integration** — How this connects to other domains
- **Implementation Notes** — GPU/CPU considerations, performance tips

This catalog is organized by sub-domain:

1. **Cellular Automata (CA)** — 9 operators
2. **Agent-Based Models (ABM)** — 12 operators
3. **Reaction-Diffusion (RD)** — 5 operators
4. **L-Systems** — 6 operators
5. **Swarm Intelligence** — 5 operators
6. **Utilities** — 8 operators

**Total: 45 operators**

---

# 1. Cellular Automata (CA)

## 1.1 ca.create

**Create CA grid**

```morphogen
ca.create(
    width: i32,
    height: i32,
    states: i32 = 2,
    neighborhood: NeighborhoodType = "Moore",
    boundary: BoundaryType = "Periodic",
    init: InitType = "random",
    seed: Option<u64> = None
) -> CAGrid2D<i32>
```

**Parameters:**
- `width`, `height` — Grid dimensions
- `states` — Number of cell states (2 for binary, 3+ for multi-state)
- `neighborhood` — `"Moore"` (8-neighbors), `"VonNeumann"` (4-neighbors), or custom kernel
- `boundary` — `"Periodic"` (torus), `"Fixed"` (constant boundary), `"Mirror"` (reflection)
- `init` — `"random"`, `"empty"` (all zeros), `"seed"` (user-provided pattern)
- `seed` — RNG seed (required for determinism if init="random")

**Use Cases:**
- Initialize CA for evolution
- Generate random starting conditions
- Load predefined patterns (gliders, spaceships, etc.)

**Example:**
```morphogen
// Game of Life grid
let grid = ca.create(
    width = 512,
    height = 512,
    states = 2,
    neighborhood = "Moore",
    boundary = "Periodic",
    init = "random",
    seed = 42
)
```

**Determinism:** Strict (with fixed seed)

---

## 1.2 ca.step

**Evolve CA one generation**

```morphogen
ca.step(
    grid: CAGrid2D<T>,
    rule: CARule
) -> CAGrid2D<T>
```

**Parameters:**
- `grid` — Current grid state
- `rule` — Transition rule (see `ca.rule_preset` or custom)

**Semantics:**
- For each cell, count live neighbors
- Apply rule to determine next state
- Return new grid (functional, no mutation)

**Use Cases:**
- Step-by-step CA evolution
- Animation loops
- Pattern analysis

**Example:**
```morphogen
let life_rule = ca.rule_preset("life")
let evolved = ca.step(grid, rule=life_rule)
```

**Determinism:** Strict

**Performance:**
- GPU: Fully parallel (each cell independent)
- Tiling: 16×16 blocks
- Memory: Double-buffering (read old grid, write new grid)

---

## 1.3 ca.step_n

**Evolve CA n generations**

```morphogen
ca.step_n(
    grid: CAGrid2D<T>,
    rule: CARule,
    steps: i32
) -> CAGrid2D<T>
```

**Use Cases:**
- Fast-forward evolution
- Generate final patterns without intermediate states

**Example:**
```morphogen
let final = ca.step_n(grid, rule=life_rule, steps=1000)
```

**Determinism:** Strict

---

## 1.4 ca.rule_preset

**Create common CA rules**

```morphogen
ca.rule_preset(name: String) -> CARule
```

**Presets:**

| Name | Description | Parameters |
|------|-------------|------------|
| `"life"` | Conway's Game of Life | B3/S23 (birth on 3, survive on 2-3) |
| `"highlife"` | HighLife variant | B36/S23 |
| `"day_and_night"` | Day & Night | B3678/S34678 |
| `"brians_brain"` | Brian's Brain (3-state) | Cyclic: dead → firing → refractory → dead |
| `"wireworld"` | Wireworld (4-state electronics) | Head → Tail → Wire (if 1-2 heads nearby) |
| `"rule30"` | Wolfram Rule 30 (1D) | Chaotic, random-looking output |
| `"rule110"` | Wolfram Rule 110 (1D) | Turing-complete |
| `"lenia_preset"` | Lenia continuous CA | Gaussian kernel + growth function |

**Example:**
```morphogen
// Wireworld electronic circuit
let circuit_rule = ca.rule_preset("wireworld")
let circuit = ca.create(128, 128, states=4, init="wireworld_circuit")
let evolved = ca.step(circuit, rule=circuit_rule)
```

---

## 1.5 ca.lenia

**Continuous cellular automaton (Lenia)**

```morphogen
ca.lenia(
    state: Field2D<f32>,
    kernel: String = "gaussian",
    growth_func: String = "gaussian",
    mu: f32 = 0.15,
    sigma: f32 = 0.015,
    r: f32 = 13.0,
    dt: f32 = 0.1
) -> Field2D<f32>
```

**Parameters:**
- `kernel` — Convolution kernel type (`"gaussian"`, `"polynomial"`)
- `growth_func` — Growth function (`"gaussian"`, `"step"`, `"polynomial"`)
- `mu` — Growth center (peak of growth function)
- `sigma` — Growth width (standard deviation)
- `r` — Kernel radius
- `dt` — Time step

**Use Cases:**
- Smooth, life-like organisms
- Organic growth patterns
- Continuous evolution (smoother than discrete CA)

**Example:**
```morphogen
let state = field.create(256, 256, 0.0)
state = field.set_region(state, center=(128, 128), radius=10, value=1.0)

for i in 0..1000 {
    state = ca.lenia(state, mu=0.15, sigma=0.015, r=13.0, dt=0.1)
}

out visual = visual.colorize(state, palette="viridis")
```

**Determinism:** Repro

**Performance:**
- Convolution-heavy (use FFT for large kernels)
- GPU: Fast for small kernels (< 64×64), FFT for large

---

## 1.6 ca.to_field

**Convert CA grid to continuous field**

```morphogen
ca.to_field(grid: CAGrid2D<i32>) -> Field2D<f32>
```

**Use Cases:**
- Visualization (render CA as image)
- Geometry generation (extrude CA pattern)
- Cross-domain coupling (CA → RD, CA → Physics)

**Example:**
```morphogen
let ca = ca.step_n(grid, rule=life_rule, steps=100)
let field = ca.to_field(ca)
let heightmap = field * 10.0  // Scale for extrusion
let geometry = geom.extrude(heightmap, height=heightmap)
```

---

## 1.7 ca.from_field

**Convert field to CA grid (thresholding)**

```morphogen
ca.from_field(
    field: Field2D<f32>,
    threshold: f32 = 0.5,
    states: i32 = 2
) -> CAGrid2D<i32>
```

**Use Cases:**
- Initialize CA from image/texture
- Couple RD → CA

---

## 1.8 ca.diffuse

**Add diffusion to continuous CA**

```morphogen
ca.diffuse(
    grid: CAGrid2D<f32>,
    coeff: f32,
    dt: f32
) -> CAGrid2D<f32>
```

**Use Cases:**
- Smooth CA states
- Hybrid CA-RD systems

**Example:**
```morphogen
state = ca.lenia(state, ...)
state = ca.diffuse(state, coeff=0.1, dt=0.01)
```

---

## 1.9 ca.pattern_detect

**Detect known patterns (gliders, oscillators)**

```morphogen
ca.pattern_detect(
    grid: CAGrid2D<i32>,
    pattern: String
) -> Array<Vec2<i32>>
```

**Patterns:**
- `"glider"`, `"lwss"` (lightweight spaceship), `"blinker"`, `"toad"`, etc.

**Use Cases:**
- Pattern recognition
- Testing CA evolution correctness

---

# 2. Agent-Based Models (ABM)

## 2.1 agent.create

**Create agent population**

```morphogen
agent.create<A>(
    count: i32,
    bounds: BoundingBox,
    distribution: String = "random",
    seed: u64,
    init_fn: Option<Fn(i32) -> Agent<A>> = None
) -> Agents<A>
```

**Parameters:**
- `count` — Number of agents
- `bounds` — Spatial bounds (2D or 3D)
- `distribution` — `"random"`, `"grid"`, `"circle"`, `"poisson_disk"`
- `seed` — RNG seed (required)
- `init_fn` — Custom initialization function

**Example:**
```morphogen
let boids = agent.create(
    count = 500,
    bounds = box(100m, 100m),
    distribution = "random",
    seed = 42
)
```

**Determinism:** Strict (with fixed seed)

---

## 2.2 agent.boids

**Flocking behavior (Reynolds boids)**

```morphogen
agent.boids(
    agents: Agents<BoidState>,
    separation: f32 = 1.0,
    alignment: f32 = 1.0,
    cohesion: f32 = 1.0,
    perception_radius: f32<m> = 10m,
    max_speed: f32<m/s> = 5m/s,
    max_force: f32<N> = 1N,
    dt: Time = 0.016s
) -> Agents<BoidState>
```

**Parameters:**
- `separation` — Avoidance weight
- `alignment` — Velocity matching weight
- `cohesion` — Centering weight
- `perception_radius` — Neighbor detection radius
- `max_speed` — Speed limit
- `max_force` — Steering force limit
- `dt` — Time step

**Semantics:**
1. For each agent, find neighbors within perception radius
2. Compute separation force (avoid crowding)
3. Compute alignment force (match velocity)
4. Compute cohesion force (move toward center)
5. Sum forces, clamp, integrate

**Use Cases:**
- Flocking animation (birds, fish)
- Crowd simulation
- Swarm robotics
- Visual effects

**Example:**
```morphogen
scene Flocking {
    let boids = agent.create(count=1000, bounds=box(200m, 200m), seed=42)

    step(dt: Time) {
        boids = agent.boids(
            boids,
            separation = 1.5,
            alignment = 1.0,
            cohesion = 0.8,
            perception_radius = 15m,
            max_speed = 10m/s,
            dt = dt
        )

        out visual = visual.render_agents(boids, color="cyan")
    }
}
```

**Determinism:** Strict (with stable neighbor ordering)

**Performance:**
- Spatial indexing: Grid or k-d tree (O(N log N))
- GPU: Parallel force computation + reduction
- Memory: Minimize neighbor list allocations

---

## 2.3 agent.vicsek

**Vicsek model (active matter physics)**

```morphogen
agent.vicsek(
    agents: Agents<VicsekState>,
    speed: f32<m/s>,
    noise: f32,
    radius: f32<m>,
    dt: Time
) -> Agents<VicsekState>
```

**Parameters:**
- `speed` — Constant agent speed
- `noise` — Angular noise amplitude (η in literature)
- `radius` — Interaction radius
- `dt` — Time step

**Semantics:**
1. Each agent moves at constant speed
2. Align with average direction of neighbors
3. Add angular noise

**Use Cases:**
- Active matter physics research
- Phase transitions in collective motion
- Simplified flocking (no cohesion/separation)

**Example:**
```morphogen
let agents = agent.create(count=5000, bounds=box(100m, 100m), seed=42)

agents = agent.vicsek(
    agents,
    speed = 1m/s,
    noise = 0.1,
    radius = 1m,
    dt = 0.1s
)
```

**Determinism:** Repro (noise is deterministic with seed)

---

## 2.4 agent.schelling

**Schelling segregation model**

```morphogen
agent.schelling(
    agents: Agents<SchellingState>,
    threshold: f32 = 0.3,
    types: i32 = 2,
    grid_size: (i32, i32),
    dt: Time
) -> Agents<SchellingState>
```

**Parameters:**
- `threshold` — Fraction of similar neighbors required for happiness
- `types` — Number of agent types
- `grid_size` — Discrete grid for agent placement

**Semantics:**
1. Check each agent's neighborhood
2. If fraction of similar neighbors < threshold, mark as unhappy
3. Move unhappy agents to random empty cells

**Use Cases:**
- Social dynamics modeling
- Segregation research
- Educational demos (emergence of segregation from mild preferences)

**Example:**
```morphogen
let agents = agent.create(count=1000, bounds=grid(50, 50), seed=42)

agents = agent.schelling(
    agents,
    threshold = 0.3,  // Want ≥30% similar neighbors
    types = 2,
    grid_size = (50, 50)
)
```

---

## 2.5 agent.predator_prey

**Predator-prey ecology (Lotka-Volterra ABM)**

```morphogen
agent.predator_prey(
    prey: Agents<PreyState>,
    predators: Agents<PredatorState>,
    prey_birth_rate: f32 = 0.1,
    predation_rate: f32 = 0.01,
    predator_death_rate: f32 = 0.05,
    predator_efficiency: f32 = 0.1,
    dt: Time
) -> (Agents<PreyState>, Agents<PredatorState>)
```

**Semantics:**
1. Prey reproduce at `prey_birth_rate`
2. Predators hunt prey within radius
3. Successful hunts → prey removed, predator gains energy
4. Predators die if energy depletes

**Use Cases:**
- Ecology modeling
- Population dynamics
- Educational simulations

**Example:**
```morphogen
let prey = agent.create(count=500, bounds=box(100m, 100m), seed=42)
let predators = agent.create(count=50, bounds=box(100m, 100m), seed=43)

(prey, predators) = agent.predator_prey(
    prey, predators,
    prey_birth_rate = 0.05,
    predation_rate = 0.02,
    predator_death_rate = 0.01,
    dt = 0.1s
)
```

---

## 2.6 agent.crowd_dynamics

**Crowd simulation (social forces model)**

```morphogen
agent.crowd_dynamics(
    agents: Agents<CrowdState>,
    goals: Array<Vec2<m>>,
    obstacles: Array<Obstacle>,
    personal_space: f32<m> = 0.5m,
    max_speed: f32<m/s> = 1.4m/s,
    dt: Time
) -> Agents<CrowdState>
```

**Use Cases:**
- Pedestrian simulation
- Evacuation modeling
- Architecture planning

---

## 2.7 agent.integrate

**Integrate agent dynamics (generic ODE/SDE solver)**

```morphogen
agent.integrate(
    agents: Agents<A>,
    forces: Fn(Agent<A>, neighbors) -> Vec2<N>,
    method: String = "rk4",
    dt: Time
) -> Agents<A>
```

**Methods:**
- `"euler"`, `"rk2"`, `"rk4"`, `"verlet"`

**Use Cases:**
- Custom agent dynamics
- Physics-based agents

---

## 2.8 agent.to_field

**Rasterize agents to field (particle-in-cell)**

```morphogen
agent.to_field<A>(
    agents: Agents<A>,
    property: String,
    resolution: (i32, i32),
    interpolation: String = "linear"
) -> Field2D<f32>
```

**Properties:**
- `"density"` — Agent count per cell
- `"velocity_x"`, `"velocity_y"` — Average velocity
- Custom attributes

**Use Cases:**
- Visualize agent density
- Couple agents → fields
- Sonification (density → audio)

**Example:**
```morphogen
let density = agent.to_field(boids, property="density", resolution=(512, 512))
out visual = visual.colorize(density, palette="inferno")
```

---

## 2.9 agent.from_field

**Sample field at agent positions**

```morphogen
agent.from_field<A>(
    agents: Agents<A>,
    field: Field2D<f32>,
    target_attr: String
) -> Agents<A>
```

**Use Cases:**
- Field → agents (e.g., temperature field affects agent speed)
- Couple RD → agents
- Environmental influences

**Example:**
```morphogen
// Temperature field affects agent speed
let temperature = field.laplacian(heat_source, dt=0.01)
agents = agent.from_field(agents, field=temperature, target_attr="speed")
```

---

## 2.10 agent.neighbors

**Find neighbors within radius**

```morphogen
agent.neighbors<A>(
    agents: Agents<A>,
    radius: f32<m>,
    method: String = "grid"
) -> Array<Array<AgentRef>>
```

**Methods:**
- `"brute"` — O(N²) exhaustive search
- `"grid"` — Grid-based spatial index
- `"kdtree"` — K-d tree

**Use Cases:**
- Custom interaction rules
- Debugging neighbor detection

---

## 2.11 agent.remove

**Remove agents by predicate**

```morphogen
agent.remove<A>(
    agents: Agents<A>,
    predicate: Fn(Agent<A>) -> bool
) -> Agents<A>
```

**Example:**
```morphogen
// Remove agents outside bounds
agents = agent.remove(agents, |a| a.position.x > 100m)
```

---

## 2.12 agent.spawn

**Spawn new agents**

```morphogen
agent.spawn<A>(
    agents: Agents<A>,
    new_agents: Array<Agent<A>>
) -> Agents<A>
```

**Use Cases:**
- Agent reproduction
- Dynamic agent creation

---

# 3. Reaction-Diffusion Systems

## 3.1 rd.gray_scott

**Gray-Scott reaction-diffusion**

```morphogen
rd.gray_scott(
    u: Field2D<f32>,
    v: Field2D<f32>,
    Du: f32 = 0.16,
    Dv: f32 = 0.08,
    f: f32,
    k: f32,
    dt: f32 = 1.0
) -> (Field2D<f32>, Field2D<f32>)
```

**Parameters:**
- `u` — Activator concentration
- `v` — Inhibitor concentration
- `Du`, `Dv` — Diffusion coefficients
- `f` — Feed rate (adds u, removes v)
- `k` — Kill rate (converts v → nothing)
- `dt` — Time step

**Equations:**
```
∂u/∂t = Du ∇²u - uv² + f(1 - u)
∂v/∂t = Dv ∇²v + uv² - (f + k)v
```

**Parameter Regimes:**
| f | k | Pattern |
|---|---|---------|
| 0.04 | 0.06 | Spots |
| 0.02 | 0.05 | Stripes |
| 0.06 | 0.06 | Waves |

**Use Cases:**
- Biological patterns (spots, stripes)
- Texture generation
- Procedural design

**Example:**
```morphogen
let u = field.create(256, 256, 1.0)
let v = field.create(256, 256, 0.0)
v = field.set_region(v, center=(128, 128), radius=10, value=1.0)

for i in 0..5000 {
    (u, v) = rd.gray_scott(u, v, f=0.055, k=0.062, dt=1.0)
}

out visual = visual.colorize(v, palette="magma")
```

**Determinism:** Repro

**Performance:**
- Laplacian: 5-point stencil (2D) or 7-point (3D)
- Integration: Forward Euler (explicit), or implicit (future)
- GPU: Tile + shared memory for stencil

---

## 3.2 rd.turing

**General Turing pattern system**

```morphogen
rd.turing(
    field: Field2D<f32>,
    a: f32,
    b: f32,
    D1: f32,
    D2: f32,
    dt: f32
) -> Field2D<f32>
```

**Equations:**
```
∂u/∂t = D1 ∇²u + a*u - b*v
∂v/∂t = D2 ∇²v + c*u - d*v
```

**Use Cases:**
- Generic pattern formation
- Custom RD systems

---

## 3.3 rd.belousov_zhabotinsky

**Belousov-Zhabotinsky reaction**

```morphogen
rd.belousov_zhabotinsky(
    u: Field2D<f32>,
    v: Field2D<f32>,
    w: Field2D<f32>,
    params: BZParams,
    dt: f32
) -> (Field2D<f32>, Field2D<f32>, Field2D<f32>)
```

**Use Cases:**
- Chemical wave simulation
- Spiral pattern formation

---

## 3.4 rd.to_geometry

**Convert RD pattern to geometry (marching cubes)**

```morphogen
rd.to_geometry(
    field: Field2D<f32>,
    threshold: f32 = 0.5
) -> Mesh
```

**Use Cases:**
- RD → 3D surface
- Procedural modeling

**Example:**
```morphogen
(u, v) = rd.gray_scott(u, v, f=0.04, k=0.06)
let mesh = rd.to_geometry(v, threshold=0.5)
let surface = geom.from_mesh(mesh)
```

---

## 3.5 rd.init

**Initialize RD system**

```morphogen
rd.init(
    width: i32,
    height: i32,
    u_init: f32 = 1.0,
    v_init: f32 = 0.0,
    perturbation: Option<Perturbation> = None
) -> (Field2D<f32>, Field2D<f32>)
```

**Perturbations:**
- `"circle"` — Circular perturbation
- `"random"` — Random noise
- `"stripe"` — Vertical stripe

---

# 4. L-Systems

## 4.1 lsys.create

**Create L-system**

```morphogen
lsys.create(
    axiom: String,
    rules: Map<Char, String>,
    angle: f32<deg> = 25deg,
    step_size: f32<m> = 1m
) -> LSystem
```

**Example:**
```morphogen
let koch = lsys.create(
    axiom = "F",
    rules = {'F': "F+F-F-F+F"},
    angle = 90deg,
    step_size = 1m
)
```

---

## 4.2 lsys.evolve

**Evolve L-system string n generations**

```morphogen
lsys.evolve(
    lsys: LSystem,
    iterations: i32
) -> LSystemState
```

**Example:**
```morphogen
let evolved = lsys.evolve(koch, iterations=4)
```

**Determinism:** Strict

---

## 4.3 lsys.to_geometry

**Interpret L-system as turtle graphics → 3D geometry**

```morphogen
lsys.to_geometry(
    state: LSystemState,
    angle: f32<deg>,
    step_size: f32<m>,
    diameter: f32<m> = 0.1m,
    diameter_scale: f32 = 0.9
) -> Mesh
```

**Turtle Commands:**
- `F` — Forward (draw line)
- `+` — Rotate right
- `-` — Rotate left
- `[` — Push state (stack)
- `]` — Pop state

**Example:**
```morphogen
let tree = lsys.create(
    axiom = "F",
    rules = {'F': "FF+[+F-F-F]-[-F+F+F]"},
    angle = 25deg
)

let evolved = lsys.evolve(tree, iterations=5)
let mesh = lsys.to_geometry(
    evolved,
    angle = 25deg,
    step_size = 0.5m,
    diameter = 0.1m,
    diameter_scale = 0.7  // Branches taper
)
```

**Determinism:** Strict

---

## 4.4 lsys.preset

**L-system presets**

```morphogen
lsys.preset(name: String) -> LSystem
```

**Presets:**
- `"fractal_tree"`
- `"koch_curve"`
- `"sierpinski_triangle"`
- `"dragon_curve"`
- `"hilbert_curve"`

---

## 4.5 lsys.parametric

**Parametric L-system (parameters in rules)**

```morphogen
lsys.parametric(
    axiom: String,
    rules: Map<Char, Fn(params) -> String>,
    iterations: i32
) -> LSystemState
```

**Example:**
```morphogen
// Branch length decreases with depth
let tree = lsys.parametric(
    axiom = "F(1.0)",
    rules = {
        'F': |len| f"F({len})[+F({len*0.7})F({len*0.7})][-F({len*0.6})]"
    },
    iterations = 5
)
```

---

## 4.6 lsys.3d

**3D L-system (rotations in X, Y, Z)**

```morphogen
lsys.3d(
    axiom: String,
    rules: Map<Char, String>,
    rotations: Map<Char, Rotation3D>
) -> LSystem
```

**Commands:**
- `+`, `-` — Rotate X
- `&`, `^` — Rotate Y
- `\`, `/` — Rotate Z

---

# 5. Swarm Intelligence

## 5.1 swarm.ants

**Ant Colony Optimization (ACO)**

```morphogen
swarm.ants(
    graph: Graph,
    start: NodeRef,
    goal: NodeRef,
    num_ants: i32 = 50,
    evaporation: f32 = 0.1,
    alpha: f32 = 1.0,
    beta: f32 = 2.0,
    iterations: i32 = 100,
    seed: u64
) -> (Path, Field<f32>)
```

**Parameters:**
- `graph` — Network to search
- `start`, `goal` — Start and end nodes
- `num_ants` — Number of ants per iteration
- `evaporation` — Pheromone evaporation rate
- `alpha` — Pheromone importance
- `beta` — Heuristic (distance) importance
- `iterations` — Number of iterations
- `seed` — RNG seed

**Returns:**
- Best path found
- Pheromone field (for visualization)

**Use Cases:**
- Pathfinding
- TSP (Traveling Salesman Problem)
- Network routing

**Example:**
```morphogen
let graph = graph.from_geometry(pcb_layout)
let (path, pheromones) = swarm.ants(
    graph,
    start = pin_a,
    goal = pin_b,
    num_ants = 50,
    iterations = 100,
    seed = 42
)

out geometry = geom.from_path(path, width=0.2mm)
```

**Determinism:** Repro

---

## 5.2 swarm.slime_mold

**Physarum polycephalum network formation**

```morphogen
swarm.slime_mold(
    field: Field2D<f32>,
    food_sources: Array<Vec2<m>>,
    dt: f32 = 0.1,
    sensitivity: f32 = 1.0,
    iterations: i32 = 1000
) -> Field2D<f32>
```

**Semantics:**
1. Initialize field with food sources
2. Diffuse nutrients
3. Grow along nutrient gradients
4. Prune weak connections

**Use Cases:**
- Network generation (roads, veins, wiring)
- Optimal transport networks
- Procedural vascular systems

**Example:**
```morphogen
// Generate road network connecting cities
let field = field.create(512, 512, 0.0)
let cities = [vec2(100, 100), vec2(400, 100), vec2(250, 400)]

let network = swarm.slime_mold(
    field,
    food_sources = cities,
    dt = 0.1,
    iterations = 5000
)

let roads = network.threshold(0.3).to_graph()
out geometry = geom.from_graph(roads, diameter=2m)
```

**Determinism:** Repro

---

## 5.3 swarm.firefly

**Firefly algorithm**

```morphogen
swarm.firefly(
    objective: Fn(Vec<f32>) -> f32,
    bounds: Array<(f32, f32)>,
    num_fireflies: i32 = 30,
    alpha: f32 = 0.5,
    beta: f32 = 1.0,
    gamma: f32 = 1.0,
    iterations: i32 = 100,
    seed: u64
) -> (Vec<f32>, f32)
```

**Use Cases:**
- Continuous optimization
- Parameter tuning

---

## 5.4 swarm.bees

**Artificial Bee Colony (ABC)**

```morphogen
swarm.bees(
    objective: Fn(Vec<f32>) -> f32,
    bounds: Array<(f32, f32)>,
    num_bees: i32 = 50,
    limit: i32 = 10,
    iterations: i32 = 100,
    seed: u64
) -> (Vec<f32>, f32)
```

---

## 5.5 swarm.fish_school

**Fish School Search (FSS)**

```morphogen
swarm.fish_school(
    objective: Fn(Vec<f32>) -> f32,
    bounds: Array<(f32, f32)>,
    num_fish: i32 = 30,
    iterations: i32 = 100,
    seed: u64
) -> (Vec<f32>, f32)
```

---

# 6. Utilities

## 6.1 emergence.visualize

**Visualize emergence system**

```morphogen
emergence.visualize(
    system: CAGrid2D | Agents | Field2D,
    palette: String = "viridis",
    render_mode: String = "density"
) -> Image
```

---

## 6.2 emergence.animate

**Animate emergence system evolution**

```morphogen
emergence.animate(
    system: CAGrid2D | Agents,
    rule: Rule,
    steps: i32,
    fps: i32 = 30
) -> Video
```

---

## 6.3 emergence.export_geometry

**Export emergence result as geometry**

```morphogen
emergence.export_geometry(
    system: CAGrid2D | Field2D,
    method: String = "extrude",
    scale: f32<m> = 1m
) -> Mesh
```

**Methods:**
- `"extrude"` — Height extrusion
- `"isosurface"` — Marching cubes
- `"network"` — Graph → tubes

---

## 6.4 emergence.sonify

**Sonify emergence system**

```morphogen
emergence.sonify(
    system: CAGrid2D | Agents,
    mapping: String = "density_to_freq",
    freq_range: (f32<Hz>, f32<Hz>) = (200Hz, 2000Hz)
) -> Stream<f32, time, audio>
```

**Mappings:**
- `"density_to_freq"` — Cell/agent density → frequency
- `"position_to_pan"` — Agent position → stereo panning
- `"velocity_to_pitch"` — Agent velocity → pitch bend

---

## 6.5 emergence.couple_field

**Couple emergence system to field**

```morphogen
emergence.couple_field(
    agents: Agents<A>,
    field: Field2D<f32>,
    mode: String = "bidirectional"
) -> (Agents<A>, Field2D<f32>)
```

**Modes:**
- `"agents_to_field"` — One-way coupling
- `"field_to_agents"`
- `"bidirectional"`

---

## 6.6 emergence.measure

**Measure emergence metrics**

```morphogen
emergence.measure(
    system: CAGrid2D | Agents,
    metric: String
) -> f32
```

**Metrics:**
- `"entropy"` — Shannon entropy
- `"clustering"` — Spatial clustering
- `"order_parameter"` — Phase transition order parameter (Vicsek)

---

## 6.7 emergence.save

**Save emergence system state**

```morphogen
emergence.save(
    system: CAGrid2D | Agents,
    path: String
) -> ()
```

---

## 6.8 emergence.load

**Load emergence system state**

```morphogen
emergence.load(
    path: String
) -> CAGrid2D | Agents
```

---

# Cross-Domain Integration Patterns

## Pattern 1: Emergence → Geometry

**CA/RD Pattern → 3D Surface**

```morphogen
// 1. Generate pattern
let (u, v) = rd.gray_scott(u, v, f=0.04, k=0.06)

// 2. Convert to heightmap
let heightmap = v * 10.0

// 3. Create geometry
let plane = geom.plane(200mm, 200mm, resolution=(512, 512))
let surface = geom.displace(plane, heightmap, scale=20mm)

// 4. Export
io.export_stl(surface, "organic_pattern.stl")
```

---

## Pattern 2: Emergence → Physics

**Slime Mold Network → Structural Analysis**

```morphogen
// 1. Generate network
let network = swarm.slime_mold(field, food_sources=anchor_points)

// 2. Convert to 3D structure
let structure = geom.from_network(network, diameter=5mm)

// 3. Physics simulation
let stress = physics.stress_test(structure, load=vec3(0, -100N, 0))

// 4. Visualize
out visual = visual.render_stress(stress, palette="plasma")
```

---

## Pattern 3: Emergence → Audio

**Boids → Sonification**

```morphogen
scene BoidsAudio {
    let boids = agent.create(count=100, bounds=box(100m, 100m), seed=42)

    step(dt: Time) {
        // Update boids
        boids = agent.boids(boids, separation=1.5, dt=dt)

        // Sonify density
        let density = agent.to_field(boids, property="density", resolution=(64, 64))
        let avg_density = field.reduce(density, "mean")

        // Audio output
        let freq = 200Hz + avg_density * 800Hz
        out audio = osc.sine(freq)
    }
}
```

---

## Pattern 4: Emergence → Optimization

**PSO Visualization (from OptimizationDomain)**

```morphogen
// Run PSO
let result = opt.particle_swarm(
    objective = |params| chamber_efficiency(params),
    bounds = [(50mm, 200mm); 8],
    particles = 30,
    iterations = 100,
    seed = 42
)

// Visualize swarm history as agents
let swarm_agents = emergence.from_pso_history(result.history)
out visual = visual.animate_agents(swarm_agents)
```

---

## Pattern 5: Emergence → Acoustics

**Swarm → Acoustic Scattering**

```morphogen
scene SwarmScattering {
    let boids = agent.create(count=1000, bounds=box(50m, 50m), seed=42)
    let source = acoustic.point_source(position=vec3(0, 0, 0), freq=1000Hz)

    step(dt: Time) {
        // Update boids
        boids = agent.boids(boids, dt=dt)

        // Acoustic scattering
        let positions = agent.positions(boids)
        let wave = acoustic.propagate_with_scatterers(
            source,
            scatterers = positions,
            radius = 0.1m
        )

        // Audio output
        let mic_pressure = acoustic.sample(wave, position=vec3(10, 0, 0))
        out audio = acoustic.to_audio(mic_pressure)
    }
}
```

---

# Implementation Roadmap

## Phase 1: Core Infrastructure (v0.10)
- [ ] `CAGrid2D<T>`, `CAGrid3D<T>` types
- [ ] `Agents<A>` container
- [ ] Spatial indexing (grid, k-d tree)
- [ ] Boundary condition handling

## Phase 2: CA (v0.10)
- [ ] `ca.create`, `ca.step`, `ca.step_n`
- [ ] `ca.rule_preset` (9 presets)
- [ ] `ca.lenia`
- [ ] `ca.to_field`, `ca.from_field`

## Phase 3: ABM (v0.10)
- [ ] `agent.create`
- [ ] `agent.boids`, `agent.vicsek`, `agent.schelling`
- [ ] `agent.to_field`, `agent.from_field`
- [ ] Neighbor search optimization

## Phase 4: RD (v0.11)
- [ ] `rd.gray_scott`, `rd.turing`
- [ ] `rd.to_geometry` (marching cubes)

## Phase 5: L-Systems (v0.11)
- [ ] `lsys.create`, `lsys.evolve`
- [ ] `lsys.to_geometry` (turtle interpreter)
- [ ] Presets (trees, fractals)

## Phase 6: Swarm (v1.0)
- [ ] `swarm.ants`, `swarm.slime_mold`, `swarm.firefly`
- [ ] Integration with OptimizationDomain

## Phase 7: Cross-Domain Examples (v1.0)
- [ ] 10+ examples showing integration patterns

---

# Testing Strategy

## Determinism Tests
```morphogen
// CA determinism
let ca1 = ca.create(128, 128, seed=42)
let ca2 = ca.create(128, 128, seed=42)
assert_eq!(ca1, ca2)

// Boids determinism
let boids1 = agent.boids(agents, seed=42)
let boids2 = agent.boids(agents, seed=42)
assert_eq!(boids1, boids2)
```

## Conservation Tests
```morphogen
// Momentum conservation (boids)
let p_before = agent.total_momentum(boids)
boids = agent.boids(boids, dt=0.1s)
let p_after = agent.total_momentum(boids)
assert_approx_eq!(p_before, p_after, tol=1e-6)
```

## Pattern Recognition Tests
```morphogen
// Life: glider moves diagonally
let glider = ca.load_pattern("glider")
let evolved = ca.step_n(glider, rule=life_rule, steps=4)
let position = ca.find_pattern(evolved, "glider")
assert_eq!(position, vec2(1, 1))
```

---

# Performance Benchmarks

## Target Performance

| Operator | Grid/Agents | GPU Speedup | Throughput |
|----------|-------------|-------------|------------|
| `ca.step` | 1024² CA | 50x | 1000 fps |
| `agent.boids` | 10k agents | 20x | 60 fps |
| `rd.gray_scott` | 512² | 100x | 500 fps |
| `lsys.to_geometry` | depth=10 | CPU-only | 1 sec |
| `swarm.slime_mold` | 512², 1k iter | 50x | 10 sec |

---

# Summary

The EmergenceDomain provides **45 operators** spanning:
- ✅ 9 CA operators
- ✅ 12 ABM operators
- ✅ 5 RD operators
- ✅ 6 L-system operators
- ✅ 5 swarm intelligence operators
- ✅ 8 utilities

This enables:
- 🎨 Generative art and design
- 🧬 Biological morphogenesis modeling
- 🤖 Swarm robotics algorithms
- 🌊 Complex adaptive systems research
- 🔗 Cross-domain workflows (Geometry, Physics, Audio, Optimization)

**No existing tool provides this level of integration.**

---

**End of Catalog**
