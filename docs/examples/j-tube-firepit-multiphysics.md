# J-Tube Fire Pit: A Multi-Physics Engineering Example for Morphogen

**Version:** 1.0
**Status:** Design Document
**Last Updated:** 2025-11-15
**Authors:** Morphogen Architecture Team

---

## Overview

The J-tube fire pit is a **multi-physics system in steel** — and Morphogen is exactly the kind of platform that should model it. This document demonstrates how Morphogen's operator graph paradigm extends from audio/graphics into engineering physics domains.

### What This Document Demonstrates

1. **Physical System Description** — How the J-tube fire pit works as a multi-physics engine
2. **Morphogen Simulation Pipeline** — Stage-by-stage modeling from inputs to outputs
3. **New Domain Requirements** — FluidNetwork, ThermalODE, FluidJet, CombustionLight domains
4. **Operator Specifications** — Detailed operator definitions for each domain
5. **Cross-Domain Integration** — How geometry, thermal, fluid, and combustion domains compose
6. **Design Optimization** — Using Morphogen for parametric design iteration

### Why This Matters

The J-tube fire pit validates Morphogen's multi-domain vision:
- It's not "just another use case" — it proves the operator graph mental model works for **engineering design**
- It pushes Morphogen into **physics modeling**, not just audio/graphics
- It shows that domains like FluidNetwork, ThermalODE, and CombustionLight are **worth formalizing**
- It demonstrates **reference-based composition** (anchors, frames) across physical systems

---

## 1. The J-Tube Fire Pit: Physical System

### 1.1 The Problem: Smokeless Fire

Traditional fire pits are smoky because incomplete combustion produces:
- Unburnt hydrocarbons
- Carbon monoxide (CO)
- Soot particles

**Solution:** **Secondary combustion air** — inject preheated air into the flame zone to burn the smoke.

---

### 1.2 The J-Tube Design

Around the fire pit, you have **external J-shaped tubes**. Each tube:

1. **Inhales cool air** near the ground (intake at the foot of the J)
2. **Preheats that air** as it runs up alongside hot metal and hot exhaust
3. **Injects it as high-speed jets** into the flame zone through holes or nozzles (possibly with rotatable caps/sleeves)

Those jets provide **secondary combustion air**:
- They burn smoke (unburnt hydrocarbons, CO, soot)
- They increase flame temperature
- They create visible jet structures / halos when geometry is tuned

**Physically, the system is:**
```
hot core → buoyant draft → pressure drop → air network in J-tubes
         → preheated jets → secondary combustion in flame
```

That's a **graph of physics operators**, which is exactly how Morphogen thinks.

---

### 1.3 Design Parameters

**Geometry:**
- Pit diameter, height, wall thickness
- J-tube length, diameter, bends, nozzle shape
- Number of tubes, spacing

**Material Properties:**
- Steel conductivity, emissivity, wall thickness

**Ambient Conditions:**
- Temperature (Tₐₘb), pressure, wind

**Fire State:**
- Approximate flame temperature
- Heat release rate (fuel burn rate)

---

## 2. Simulating the J-Tube System in Morphogen

We'll walk through a Morphogen-style simulation pipeline from **inputs → outputs**, showing which domains/operators handle each step.

---

### Stage 0: Inputs & Geometry

**Goal:** Define the physical configuration.

**Morphogen Domains:** `GeometryDomain`

**Types:**
- `Pipe` — J-tube geometry
- `Chamber` — Fire pit core
- `Surface` — Walls, nozzles
- `Frame` — Coordinate frames for assembly

**Refs / Anchors:**
- `fire_core.center`
- `tube[i].inlet`, `tube[i].outlet`
- `tube[i].heated_section`

**Morphogen Code:**
```morphogen
// Fire pit geometry
let pit = geom.cylinder(
    radius = 250mm,
    height = 400mm,
    wall_thickness = 3mm
)

// J-tube geometry (parametric)
fn create_j_tube(
    length: Length,
    diameter: Length,
    bend_radius: Length,
    nozzle_area: Area
) -> Pipe {
    let vertical = sketch.line(origin, vertical_dir * length)
    let bend = sketch.arc(radius=bend_radius, angle=90deg)
    let horizontal = sketch.line(bend.end, horizontal_dir * 50mm)

    let path = sketch.composite([vertical, bend, horizontal])
    let profile = sketch.circle(radius=diameter/2)

    return sweep(profile, path)
}

// Create 6 J-tubes around pit
let tubes = pattern.circular(
    create_j_tube(
        length = 300mm,
        diameter = 25mm,
        bend_radius = 50mm,
        nozzle_area = 100mm²
    ),
    axis = "z",
    count = 6,
    center = pit.anchor("center")
)
```

**Key Insight:** Geometry provides **anchor points** for the fluid network and thermal modeling.

---

### Stage 1: Draft Pressure (the "engine")

**Goal:** Estimate the pressure drop that drives flow through the tubes.

**Physics:** Stack effect / draft

$$
\Delta P \approx \rho g H \left(\frac{1}{T_{\text{amb}}} - \frac{1}{T_{\text{flue}}}\right)
$$

**Morphogen Domain:** `FluidNetworkDomain`

**Operators:**
- `draft_pressure(chamber_ref, chimney_height, T_amb, T_hot) -> ΔP`
- `chimney_effect(chamber_geometry, flame_temp) -> ΔP`

**Morphogen Code:**
```morphogen
// Estimate draft pressure
let delta_p = fluid_net.draft_pressure(
    chamber = pit,
    height = pit.measure.height(),
    T_amb = 293K,  // 20°C
    T_hot = 1200K  // Flame temperature
)

// Result: ΔP ≈ 5-15 Pa (typical for fire pits)
```

**Graph Structure:**
```
[FireState, Geometry] --draft_pressure--> [ΔP: Scalar<Pa>]
```

---

### Stage 2: Flow Through Each J-Tube (network solve)

**Goal:** Given ΔP, compute flow rate through each J-tube and speed at its outlet.

**Physics:** Each tube is a pipe with friction and minor losses (Darcy-Weisbach + bend loss + entrance/exit losses).

**Morphogen Domain:** `FluidNetworkDomain`

**Types:**
- `Tube` — Pipe geometry + flow resistance
- `Junction` — Connection points
- `FlowNet` — Graph of tubes + junctions
- `TubeRef` — Reference to tube in network

**Operators:**
- `tube_resistance(tube_geometry, roughness, Re_guess) -> R_tube`
- `network_solve(ΔP, tubes[]) -> {m_dot[i], p_in[i], p_out[i]}`

**Morphogen Code:**
```morphogen
// Create fluid network
let flow_net = fluid_net.create()

// Add each J-tube to network
for tube in tubes {
    let resistance = fluid_net.tube_resistance(
        geometry = tube,
        roughness = 0.045mm,  // Steel roughness
        fluid = air,
        Re_guess = 5000  // Turbulent flow
    )

    flow_net.add_tube(tube, resistance)
}

// Solve network for flows
let flow_solution = fluid_net.network_solve(
    net = flow_net,
    delta_p = delta_p
)

// Extract per-tube results
let m_dot = flow_solution.mass_flow  // kg/s per tube
let v_exit = flow_solution.exit_velocity  // m/s at nozzle
```

**Output:** Mass flow `m_dot[i]`, volumetric flow `Q[i]`, exit velocity `v_exit[i]` for each tube.

**Graph Structure:**
```
[ΔP, TubeGeometries] --network_solve--> [FlowPerTube]
```

**Key Insight:** This is structurally similar to:
- Circuit solver (electrical networks)
- DSP graph solver (audio routing)
- N-body integrator (force summation)

So Morphogen's general solver infrastructure is **reused**.

---

### Stage 3: Preheating Along the Tube (1D thermal ODE)

**Goal:** Estimate air temperature at the nozzle.

**Physics:** Treat each tube as a 1D heated duct:

$$
\dot{m} c_p \frac{dT_{\text{air}}}{dx} = h A_s (T_{\text{wall}}(x) - T_{\text{air}})
$$

**Morphogen Domain:** `ThermalODEDomain`

**Types:**
- `TubeSegment` — 1D thermal model
- `ThermalProfile` — Temperature distribution
- `WallTempModel` — Wall temperature as function of position

**Operators:**
- `wall_temp_model(fire_state, tube_position) -> T_wall(x)`
- `heat_transfer_1D(tube_ref, m_dot, T_in, wall_temp_model) -> T_out`

**Morphogen Code:**
```morphogen
// Estimate wall temperature profile
let wall_temp = thermal.wall_temp_model(
    fire_state = pit.fire_state,
    tube = tube,
    position = tube.heated_section
)

// Solve 1D heat transfer ODE
let T_jet = thermal.heat_transfer_1D(
    tube = tube,
    m_dot = m_dot[i],
    T_in = T_amb,
    wall_temp = wall_temp,
    integrator = "rk4"
)

// Result: T_jet ≈ 400-600K (preheated air)
```

**Internally:** This is just an ODE integrator node in the Morphogen graph, lowered to:
- Explicit Euler (fast, approximate)
- Runge-Kutta
- Vectorized stepper on GPU

**Graph Structure:**
```
[TubeRef, m_dot[i], T_amb, FireState] --heat_transfer_1D--> [T_jet[i]]
```

---

### Stage 4: Jet Behavior at the Outlet (mixing & momentum)

**Goal:** Understand how each jet interacts with the flame plume.

**Physics:**
- Jet velocity `v_jet[i]`
- Jet direction (angle relative to radial/tangential)
- Reynolds number, entrainment estimate

**Morphogen Domain:** `FluidJetDomain`

**Types:**
- `Jet` — Velocity, temperature, direction, area
- `JetArray` — Collection of jets
- `JetRef` — Reference to jet

**Operators:**
- `jet_from_tube(tube_ref, m_dot[i], T_jet[i]) -> Jet`
- `jet_reynolds(Jet) -> Re`
- `jet_entrainment(Jet, plume_state) -> mixing_factor`
- `jet_visualization(JetArray, fire_core.ref) -> Field2D/3D`

**Morphogen Code:**
```morphogen
// Create jet from tube exit
let jet = fluid_jet.from_tube(
    tube = tube,
    m_dot = m_dot[i],
    T_out = T_jet[i]
)

// Compute Reynolds number
let Re = fluid_jet.reynolds(jet)

// Estimate entrainment with plume
let mixing = fluid_jet.entrainment(
    jet = jet,
    plume = pit.plume_state
)

// Visualize jet field (CFD-lite)
let jet_field = fluid_jet.visualization(
    jets = all_jets,
    frame = pit.anchor("center").frame()
)
```

**Output:**
- `Jets[]` — Array of jet objects
- `MixingEstimate` — How well jets penetrate flame zone

**Graph Structure:**
```
[Tube, m_dot, T_jet] --jet_from_tube--> [Jets[]]
[Jets, FlameState] --jet_entrainment--> [MixingEstimate]
```

**Visualization:** The `jet_visualization` operator doesn't run full CFD; it generates a **vector field** (quiver plot) or parametric curve for quick "CFD-lite" view.

---

### Stage 5: Combustion + Smoke Index (simplified chemistry)

**Goal:** Estimate how effective the jets are at reburning smoke.

**Physics:** Approximate combustion metrics:
- Equivalence ratio (φ)
- Oxygen availability
- Residence time in hot zone
- Smoke reduction factor

**Morphogen Domain:** `CombustionLightDomain`

**Types:**
- `MixtureState` — Fuel, air, equivalence ratio
- `CombustionZone` — Temperature, pressure, residence time
- `SmokeIndex` — Smoke reduction metric (0 = smoky, 1 = clean)

**Operators:**
- `equivalence_ratio(fuel_rate, primary_air, secondary_air_total) -> φ`
- `zone_temperature(FireState, JetHeating) -> T_zone`
- `smoke_reduction(φ, T_zone, mixing_factor, residence_time) -> SmokeIndex`

**Morphogen Code:**
```morphogen
// Compute equivalence ratio
let phi = combustion.equivalence_ratio(
    fuel_rate = pit.fuel_rate,
    primary_air = pit.primary_air_flow,
    secondary_air = sum(m_dot)  // Total from all tubes
)

// Estimate combustion zone temperature
let T_zone = combustion.zone_temperature(
    fire_state = pit.fire_state,
    jet_heating = sum(m_dot * T_jet)
)

// Estimate smoke reduction
let smoke_index = combustion.smoke_reduction(
    phi = phi,
    T_zone = T_zone,
    mixing = mixing,
    residence_time = 0.1s  // Estimate
)

// Result: SmokeIndex ≈ 0.8 (80% smoke reduction)
```

**Graph Structure:**
```
[FuelInput, PrimaryAir, SecondaryTotalFlow] --equivalence_ratio--> [φ]
[FireState, Jets, Mixing] --zone_temperature--> [T_zone]
[φ, T_zone, Mixing, TimeScale] --smoke_reduction--> [SmokeIndex]
```

**Key Insight:** Even if this is rough, as a **design tool** it lets you compare geometries and jet configurations without going all the way to OpenFOAM.

---

### Stage 6: Visualization & Design Iteration

**Goal:** Turn all simulation outputs into design insight.

**Morphogen Domains:** `VisualizationDomain`, `OptimizationDomain`

**Operators:**
- `plot_flow_per_tube(FlowPerTube)`
- `plot_jet_map(JetArray, Geometry)`
- `plot_smoke_index_vs_param(SmokeIndex, param_sweep)`
- `animate_fire_halo(JetArray, time_variation)`
- `param_sweep(parameters, pipeline_subgraph) -> results[]`
- `search_best(parameters, objective=SmokeIndex_min) -> best_config`

**Morphogen Code:**
```morphogen
// Visualization
visual.plot_jet_map(
    jets = all_jets,
    geometry = pit,
    camera = camera.top_down()
)

// Parameter sweep
let sweep_results = optimize.param_sweep(
    params = {
        tube_diameter: range(20mm, 35mm, step=5mm),
        nozzle_area: range(50mm², 200mm², step=25mm²),
        tube_count: [4, 6, 8]
    },
    pipeline = j_tube_simulation,  // Reference to full pipeline
    objective = smoke_index
)

// Find best configuration
let best = optimize.search_best(
    results = sweep_results,
    objective = maximize(smoke_index),
    constraints = [
        v_exit < 15 m/s,  // Avoid excessive noise
        m_dot_total < 0.05 kg/s  // Reasonable air flow
    ]
)

// Report
print("Best configuration:")
print(f"  Tube diameter: {best.tube_diameter}")
print(f"  Nozzle area: {best.nozzle_area}")
print(f"  Smoke index: {best.smoke_index}")
```

**Output:** Optimal design parameters for maximum smoke reduction.

---

## 3. New Domains & Operators Morphogen Should Add

From the pipeline above, we can list the concrete domain additions Morphogen should grow.

---

### 3.1 FluidNetworkDomain

**Purpose:** 1D / lumped flow networks (fluids as circuits)

#### Core Types

```morphogen
type Tube {
    geometry: Pipe,
    diameter: Length,
    length: Length,
    roughness: Length,
    bends: i32,
    nozzle: Option<Nozzle>
}

type Junction {
    position: Vec3<m>,
    connected_tubes: List<TubeRef>
}

type FlowNet {
    tubes: List<Tube>,
    junctions: List<Junction>,
    graph: Graph
}

type TubeRef = Ref<Tube>
type JunctionRef = Ref<Junction>
```

#### Key Operators

**Draft Pressure:**
```json
{
  "name": "draft_pressure",
  "domain": "fluid_network",
  "layer": 4,
  "inputs": [
    {"name": "chamber_ref", "type": "Ref<Chamber>"},
    {"name": "height", "type": "f32<m>"},
    {"name": "T_amb", "type": "f32<K>"},
    {"name": "T_hot", "type": "f32<K>"}
  ],
  "outputs": [
    {"name": "delta_p", "type": "f32<Pa>"}
  ],
  "determinism": "strict",
  "description": "Compute draft pressure from stack effect"
}
```

**Tube Resistance:**
```json
{
  "name": "tube_resistance",
  "domain": "fluid_network",
  "layer": 4,
  "inputs": [
    {"name": "tube_ref", "type": "TubeRef"},
    {"name": "roughness", "type": "f32<m>"},
    {"name": "Re_guess", "type": "f32"}
  ],
  "outputs": [
    {"name": "R_tube", "type": "f32<Pa·s²/kg²>"}
  ],
  "determinism": "strict",
  "description": "Compute flow resistance (Darcy-Weisbach + minor losses)"
}
```

**Network Solve:**
```json
{
  "name": "network_solve",
  "domain": "fluid_network",
  "layer": 4,
  "inputs": [
    {"name": "net", "type": "FlowNet"},
    {"name": "delta_p", "type": "f32<Pa>"}
  ],
  "outputs": [
    {"name": "flows", "type": "Array<MassFlow>"},
    {"name": "pressures", "type": "Array<Pressure>"}
  ],
  "determinism": "repro",
  "lowering": {"dialect": "morphogen.fluid", "template": "mna_fluid_solver"},
  "description": "Solve fluid network (Modified Nodal Analysis)"
}
```

**Use Cases:**
- Fire pits
- HVAC systems
- Muffler design
- Intake manifolds
- Pneumatic circuits

---

### 3.2 ThermalODEDomain

**Purpose:** Simple 1D thermal modeling (heat in tubes, walls, rods)

#### Core Types

```morphogen
type ThermalSegment {
    length: Length,
    diameter: Length,
    wall_thickness: Length,
    conductivity: f32<W/(m·K)>,
    emissivity: f32
}

type WallTempModel {
    profile: Fn(x: f32<m>) -> f32<K>
}

type ThermalProfile {
    temperatures: Array<f32<K>>,
    positions: Array<f32<m>>
}
```

#### Key Operators

**Heat Transfer 1D:**
```json
{
  "name": "heat_transfer_1D",
  "domain": "thermal_ode",
  "layer": 4,
  "inputs": [
    {"name": "segment_ref", "type": "Ref<ThermalSegment>"},
    {"name": "m_dot", "type": "f32<kg/s>"},
    {"name": "T_in", "type": "f32<K>"},
    {"name": "wall_temp_model", "type": "WallTempModel"}
  ],
  "params": [
    {"name": "integrator", "type": "Enum", "default": "rk4", "enum": ["euler", "rk2", "rk4"]}
  ],
  "outputs": [
    {"name": "T_out", "type": "f32<K>"}
  ],
  "determinism": "repro",
  "description": "Solve 1D heat transfer ODE along tube"
}
```

**Lumped Capacity:**
```json
{
  "name": "lumped_capacity",
  "domain": "thermal_ode",
  "layer": 4,
  "inputs": [
    {"name": "body_ref", "type": "Ref<Body>"},
    {"name": "heat_input", "type": "f32<W>"}
  ],
  "outputs": [
    {"name": "T_body", "type": "Stream<f32<K>,time>"}
  ],
  "determinism": "repro",
  "description": "Lumped thermal capacity model"
}
```

**Use Cases:**
- Fire pits
- Heat pipes
- Hotend design (3D printers)
- Battery thermal management
- Heat exchangers

---

### 3.3 FluidJetDomain

**Purpose:** Model jet exits and their interaction with surrounding flow

#### Core Types

```morphogen
type Jet {
    flow: f32<kg/s>,
    velocity: f32<m/s>,
    temperature: f32<K>,
    direction: Vec3,
    area: f32<m²>
}

type JetArray {
    jets: List<Jet>,
    count: i32
}
```

#### Key Operators

**Jet from Tube:**
```json
{
  "name": "jet_from_tube",
  "domain": "fluid_jet",
  "layer": 4,
  "inputs": [
    {"name": "tube_ref", "type": "TubeRef"},
    {"name": "m_dot", "type": "f32<kg/s>"},
    {"name": "T_out", "type": "f32<K>"}
  ],
  "outputs": [
    {"name": "jet", "type": "Jet"}
  ],
  "determinism": "strict",
  "description": "Create jet from tube exit conditions"
}
```

**Jet Reynolds:**
```json
{
  "name": "jet_reynolds",
  "domain": "fluid_jet",
  "layer": 4,
  "inputs": [
    {"name": "jet", "type": "Jet"}
  ],
  "outputs": [
    {"name": "Re", "type": "f32"}
  ],
  "determinism": "strict",
  "description": "Compute jet Reynolds number"
}
```

**Jet Entrainment:**
```json
{
  "name": "jet_entrainment",
  "domain": "fluid_jet",
  "layer": 4,
  "inputs": [
    {"name": "jet", "type": "Jet"},
    {"name": "plume_ref", "type": "Ref<Plume>"}
  ],
  "outputs": [
    {"name": "mixing_factor", "type": "f32"}
  ],
  "determinism": "repro",
  "description": "Estimate jet entrainment and mixing"
}
```

**Jet Field Visualization:**
```json
{
  "name": "jet_field",
  "domain": "fluid_jet",
  "layer": 6,
  "inputs": [
    {"name": "jet_array", "type": "JetArray"},
    {"name": "frame_ref", "type": "FrameRef"}
  ],
  "outputs": [
    {"name": "field", "type": "Field2D<Vec2>"}
  ],
  "determinism": "repro",
  "description": "Generate vector field visualization of jets (CFD-lite)"
}
```

**Use Cases:**
- Fire pits
- Burners
- Rocket nozzles
- Spray systems
- Ventilation

---

### 3.4 CombustionLightDomain

**Purpose:** Approximated combustion metrics without heavy CFD

#### Core Types

```morphogen
type MixtureState {
    fuel_rate: f32<kg/s>,
    air_rate: f32<kg/s>,
    equivalence_ratio: f32
}

type CombustionZone {
    temperature: f32<K>,
    pressure: f32<Pa>,
    residence_time: f32<s>
}

type SmokeIndex {
    value: f32,  // 0 = smoky, 1 = clean
    reduction_factor: f32
}
```

#### Key Operators

**Equivalence Ratio:**
```json
{
  "name": "equivalence_ratio",
  "domain": "combustion_light",
  "layer": 4,
  "inputs": [
    {"name": "fuel_rate", "type": "f32<kg/s>"},
    {"name": "air_rate", "type": "f32<kg/s>"}
  ],
  "outputs": [
    {"name": "phi", "type": "f32"}
  ],
  "determinism": "strict",
  "description": "Compute equivalence ratio (φ = actual/stoichiometric)"
}
```

**Zone Temperature:**
```json
{
  "name": "zone_temperature",
  "domain": "combustion_light",
  "layer": 4,
  "inputs": [
    {"name": "fire_state", "type": "Ref<FireState>"},
    {"name": "jet_info", "type": "JetArray"}
  ],
  "outputs": [
    {"name": "T_zone", "type": "f32<K>"}
  ],
  "determinism": "repro",
  "description": "Estimate combustion zone temperature"
}
```

**Smoke Reduction:**
```json
{
  "name": "smoke_reduction",
  "domain": "combustion_light",
  "layer": 4,
  "inputs": [
    {"name": "phi", "type": "f32"},
    {"name": "T_zone", "type": "f32<K>"},
    {"name": "mixing_factor", "type": "f32"},
    {"name": "residence_time", "type": "f32<s>"}
  ],
  "outputs": [
    {"name": "smoke_index", "type": "SmokeIndex"}
  ],
  "determinism": "repro",
  "description": "Estimate smoke reduction effectiveness"
}
```

**Use Cases:**
- Fire pits
- Burners
- Mufflers
- Engine exhaust modeling
- Incinerators

---

### 3.5 Geometry Extensions (Pipe-Centric Primitives)

**New Geometry Ops:**

**Pipe Primitive:**
```json
{
  "name": "pipe",
  "domain": "geometry",
  "layer": 6,
  "inputs": [
    {"name": "centerline", "type": "Curve"},
    {"name": "diameter", "type": "f32<m>"},
    {"name": "thickness", "type": "f32<m>"}
  ],
  "outputs": [
    {"name": "pipe", "type": "Pipe"}
  ],
  "determinism": "strict",
  "description": "Create pipe solid from centerline curve"
}
```

**Pipe Anchors:**
```json
{
  "name": "pipe_anchors",
  "domain": "geometry",
  "layer": 6,
  "inputs": [
    {"name": "pipe", "type": "Pipe"}
  ],
  "outputs": [
    {"name": "anchors", "type": "AnchorSet"}
  ],
  "determinism": "strict",
  "description": "Generate anchors: inlet, outlet, heated_section, midpoints"
}
```

**Auto-Generated Anchors for Pipes:**
- `.inlet` — Pipe inlet position and orientation
- `.outlet` — Pipe outlet position and orientation
- `.heated_section` — Section along hot wall
- `.midpoint[n]` — Points along centerline
- `.axis` — Pipe axis direction

---

### 3.6 Visualization & Optimization Helpers

**Visualization Ops:**

```json
{
  "name": "plot_quiver",
  "domain": "visualization",
  "layer": 6,
  "inputs": [
    {"name": "field_ref", "type": "Field2D<Vec2>"}
  ],
  "outputs": [
    {"name": "image", "type": "Image"}
  ],
  "determinism": "repro"
}
```

```json
{
  "name": "plot_polar",
  "domain": "visualization",
  "layer": 6,
  "inputs": [
    {"name": "angles", "type": "Array<f32<rad>>"},
    {"name": "magnitudes", "type": "Array<f32>"}
  ],
  "outputs": [
    {"name": "image", "type": "Image"}
  ],
  "determinism": "repro"
}
```

**Optimization Ops:**

```json
{
  "name": "param_sweep",
  "domain": "optimization",
  "layer": 7,
  "inputs": [
    {"name": "param_space", "type": "ParamSpace"},
    {"name": "subgraph", "type": "Graph"}
  ],
  "outputs": [
    {"name": "results", "type": "ResultsTable"}
  ],
  "determinism": "repro",
  "description": "Sweep parameters and collect results"
}
```

```json
{
  "name": "search_best",
  "domain": "optimization",
  "layer": 7,
  "inputs": [
    {"name": "result_table", "type": "ResultsTable"},
    {"name": "objective_expr", "type": "Expr"}
  ],
  "outputs": [
    {"name": "best_config", "type": "Config"}
  ],
  "determinism": "repro",
  "description": "Find best configuration by objective"
}
```

---

## 4. Complete Morphogen Pipeline Example

Here's the full J-tube fire pit simulation as a Morphogen program:

```morphogen
scene JTubeFirePit {
    // === STAGE 0: GEOMETRY ===

    // Fire pit core
    let pit = geom.cylinder(
        radius = 250mm,
        height = 400mm,
        wall_thickness = 3mm
    )

    // J-tube template
    fn create_j_tube(
        diameter: Length,
        nozzle_area: Area
    ) -> Pipe {
        let vertical = sketch.line(origin, Vec3(0, 0, 300mm))
        let bend = sketch.arc(radius=50mm, angle=90deg)
        let horizontal = sketch.line(bend.end, Vec3(50mm, 0, 0))

        let path = sketch.composite([vertical, bend, horizontal])
        let profile = sketch.circle(radius=diameter/2)

        return sweep(profile, path)
    }

    // Create 6 J-tubes around pit
    let tube_diameter = param(25mm, range=[20mm, 35mm])
    let nozzle_area = param(100mm², range=[50mm², 200mm²])

    let tubes = pattern.circular(
        create_j_tube(tube_diameter, nozzle_area),
        axis = "z",
        count = 6,
        center = pit.anchor("center")
    )

    // === STAGE 1: DRAFT PRESSURE ===

    let T_amb = 293K
    let T_flame = 1200K

    let delta_p = fluid_net.draft_pressure(
        chamber = pit,
        height = pit.measure.height(),
        T_amb = T_amb,
        T_hot = T_flame
    )

    // === STAGE 2: FLOW NETWORK ===

    let flow_net = fluid_net.create()

    for tube in tubes {
        let resistance = fluid_net.tube_resistance(
            geometry = tube,
            roughness = 0.045mm,
            fluid = air,
            Re_guess = 5000
        )
        flow_net.add_tube(tube, resistance)
    }

    let flow_solution = fluid_net.network_solve(
        net = flow_net,
        delta_p = delta_p
    )

    // === STAGE 3: THERMAL PREHEATING ===

    let T_jets = []
    for (i, tube) in enumerate(tubes) {
        let wall_temp = thermal.wall_temp_model(
            fire_state = pit.fire_state,
            tube = tube,
            position = tube.heated_section
        )

        let T_jet = thermal.heat_transfer_1D(
            tube = tube,
            m_dot = flow_solution.mass_flow[i],
            T_in = T_amb,
            wall_temp = wall_temp,
            integrator = "rk4"
        )

        T_jets.push(T_jet)
    }

    // === STAGE 4: JETS ===

    let jets = []
    for (i, tube) in enumerate(tubes) {
        let jet = fluid_jet.from_tube(
            tube = tube,
            m_dot = flow_solution.mass_flow[i],
            T_out = T_jets[i]
        )
        jets.push(jet)
    }

    let mixing = fluid_jet.entrainment(
        jets = jets,
        plume = pit.plume_state
    )

    // === STAGE 5: COMBUSTION ===

    let total_secondary_air = sum(flow_solution.mass_flow)

    let phi = combustion.equivalence_ratio(
        fuel_rate = pit.fuel_rate,
        air_rate = pit.primary_air_flow + total_secondary_air
    )

    let T_zone = combustion.zone_temperature(
        fire_state = pit.fire_state,
        jet_info = jets
    )

    let smoke_index = combustion.smoke_reduction(
        phi = phi,
        T_zone = T_zone,
        mixing = mixing,
        residence_time = 0.1s
    )

    // === STAGE 6: VISUALIZATION ===

    let jet_field = fluid_jet.jet_field(
        jet_array = jets,
        frame_ref = pit.anchor("center").frame()
    )

    out visual = visual.plot_quiver(jet_field)

    // === OUTPUTS ===

    export {
        smoke_index: smoke_index.value,
        total_flow: total_secondary_air,
        avg_jet_temp: mean(T_jets),
        avg_jet_velocity: mean(flow_solution.exit_velocity)
    }
}
```

---

## 5. Why This Matters for Morphogen

### 5.1 Validates the Morphogen Mental Model

The J-tube fire pit proves that **operator graphs work for engineering**:
- Each stage is an operator or operator chain
- Domains compose cleanly (Geometry → Fluid → Thermal → Combustion → Visual)
- References/anchors connect physical systems

### 5.2 Pushes Morphogen into Engineering Modeling

Not just audio/graphics — now Morphogen handles:
- Fire pits
- Mufflers
- Intakes
- HVAC
- Heat exchangers
- Burners

All with the same **operator graph paradigm**.

### 5.3 Shows New Domains Are Worth Formalizing

FluidNetwork, ThermalODE, FluidJet, CombustionLight are **reusable** across:
- 2-stroke engines (mufflers, intakes)
- HVAC systems (air flow networks)
- Thermal management (cooling, heating)
- Combustion systems (engines, furnaces, burners)

### 5.4 Reinforces References/Anchors/Frames

The pipe anchors (`.inlet`, `.outlet`, `.heated_section`) are **critical** for:
- Connecting fluid networks to geometry
- Defining thermal boundary conditions
- Positioning jets in space

This validates ADR-002's anchor system.

---

## 6. Cross-Domain Integration Patterns

### 6.1 Geometry → FluidNetwork

**Pattern:** Geometry provides **pipe objects** with anchors, FluidNetwork uses those to build the flow graph.

```morphogen
let tube = geom.pipe(centerline, diameter, thickness)
let resistance = fluid_net.tube_resistance(geometry=tube, ...)
```

**Key:** Anchors (`.inlet`, `.outlet`) define connection points.

---

### 6.2 FluidNetwork → ThermalODE

**Pattern:** Flow rates from network solve become inputs to thermal ODE.

```morphogen
let flow_solution = fluid_net.network_solve(...)
let T_out = thermal.heat_transfer_1D(m_dot=flow_solution.mass_flow[i], ...)
```

**Key:** Mass flow is the **coupling variable**.

---

### 6.3 ThermalODE → FluidJet

**Pattern:** Exit temperature from thermal ODE defines jet properties.

```morphogen
let T_jet = thermal.heat_transfer_1D(...)
let jet = fluid_jet.from_tube(T_out=T_jet, ...)
```

**Key:** Temperature is the **coupling variable**.

---

### 6.4 FluidJet → CombustionLight

**Pattern:** Jet array feeds combustion zone temperature and mixing.

```morphogen
let jets = [...]
let T_zone = combustion.zone_temperature(jet_info=jets)
let smoke_index = combustion.smoke_reduction(T_zone=T_zone, mixing=..., ...)
```

**Key:** Jets provide **energy input** and **mixing enhancement**.

---

### 6.5 All Domains → Visualization

**Pattern:** All outputs can be visualized.

```morphogen
visual.plot_quiver(jet_field)
visual.plot_line(T_jets)
visual.plot_smoke_index(smoke_index)
```

**Key:** Visualization is **cross-cutting**.

---

## 7. Design Optimization Workflow

### 7.1 Parameter Sweep

```morphogen
let sweep_results = optimize.param_sweep(
    params = {
        tube_diameter: range(20mm, 35mm, step=5mm),
        nozzle_area: range(50mm², 200mm², step=25mm²),
        tube_count: [4, 6, 8]
    },
    pipeline = JTubeFirePit,
    objective = smoke_index
)
```

**Output:** Table of (parameters, smoke_index, flow_rate, jet_temp, ...) for all combinations.

---

### 7.2 Objective Optimization

```morphogen
let best = optimize.search_best(
    results = sweep_results,
    objective = maximize(smoke_index),
    constraints = [
        v_exit < 15 m/s,  // Noise limit
        m_dot_total < 0.05 kg/s  // Flow limit
    ]
)
```

**Output:** Best design parameters.

---

### 7.3 Sensitivity Analysis

```morphogen
let sensitivity = optimize.sensitivity_analysis(
    base_config = best,
    perturb = [tube_diameter, nozzle_area],
    delta = 0.1  // ±10%
)
```

**Output:** How sensitive smoke_index is to each parameter.

---

## 8. Testing Strategy

### 8.1 Unit Tests (Per Domain)

**FluidNetwork:**
- Draft pressure matches analytical formula
- Network solve conserves mass
- Resistance matches Darcy-Weisbach

**ThermalODE:**
- 1D heat transfer converges with integrator order
- Wall temperature interpolation is smooth

**FluidJet:**
- Reynolds number matches analytical formula
- Jet visualization produces expected vector field

**CombustionLight:**
- Equivalence ratio matches stoichiometry
- Smoke reduction is monotonic with φ, T_zone

---

### 8.2 Integration Tests (Cross-Domain)

**Geometry → FluidNetwork:**
- Pipe anchors correctly define inlet/outlet
- Tube resistance scales with length

**FluidNetwork → ThermalODE:**
- Mass flow from network solve feeds thermal ODE
- Temperature rise is physically reasonable

**Full Pipeline:**
- End-to-end smoke_index is in expected range (0.5-0.9)
- Changing tube_diameter produces expected trends

---

### 8.3 Validation Tests (Physical Reality)

**Experimental Data:**
- Measure actual J-tube fire pit smoke output
- Compare Morphogen smoke_index to visual smoke assessment
- Validate jet velocity with anemometer

**CFD Comparison:**
- Run OpenFOAM simulation of same geometry
- Compare Morphogen jet_field to CFD velocity field
- Validate Morphogen's "CFD-lite" approximations

---

## 9. Future Extensions

### 9.1 Full CFD Integration

**Goal:** Replace "CFD-lite" jet visualization with actual Navier-Stokes solve.

**Approach:**
- Add `CFDDomain` with operators like `navier_stokes_step`
- Lower to existing CFD backends (OpenFOAM, SU2, etc.)
- Use Morphogen for **setup and post-processing**, CFD for **solving**

---

### 9.2 Structural Mechanics

**Goal:** Analyze thermal stresses in steel walls.

**Domains:**
- `StructuralDomain` — FEA for stress/strain
- `ThermalStructuralCoupling` — Thermal expansion → stress

**Use Case:** Ensure fire pit doesn't warp or crack under thermal loads.

---

### 9.3 Time-Varying Simulations

**Goal:** Model transient behavior (startup, wind gusts, fuel changes).

**Approach:**
- Add time-stepping to all domains
- Couple thermal mass (steel heating up) with flow
- Animate jet behavior over time

**Output:** Video of fire pit reaching steady-state.

---

### 9.4 Interactive Design Tool

**Goal:** Web-based parametric design tool powered by Morphogen.

**Tech Stack:**
- Frontend: React + Three.js (geometry preview)
- Backend: Morphogen pipeline (WASM or server-side)
- Interaction: Sliders for tube_diameter, nozzle_area, etc.
- Output: Real-time smoke_index, jet visualization

**User Experience:**
```
User adjusts tube_diameter slider
→ Morphogen recomputes pipeline (< 100ms)
→ 3D view updates jet visualization
→ Smoke index gauge updates
```

---

## 10. Summary

### What We've Shown

1. **Physical System** — J-tube fire pit as multi-physics engine
2. **Morphogen Pipeline** — 6 stages from geometry to smoke reduction
3. **New Domains** — FluidNetwork, ThermalODE, FluidJet, CombustionLight
4. **Operator Specs** — Detailed definitions for each domain
5. **Cross-Domain Flows** — How domains compose via references/anchors
6. **Design Optimization** — Parameter sweeps and objective search

### Why This Validates Morphogen

- **Operator graphs work for engineering** — Not just audio/graphics
- **Domains compose cleanly** — Geometry → Fluid → Thermal → Combustion
- **References/anchors are critical** — Physical connection points
- **Morphogen becomes a design platform** — Not just a runtime

### Next Steps

1. **Formalize domains** — Add FluidNetwork, ThermalODE, FluidJet, CombustionLight to Morphogen core
2. **Implement operators** — Build reference implementations (Python first)
3. **MLIR lowering** — Define dialects and lowering passes
4. **Example implementations** — J-tube fire pit, muffler, heat exchanger
5. **Documentation** — Update ../architecture/domain-architecture.md, operator registry

---

## References

### Related Documentation

- **[ADR-002: Cross-Domain Architectural Patterns](../adr/002-cross-domain-architectural-patterns.md)** — Reference systems, anchors, operator registries
- **[architecture/domain-architecture.md](../architecture/domain-architecture.md)** — Comprehensive domain vision
- **[specifications/operator-registry.md](../specifications/operator-registry.md)** — Operator metadata and layering
- **[specifications/coordinate-frames.md](../specifications/coordinate-frames.md)** — Frames and anchor system
- **[specifications/geometry.md](../specifications/geometry.md)** — Geometry domain (TiaCAD patterns)

### External References

- **J-Tube Fire Pit Design** — Original design concept
- **Darcy-Weisbach Equation** — Pipe flow resistance
- **Stack Effect** — Natural draft in chimneys
- **Secondary Combustion** — Smoke reduction via air injection

---

**Conclusion:**

The J-tube fire pit is not "just another use case" — it's a **proof point** that Morphogen's operator graph paradigm extends naturally from audio/graphics into **engineering physics**. By formalizing domains like FluidNetwork, ThermalODE, FluidJet, and CombustionLight, Morphogen becomes a **multi-physics design platform** capable of modeling fire pits, mufflers, engines, HVAC systems, and beyond.

**Morphogen is not a library. Morphogen is a platform.**

---

**End of Document**
