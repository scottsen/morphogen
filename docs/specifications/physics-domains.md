# SPEC: Physics Domains for Engineering Modeling

**Version:** 1.0
**Status:** Proposed
**Last Updated:** 2025-11-15

---

## Overview

This document specifies **four new physics domains** for Morphogen that enable multi-physics engineering modeling. These domains emerged from the J-tube fire pit design example and are generally applicable to thermal-fluid systems, combustion, HVAC, and mechanical engineering.

### Domains Covered

1. **FluidNetworkDomain** — 1D lumped flow networks (pipes, ducts, circuits)
2. **ThermalODEDomain** — 1D thermal modeling (heat transfer in pipes, rods, walls)
3. **FluidJetDomain** — Jet modeling and visualization
4. **CombustionLightDomain** — Simplified combustion metrics (no detailed chemistry)

### Design Principles

- **Deterministic by default** — All operators are repro or strict unless marked otherwise
- **Unit-safe** — All physical quantities have units (Pa, K, kg/s, W/(m·K), etc.)
- **Reference-based composition** — Uses Morphogen's anchor/frame system from GeometryDomain
- **Layered complexity** — Simple approximations first, can upgrade to full CFD/FEA later
- **Cross-domain integration** — Clean interfaces with Geometry, Fields, Visualization domains

---

## 1. FluidNetworkDomain

**Purpose:** Model 1D lumped flow networks (fluids treated as circuits)

**Why:** Essential for:
- Fire pits (J-tube air supply)
- HVAC systems (duct networks)
- Mufflers (exhaust flow paths)
- Intake manifolds (air distribution)
- Pneumatic circuits

**Analogy:** Fluid networks are to fluids what electrical circuits are to charge — resistances, pressures, flows.

---

### 1.1 Core Types

```morphogen
// Pipe geometry + flow properties
type Tube {
    geometry: Pipe,          // Geometric representation
    diameter: Length,        // Internal diameter
    length: Length,          // Total length
    roughness: Length,       // Surface roughness (for friction)
    bends: i32,              // Number of bends (minor losses)
    nozzle: Option<Nozzle>,  // Exit nozzle (if any)
    fluid: Fluid             // Fluid properties (air, water, etc.)
}

// Connection point in network
type Junction {
    position: Vec3<m>,
    connected_tubes: List<TubeRef>,
    boundary_condition: Option<BC>  // Pressure/flow BC
}

// Complete fluid network (graph)
type FlowNet {
    tubes: List<Tube>,
    junctions: List<Junction>,
    graph: Graph             // Connectivity graph
}

// Fluid properties
type Fluid {
    density: f32<kg/m³>,
    viscosity: f32<Pa·s>,
    specific_heat: f32<J/(kg·K)>
}

// References
type TubeRef = Ref<Tube>
type JunctionRef = Ref<Junction>
```

---

### 1.2 Operators

#### draft_pressure

**Compute draft pressure from stack effect (natural convection)**

```json
{
  "name": "draft_pressure",
  "domain": "fluid_network",
  "category": "pressure",
  "layer": 4,
  "description": "Compute draft pressure from stack effect in vertical chamber",

  "inputs": [
    {"name": "chamber_ref", "type": "Ref<Chamber>"},
    {"name": "height", "type": "f32<m>", "description": "Chimney/chamber height"},
    {"name": "T_amb", "type": "f32<K>", "description": "Ambient temperature"},
    {"name": "T_hot", "type": "f32<K>", "description": "Hot gas temperature"}
  ],

  "outputs": [
    {"name": "delta_p", "type": "f32<Pa>", "description": "Draft pressure"}
  ],

  "params": [
    {"name": "g", "type": "f32<m/s²>", "default": "9.81m/s²", "description": "Gravitational acceleration"}
  ],

  "determinism": "strict",
  "rate": "control",

  "numeric_properties": {
    "stable": true,
    "monotonic": true
  },

  "implementation": {
    "python": "morphogen.stdlib.fluid_network.draft_pressure",
    "formula": "ΔP = ρ·g·H·(1/T_amb - 1/T_hot)"
  },

  "tests": [
    {
      "name": "draft_pressure_standard",
      "inputs": {
        "height": "5m",
        "T_amb": "293K",
        "T_hot": "773K"
      },
      "expected_output": {"delta_p": "~10Pa"},
      "tolerance": 0.5
    }
  ]
}
```

---

#### tube_resistance

**Compute flow resistance for a tube (Darcy-Weisbach + minor losses)**

```json
{
  "name": "tube_resistance",
  "domain": "fluid_network",
  "category": "resistance",
  "layer": 4,
  "description": "Compute flow resistance (pressure drop per flow rate)",

  "inputs": [
    {"name": "tube_ref", "type": "TubeRef"},
    {"name": "roughness", "type": "f32<m>", "description": "Surface roughness"},
    {"name": "Re_guess", "type": "f32", "description": "Estimated Reynolds number"}
  ],

  "outputs": [
    {"name": "R_tube", "type": "f32<Pa·s²/kg²>", "description": "Flow resistance"}
  ],

  "determinism": "strict",
  "rate": "control",

  "numeric_properties": {
    "positive": true,
    "monotonic_with_length": true
  },

  "implementation": {
    "python": "morphogen.stdlib.fluid_network.tube_resistance",
    "formula": "R = (f·L/D + K_bends)·(1/(2·ρ·A²))"
  },

  "notes": [
    "Uses Colebrook-White equation for friction factor",
    "Includes minor losses (bends, contractions, expansions)",
    "Assumes turbulent flow (Re > 2300)"
  ]
}
```

---

#### network_solve

**Solve fluid network for flows and pressures (Modified Nodal Analysis)**

```json
{
  "name": "network_solve",
  "domain": "fluid_network",
  "category": "solver",
  "layer": 4,
  "description": "Solve fluid network using Modified Nodal Analysis (MNA)",

  "inputs": [
    {"name": "net", "type": "FlowNet"},
    {"name": "delta_p", "type": "f32<Pa>", "description": "Driving pressure"}
  ],

  "outputs": [
    {"name": "flows", "type": "Array<f32<kg/s>>", "description": "Mass flow per tube"},
    {"name": "pressures", "type": "Array<f32<Pa>>", "description": "Pressure per junction"}
  ],

  "params": [
    {"name": "solver", "type": "Enum", "default": "cg", "enum": ["direct", "cg", "gmres"]},
    {"name": "tolerance", "type": "f32", "default": "1e-6"}
  ],

  "determinism": "repro",
  "rate": "control",

  "numeric_properties": {
    "linear": true,
    "sparse": true,
    "positive_definite": true
  },

  "lowering_hints": {
    "dialect": "morphogen.fluid",
    "template": "mna_fluid_solver",
    "parallelize": true
  },

  "implementation": {
    "python": "morphogen.stdlib.fluid_network.network_solve",
    "algorithm": "Modified Nodal Analysis (MNA)"
  },

  "notes": [
    "Constructs system matrix from tube resistances",
    "Solves Ax = b for junction pressures",
    "Back-substitutes for tube flows",
    "Assumes incompressible flow"
  ]
}
```

---

### 1.3 Use Cases

- **Fire pits** — J-tube secondary air supply
- **HVAC** — Air duct networks, pressure balancing
- **Mufflers** — Exhaust flow paths, back-pressure
- **Intake manifolds** — Air distribution to cylinders
- **Pneumatic circuits** — Compressed air systems
- **Plumbing** — Water supply networks (with appropriate fluid properties)

---

## 2. ThermalODEDomain

**Purpose:** 1D thermal modeling (heat transfer in tubes, rods, walls)

**Why:** Essential for:
- Fire pits (air preheating in J-tubes)
- Heat exchangers (tube-side heat transfer)
- Hotends (3D printer nozzle heating)
- Battery thermal management (lumped capacity)
- Heat pipes (thermal conduction)

**Approach:** Solve 1D ODEs for temperature evolution along spatial coordinates or in time.

---

### 2.1 Core Types

```morphogen
// 1D thermal segment (tube, rod, wall)
type ThermalSegment {
    length: Length,
    diameter: Length,
    wall_thickness: Length,
    conductivity: f32<W/(m·K)>,
    emissivity: f32,
    fluid: Option<Fluid>  // If fluid flows through
}

// Wall temperature profile (boundary condition)
type WallTempModel {
    profile: Fn(x: f32<m>) -> f32<K>
}

// Temperature distribution along segment
type ThermalProfile {
    temperatures: Array<f32<K>>,
    positions: Array<f32<m>>
}
```

---

### 2.2 Operators

#### wall_temp_model

**Estimate wall temperature profile based on fire state**

```json
{
  "name": "wall_temp_model",
  "domain": "thermal_ode",
  "category": "boundary_condition",
  "layer": 4,
  "description": "Estimate wall temperature profile along tube",

  "inputs": [
    {"name": "fire_state", "type": "Ref<FireState>"},
    {"name": "tube", "type": "TubeRef"},
    {"name": "position", "type": "Anchor", "description": "Tube section (e.g., heated_section)"}
  ],

  "outputs": [
    {"name": "wall_temp", "type": "WallTempModel"}
  ],

  "params": [
    {"name": "model", "type": "Enum", "default": "linear", "enum": ["constant", "linear", "exponential"]}
  ],

  "determinism": "repro",
  "rate": "control",

  "implementation": {
    "python": "morphogen.stdlib.thermal_ode.wall_temp_model"
  },

  "notes": [
    "Simple approximation (not full FEA)",
    "Linear: T_wall(x) = T_hot - (T_hot - T_amb)·(x/L)",
    "Exponential: T_wall(x) = T_amb + (T_hot - T_amb)·exp(-x/L_char)"
  ]
}
```

---

#### heat_transfer_1D

**Solve 1D heat transfer ODE along tube**

```json
{
  "name": "heat_transfer_1D",
  "domain": "thermal_ode",
  "category": "heat_transfer",
  "layer": 4,
  "description": "Solve 1D heat transfer ODE for fluid heating in tube",

  "inputs": [
    {"name": "segment_ref", "type": "Ref<ThermalSegment>"},
    {"name": "m_dot", "type": "f32<kg/s>", "description": "Mass flow rate"},
    {"name": "T_in", "type": "f32<K>", "description": "Inlet temperature"},
    {"name": "wall_temp_model", "type": "WallTempModel"}
  ],

  "outputs": [
    {"name": "T_out", "type": "f32<K>", "description": "Outlet temperature"}
  ],

  "params": [
    {"name": "integrator", "type": "Enum", "default": "rk4", "enum": ["euler", "rk2", "rk4"]},
    {"name": "steps", "type": "i32", "default": "100", "description": "Integration steps"}
  ],

  "determinism": "repro",
  "rate": "control",

  "numeric_properties": {
    "order": 4,  // RK4
    "stable": true
  },

  "implementation": {
    "python": "morphogen.stdlib.thermal_ode.heat_transfer_1D",
    "formula": "m_dot·c_p·dT/dx = h·A_s·(T_wall(x) - T_air)"
  },

  "tests": [
    {
      "name": "heat_transfer_constant_wall",
      "inputs": {
        "m_dot": "0.01kg/s",
        "T_in": "293K",
        "wall_temp": "constant(600K)",
        "length": "0.3m",
        "diameter": "0.025m"
      },
      "expected_output": {"T_out": "~450K"},
      "tolerance": 10
    }
  ]
}
```

---

#### lumped_capacity

**Lumped thermal capacity model (0D transient)**

```json
{
  "name": "lumped_capacity",
  "domain": "thermal_ode",
  "category": "transient",
  "layer": 4,
  "description": "Lumped thermal capacity model for transient heating",

  "inputs": [
    {"name": "body_ref", "type": "Ref<Body>"},
    {"name": "heat_input", "type": "f32<W>", "description": "Heat input power"}
  ],

  "outputs": [
    {"name": "T_body", "type": "Stream<f32<K>,time>"}
  ],

  "params": [
    {"name": "dt", "type": "Rate", "default": "10ms"},
    {"name": "T_initial", "type": "f32<K>", "default": "293K"}
  ],

  "determinism": "repro",
  "rate": "control",

  "numeric_properties": {
    "first_order": true,
    "stable": true
  },

  "implementation": {
    "python": "morphogen.stdlib.thermal_ode.lumped_capacity",
    "formula": "m·c_p·dT/dt = Q_in - h·A·(T - T_amb)"
  }
}
```

---

### 2.3 Use Cases

- **Fire pits** — Air preheating in J-tubes
- **Heat exchangers** — Tube-side heat transfer
- **Hotends** — 3D printer nozzle heating
- **Battery packs** — Thermal management
- **Heat pipes** — Thermal conduction
- **Radiators** — Cooling systems

---

## 3. FluidJetDomain

**Purpose:** Model jet exits and their interaction with surrounding flow

**Why:** Essential for:
- Fire pits (secondary air jets into flame)
- Burners (fuel/air injection)
- Rocket nozzles (thrust calculation)
- Spray systems (atomization)
- Ventilation (jet mixing)

**Approach:** Simplified jet modeling (momentum, entrainment, mixing) without full CFD.

---

### 3.1 Core Types

```morphogen
// Single jet
type Jet {
    flow: f32<kg/s>,        // Mass flow rate
    velocity: f32<m/s>,     // Exit velocity
    temperature: f32<K>,    // Exit temperature
    direction: Vec3,        // Jet direction (unit vector)
    area: f32<m²>,          // Nozzle area
    position: Vec3<m>       // Jet origin
}

// Collection of jets
type JetArray {
    jets: List<Jet>,
    count: i32
}
```

---

### 3.2 Operators

#### jet_from_tube

**Create jet from tube exit conditions**

```json
{
  "name": "jet_from_tube",
  "domain": "fluid_jet",
  "category": "jet",
  "layer": 4,
  "description": "Create jet from tube exit flow conditions",

  "inputs": [
    {"name": "tube_ref", "type": "TubeRef"},
    {"name": "m_dot", "type": "f32<kg/s>", "description": "Mass flow rate"},
    {"name": "T_out", "type": "f32<K>", "description": "Exit temperature"}
  ],

  "outputs": [
    {"name": "jet", "type": "Jet"}
  ],

  "determinism": "strict",
  "rate": "control",

  "implementation": {
    "python": "morphogen.stdlib.fluid_jet.jet_from_tube",
    "formula": "v = m_dot / (ρ·A)"
  },

  "notes": [
    "Uses tube outlet anchor for position",
    "Direction from tube geometry",
    "Density from temperature (ideal gas)"
  ]
}
```

---

#### jet_reynolds

**Compute jet Reynolds number**

```json
{
  "name": "jet_reynolds",
  "domain": "fluid_jet",
  "category": "dimensionless",
  "layer": 4,
  "description": "Compute jet Reynolds number",

  "inputs": [
    {"name": "jet", "type": "Jet"}
  ],

  "outputs": [
    {"name": "Re", "type": "f32"}
  ],

  "determinism": "strict",
  "rate": "control",

  "implementation": {
    "python": "morphogen.stdlib.fluid_jet.jet_reynolds",
    "formula": "Re = ρ·v·D/μ"
  }
}
```

---

#### jet_entrainment

**Estimate jet entrainment and mixing with ambient flow**

```json
{
  "name": "jet_entrainment",
  "domain": "fluid_jet",
  "category": "mixing",
  "layer": 4,
  "description": "Estimate jet entrainment and mixing factor",

  "inputs": [
    {"name": "jet", "type": "Jet"},
    {"name": "plume_ref", "type": "Ref<Plume>", "description": "Ambient plume/flow"}
  ],

  "outputs": [
    {"name": "mixing_factor", "type": "f32", "description": "Mixing effectiveness (0-1)"}
  ],

  "params": [
    {"name": "model", "type": "Enum", "default": "empirical", "enum": ["empirical", "momentum_ratio"]}
  ],

  "determinism": "repro",
  "rate": "control",

  "implementation": {
    "python": "morphogen.stdlib.fluid_jet.jet_entrainment"
  },

  "notes": [
    "Empirical correlations from jet mixing literature",
    "Momentum ratio model: mixing ~ (ρ_jet·v_jet²) / (ρ_plume·v_plume²)"
  ]
}
```

---

#### jet_field

**Generate vector field visualization of jets (CFD-lite)**

```json
{
  "name": "jet_field",
  "domain": "fluid_jet",
  "category": "visualization",
  "layer": 6,
  "description": "Generate vector field visualization of jet array",

  "inputs": [
    {"name": "jet_array", "type": "JetArray"},
    {"name": "frame_ref", "type": "FrameRef", "description": "Reference frame for visualization"}
  ],

  "outputs": [
    {"name": "field", "type": "Field2D<Vec2>"}
  ],

  "params": [
    {"name": "resolution", "type": "Vec2<i32>", "default": "(512, 512)"},
    {"name": "decay", "type": "f32", "default": "0.1", "description": "Jet decay rate"}
  ],

  "determinism": "repro",
  "rate": "control",

  "implementation": {
    "python": "morphogen.stdlib.fluid_jet.jet_field"
  },

  "notes": [
    "Simple superposition of Gaussian jet profiles",
    "Not full CFD — for quick visualization only"
  ]
}
```

---

### 3.3 Use Cases

- **Fire pits** — Secondary air jets into flame
- **Burners** — Fuel/air injection
- **Rocket nozzles** — Thrust calculation
- **Spray systems** — Atomization
- **Ventilation** — Jet mixing

---

## 4. CombustionLightDomain

**Purpose:** Simplified combustion metrics (no detailed chemistry)

**Why:** Essential for:
- Fire pits (smoke reduction estimation)
- Burners (combustion quality)
- Mufflers (exhaust composition)
- Engine exhaust (emissions estimation)

**Approach:** Approximate combustion metrics using equivalence ratio, temperature, and mixing — no detailed chemical kinetics.

---

### 4.1 Core Types

```morphogen
// Mixture composition
type MixtureState {
    fuel_rate: f32<kg/s>,
    air_rate: f32<kg/s>,
    equivalence_ratio: f32
}

// Combustion zone properties
type CombustionZone {
    temperature: f32<K>,
    pressure: f32<Pa>,
    residence_time: f32<s>
}

// Smoke/emissions index
type SmokeIndex {
    value: f32,              // 0 = very smoky, 1 = clean
    reduction_factor: f32    // Relative to baseline
}
```

---

### 4.2 Operators

#### equivalence_ratio

**Compute equivalence ratio (φ = actual/stoichiometric)**

```json
{
  "name": "equivalence_ratio",
  "domain": "combustion_light",
  "category": "mixture",
  "layer": 4,
  "description": "Compute equivalence ratio (fuel-air ratio / stoichiometric ratio)",

  "inputs": [
    {"name": "fuel_rate", "type": "f32<kg/s>"},
    {"name": "air_rate", "type": "f32<kg/s>"}
  ],

  "outputs": [
    {"name": "phi", "type": "f32"}
  ],

  "params": [
    {"name": "stoichiometric_ratio", "type": "f32", "default": "0.0676", "description": "For gasoline-air"}
  ],

  "determinism": "strict",
  "rate": "control",

  "implementation": {
    "python": "morphogen.stdlib.combustion_light.equivalence_ratio",
    "formula": "φ = (fuel/air) / (fuel/air)_stoich"
  },

  "notes": [
    "φ < 1: lean (excess air)",
    "φ = 1: stoichiometric",
    "φ > 1: rich (excess fuel)"
  ]
}
```

---

#### zone_temperature

**Estimate combustion zone temperature**

```json
{
  "name": "zone_temperature",
  "domain": "combustion_light",
  "category": "temperature",
  "layer": 4,
  "description": "Estimate combustion zone temperature from fire state and jet heating",

  "inputs": [
    {"name": "fire_state", "type": "Ref<FireState>"},
    {"name": "jet_info", "type": "JetArray", "description": "Secondary air jets"}
  ],

  "outputs": [
    {"name": "T_zone", "type": "f32<K>"}
  ],

  "params": [
    {"name": "model", "type": "Enum", "default": "energy_balance", "enum": ["energy_balance", "empirical"]}
  ],

  "determinism": "repro",
  "rate": "control",

  "implementation": {
    "python": "morphogen.stdlib.combustion_light.zone_temperature"
  },

  "notes": [
    "Energy balance: T_zone from adiabatic flame temp + jet cooling",
    "Empirical: Correlation from experimental data"
  ]
}
```

---

#### smoke_reduction

**Estimate smoke reduction effectiveness**

```json
{
  "name": "smoke_reduction",
  "domain": "combustion_light",
  "category": "emissions",
  "layer": 4,
  "description": "Estimate smoke reduction effectiveness from secondary air",

  "inputs": [
    {"name": "phi", "type": "f32", "description": "Equivalence ratio"},
    {"name": "T_zone", "type": "f32<K>", "description": "Combustion zone temperature"},
    {"name": "mixing_factor", "type": "f32", "description": "Jet mixing effectiveness"},
    {"name": "residence_time", "type": "f32<s>", "description": "Time in hot zone"}
  ],

  "outputs": [
    {"name": "smoke_index", "type": "SmokeIndex"}
  ],

  "params": [
    {"name": "model", "type": "Enum", "default": "empirical", "enum": ["empirical", "kinetic"]}
  ],

  "determinism": "repro",
  "rate": "control",

  "implementation": {
    "python": "morphogen.stdlib.combustion_light.smoke_reduction"
  },

  "notes": [
    "Empirical: Simple correlation from literature",
    "Kinetic: Approximate reaction rate (not detailed chemistry)",
    "Factors: φ → 1 (better), T_zone ↑ (better), mixing ↑ (better), residence ↑ (better)"
  ]
}
```

---

### 4.3 Use Cases

- **Fire pits** — Smoke reduction from secondary air
- **Burners** — Combustion quality estimation
- **Mufflers** — Exhaust composition
- **Engine exhaust** — Emissions estimation
- **Incinerators** — Burn efficiency

---

## 5. Cross-Domain Integration

### 5.1 Integration Patterns

**Geometry → FluidNetwork:**
```morphogen
let tube = geom.pipe(centerline, diameter, thickness)
let resistance = fluid_net.tube_resistance(geometry=tube, ...)
```

**FluidNetwork → ThermalODE:**
```morphogen
let flow_solution = fluid_net.network_solve(...)
let T_out = thermal.heat_transfer_1D(m_dot=flow_solution.mass_flow[i], ...)
```

**ThermalODE → FluidJet:**
```morphogen
let T_jet = thermal.heat_transfer_1D(...)
let jet = fluid_jet.from_tube(T_out=T_jet, ...)
```

**FluidJet → CombustionLight:**
```morphogen
let jets = [...]
let T_zone = combustion.zone_temperature(jet_info=jets)
```

---

### 5.2 Coupling Variables

| Source Domain | Target Domain | Coupling Variable | Type |
|---------------|---------------|-------------------|------|
| Geometry | FluidNetwork | Pipe geometry | `Pipe` |
| FluidNetwork | ThermalODE | Mass flow rate | `f32<kg/s>` |
| ThermalODE | FluidJet | Exit temperature | `f32<K>` |
| FluidJet | CombustionLight | Jet array | `JetArray` |
| All | Visualization | Any output | Various |

---

## 6. Determinism & Profiles

### 6.1 Determinism Tiers

| Domain | Operators | Determinism | Rationale |
|--------|-----------|-------------|-----------|
| FluidNetwork | `draft_pressure` | Strict | Analytical formula |
| FluidNetwork | `tube_resistance` | Strict | Deterministic friction factor |
| FluidNetwork | `network_solve` | Repro | Iterative linear solver |
| ThermalODE | `heat_transfer_1D` | Repro | ODE integrator |
| FluidJet | `jet_from_tube` | Strict | Simple calculation |
| FluidJet | `jet_entrainment` | Repro | Empirical correlations |
| CombustionLight | All | Repro | Approximate models |

---

### 6.2 Profile Overrides

**Example: network_solve**

```json
{
  "profile_overrides": {
    "strict": {
      "solver": "direct",
      "precision": "f64"
    },
    "repro": {
      "solver": "cg",
      "precision": "f32",
      "tolerance": 1e-6
    },
    "live": {
      "solver": "cg",
      "precision": "f32",
      "tolerance": 1e-4
    }
  }
}
```

---

## 7. MLIR Lowering Strategy

### 7.1 Dialect Design

**morphogen.fluid dialect:**
- `fluid.draft_pressure` — Map to simple arithmetic
- `fluid.tube_resistance` — Inline or call runtime function
- `fluid.network_solve` — Lower to sparse linear algebra (linalg.sparse)

**morphogen.thermal dialect:**
- `thermal.heat_transfer_1D` — Lower to ODE integrator (loop + vectorized ops)
- `thermal.lumped_capacity` — Lower to recurrence relation

**morphogen.jet dialect:**
- `jet.from_tube` — Inline calculations
- `jet.field` — Lower to field ops (morphogen.field dialect)

**morphogen.combustion dialect:**
- `combustion.equivalence_ratio` — Inline
- `combustion.smoke_reduction` — Call runtime function (empirical model)

---

### 7.2 Example Lowering

**network_solve → Sparse Linear Algebra:**

```mlir
// High-level
%flows, %pressures = fluid.network_solve %net, %delta_p

// Lowered to linalg
%A = sparse.construct_matrix %net  // System matrix
%b = dense.construct_rhs %delta_p  // RHS vector
%x = linalg.sparse.solve %A, %b    // Solve Ax = b
%pressures = dense.extract %x
%flows = dense.back_substitute %A, %x
```

---

## 8. Testing Strategy

### 8.1 Unit Tests

**Per operator:**
- Analytical validation (where formulas exist)
- Regression tests (golden outputs)
- Edge cases (zero flow, extreme temps, etc.)

**Example: draft_pressure**
```python
def test_draft_pressure_analytical():
    delta_p = draft_pressure(height=5, T_amb=293, T_hot=773)
    expected = 1.2 * 9.81 * 5 * (1/293 - 1/773)  # ≈ 10 Pa
    assert abs(delta_p - expected) < 0.5
```

---

### 8.2 Integration Tests

**Cross-domain pipelines:**
- Geometry → FluidNetwork → ThermalODE → FluidJet
- Validate coupling variables are physically reasonable

**Example: J-tube pipeline**
```python
def test_j_tube_pipeline():
    # Setup geometry, run pipeline
    result = run_j_tube_simulation(tube_diameter=25mm, nozzle_area=100mm²)

    # Validate outputs
    assert 5 < result.delta_p < 15  # Pa
    assert 0.005 < result.m_dot < 0.02  # kg/s per tube
    assert 400 < result.T_jet < 700  # K
    assert 0.5 < result.smoke_index < 1.0
```

---

### 8.3 Validation Tests

**Compare to experimental data or high-fidelity simulations:**
- Measure actual fire pit smoke output → compare to smoke_index
- Run OpenFOAM CFD → compare to jet_field
- Measure jet velocity with anemometer → compare to v_exit

---

## 9. Implementation Roadmap

### Phase 1: Python Prototypes (v0.9)
- [ ] Implement all operators in Python
- [ ] Unit tests for each operator
- [ ] Integration tests (cross-domain pipelines)

### Phase 2: MLIR Lowering (v0.10)
- [ ] Define morphogen.fluid, morphogen.thermal, morphogen.jet, morphogen.combustion dialects
- [ ] Implement lowering passes
- [ ] Validate lowered code vs Python reference

### Phase 3: Optimization (v0.11)
- [ ] Domain-specific optimization passes
- [ ] GPU lowering for parallelizable ops
- [ ] Profile-specific tuning

### Phase 4: Documentation & Examples (v0.12)
- [ ] Complete operator documentation
- [ ] J-tube fire pit tutorial
- [ ] Additional examples (muffler, heat exchanger, burner)

---

## 10. Summary

### What We've Specified

1. **FluidNetworkDomain** — Lumped flow networks (draft, resistance, network solve)
2. **ThermalODEDomain** — 1D heat transfer (wall temp, heat transfer ODE, lumped capacity)
3. **FluidJetDomain** — Jet modeling (jet from tube, Reynolds, entrainment, visualization)
4. **CombustionLightDomain** — Simplified combustion (equivalence ratio, zone temp, smoke reduction)

### Why This Matters

- **Validates Morphogen for engineering** — Not just audio/graphics
- **Enables multi-physics design** — Fire pits, mufflers, HVAC, burners
- **Demonstrates cross-domain composition** — Geometry → Fluid → Thermal → Combustion
- **Proves reference/anchor system works** — Physical connection points

### Next Steps

1. **Update ../architecture/domain-architecture.md** — Add these 4 domains to Next-Wave section
2. **Implement Python prototypes** — Reference implementations
3. **Create J-tube tutorial** — End-to-end example
4. **Formalize operator registry** — Add to operator-registry.md

---

## References

- **[../examples/j-tube-firepit-multiphysics.md](../examples/j-tube-firepit-multiphysics.md)** — Complete design example
- **[../adr/002-cross-domain-architectural-patterns.md](../adr/002-cross-domain-architectural-patterns.md)** — Unified architecture
- **[operator-registry.md](operator-registry.md)** — Operator registry format
- **[coordinate-frames.md](coordinate-frames.md)** — Frames and anchors
- **[../architecture/domain-architecture.md](../architecture/domain-architecture.md)** — Comprehensive domain vision

---

**End of Document**
