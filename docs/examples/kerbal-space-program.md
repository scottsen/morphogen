# Kerbal Space Program Physics: A Multi-Domain Orbital Simulation Example for Morphogen

**Version:** 1.0
**Status:** Design Document
**Last Updated:** 2025-11-15
**Authors:** Morphogen Architecture Team

---

## Overview

Kerbal Space Program (KSP) is not just a game — it's a brilliant example of multi-domain physics simulation that maps perfectly onto Morphogen's operator graph architecture. This document demonstrates how Morphogen can model KSP-style physics (orbital mechanics, aerodynamics, rocket staging, part assembly, and more) and potentially become a framework for building similar simulations.

### What This Document Demonstrates

1. **Game-to-Simulation Mapping** — How KSP's gameplay mechanics map to Morphogen domains
2. **Multi-Domain Physics Pipeline** — Orbital, aerodynamic, propulsion, and structural domains working together
3. **Part-Based Assembly System** — Component composition using Morphogen's graph architecture
4. **Real-Time Simulation** — Physics stepping with GPU acceleration options
5. **Cross-Domain Integration** — Geometry, physics, audio, and visualization working in concert
6. **Educational Platform** — Morphogen as a tool for teaching orbital mechanics and aerospace engineering

### Why This Matters

KSP demonstrates that Morphogen's vision extends beyond niche engineering:
- It shows Morphogen can handle **real-time game physics**
- It validates the **part-assembly paradigm** for complex systems
- It proves **multi-physics integration** at interactive framerates
- It opens educational and scientific computing use cases
- It demonstrates **level-of-detail switching** (simplified vs. N-body gravity, etc.)

---

## 1. Kerbal Space Program: The Physics Challenge

### 1.1 What Makes KSP Special?

KSP is famous for making orbital mechanics **accessible and fun**:
- Players design rockets from modular parts
- Launch vehicles into orbit using realistic physics
- Navigate using delta-v budgets and transfer orbits
- Land on planets and moons with varying gravity and atmospheres
- Deal with staging, fuel management, and structural limitations

**The Physics Stack:**
```
Parts (engines, tanks, structure)
  ↓
Assembly (staging, connections, mass distribution)
  ↓
Forces (thrust, drag, gravity, lift)
  ↓
Integration (velocity, position updates)
  ↓
Orbital mechanics (Kepler, patched conics)
  ↓
State transitions (SOI changes, atmosphere entry/exit)
```

This is **exactly** an operator graph problem.

---

### 1.2 The Morphogen Opportunity

**Can Morphogen simulate KSP-style physics?**
👉 **Yes** — and potentially better than Unity physics.

**Can Morphogen become a framework for aerospace simulations?**
👉 **Absolutely** — with clear advantages:
- Declarative YAML assembly
- Modular operator domains
- GPU acceleration where needed
- Level-of-detail physics switching
- Integration with TiaCAD for part geometry
- Integration with AudioDomain for engine/aerodynamic sounds
- Perfect for education and research

---

## 2. Domain Mapping: KSP → Morphogen

Let's map KSP's systems onto Morphogen domains:

### 2.1 Orbital Mechanics Domain

**KSP Feature:** Patched conics orbital prediction

**Morphogen Domain:** `OrbitalMechanicsDomain`

**What It Does:**
- Solve Kepler's equation for orbital position
- Compute orbital elements (a, e, i, Ω, ω, ν)
- Predict transfer orbits (Hohmann, bi-elliptic)
- Handle sphere-of-influence (SOI) transitions
- Lambert's problem for rendezvous
- Calculate delta-v requirements

**Key Operators:**
```morphogen
orbit.kepler_solve(M, e) -> E           // Eccentric anomaly
orbit.state_to_elements(r, v, μ) -> OrbitalElements
orbit.elements_to_state(elements, t) -> (r, v)
orbit.period(a, μ) -> T
orbit.apoapsis(r, v, μ) -> r_ap
orbit.periapsis(r, v, μ) -> r_pe
orbit.hohmann_transfer(r1, r2, μ) -> (Δv1, Δv2, t_transfer)
orbit.lambert_solve(r1, r2, t_transfer, μ) -> (v1, v2)
orbit.soi_transition(state, body_from, body_to) -> state_new
```

**Physics:**
- **Keplerian orbits** — Two-body problem, analytical solution
- **Patched conics** — Simplified multi-body (one SOI at a time)
- **Optional N-body** — Full gravitational interactions (expensive)

---

### 2.2 Aerodynamics Domain

**KSP Feature:** Drag cubes + occlusion model

**Morphogen Domain:** `AerodynamicsDomain`

**What It Does:**
- Compute drag based on shape and occlusion
- Calculate lift from wings and control surfaces
- Model atmospheric density/pressure/temperature
- Compute entry heating
- Handle supersonic/hypersonic regimes

**Key Operators:**
```morphogen
aero.atmosphere(altitude, body) -> (ρ, P, T)
aero.drag(velocity, area, Cd, ρ) -> F_drag
aero.drag_cube(parts, velocity, ρ) -> F_drag
aero.lift(wing_ref, AoA, velocity, ρ) -> F_lift
aero.occlusion(parts, flow_direction) -> occlusion_factor
aero.entry_heat(velocity, ρ) -> Q_heat
aero.mach_number(velocity, T) -> Mach
aero.shock_heating(Mach, ρ) -> Q_shock
```

**Models:**
- **Drag cube** — Pre-computed drag in 6 directions per part
- **Occlusion** — Parts behind other parts contribute less drag
- **Lift** — Wing surfaces generate lift based on AoA
- **Heating** — Atmospheric entry generates heat flux
- **Optional CFD-lite** — More detailed aerodynamics for research

---

### 2.3 Rocket Equation Domain

**KSP Feature:** Delta-v calculations, staging, Isp

**Morphogen Domain:** `RocketEquationDomain`

**What It Does:**
- Compute delta-v from Tsiolkovsky equation
- Model staging (wet/dry mass transitions)
- Calculate thrust-to-weight ratio (TWR)
- Handle variable Isp (vacuum vs. sea level)
- Optimize staging sequences

**Key Operators:**
```morphogen
rocket.delta_v(m_wet, m_dry, Isp, g0) -> Δv
rocket.delta_v_stages(stages[]) -> Δv_total
rocket.mass_flow(thrust, Isp, g0) -> ṁ
rocket.burn_time(m_fuel, thrust, Isp, g0) -> t_burn
rocket.twr(thrust, mass, g_local) -> TWR
rocket.isp_altitude(Isp_vac, Isp_sl, P_amb, P_exit) -> Isp_eff
rocket.staging_optimize(parts[], target_Δv) -> optimal_staging
rocket.thrust_curve(engine_type, throttle, altitude) -> thrust
```

**Physics:**
- **Tsiolkovsky equation:** Δv = Isp g₀ ln(m₀/m_f)
- **Staging:** Each stage separation increases effective Δv
- **Variable Isp:** Engines perform differently in vacuum vs. atmosphere
- **Thrust vectoring:** Gimbal angles affect control authority

---

### 2.4 Parts & Assembly Domain

**KSP Feature:** Modular rocket construction

**Morphogen Domain:** `PartsAssemblyDomain`

**What It Does:**
- Define part types (engines, tanks, structure, etc.)
- Assemble parts into vessels
- Compute total mass, drag, thrust
- Model attachment nodes and staging groups
- Handle part failures and decouplers

**Key Types:**
```morphogen
type Part {
    mass: f32<kg>,
    cost: f32,
    attach_nodes: List<AttachNode>,
    properties: PartProperties
}

enum PartType {
    Engine { thrust, Isp_vac, Isp_sl, gimbal_range },
    FuelTank { fuel_capacity, fuel_type, dry_mass },
    AeroSurface { lift_coef, drag_coef, area },
    ReactionWheel { torque_max },
    Battery { capacity },
    Decoupler { ejection_force },
    Structural { strength }
}

type Vessel {
    parts: List<Part>,
    stages: List<Stage>,
    mass_total: f32<kg>,
    center_of_mass: Vec3<m>,
    moment_of_inertia: Mat3<kg·m²>
}

type Stage {
    parts: List<PartRef>,
    activation_group: i32
}
```

**Key Operators:**
```morphogen
assembly.create_vessel(parts[], connections[]) -> Vessel
assembly.compute_mass(vessel) -> (mass_total, mass_fuel, mass_dry)
assembly.compute_com(vessel) -> center_of_mass
assembly.compute_moi(vessel) -> moment_of_inertia
assembly.stage(vessel, stage_num) -> vessel_after_staging
assembly.part_failures(vessel, stress, heat, g_force) -> failed_parts[]
assembly.structural_integrity(vessel, forces) -> stress_map
```

**Integration with TiaCAD:**
```morphogen
// Each part has geometry from TiaCAD
let engine = tiacad.load("kerbal_parts/engine_mainsail.yaml")
let tank = tiacad.load("kerbal_parts/fuel_tank_jumbo.yaml")

// Attach parts using anchors
let vessel = assembly.create()
vessel.attach(tank, anchor="bottom")
vessel.attach(engine, tank.anchor("bottom_node"))
```

---

### 2.5 Physics Step Domain

**KSP Feature:** Real-time physics integration

**Morphogen Domain:** `PhysicsIntegrationDomain`

**What It Does:**
- Integrate forces → accelerations → velocities → positions
- Handle rigid body rotation (quaternions)
- Compute torques from thrust vectoring, aerodynamics, RCS
- Apply gravity (point source or N-body)
- Update state at fixed timestep (0.02s typical)

**Key Operators:**
```morphogen
physics.gravity_accel(position, bodies[]) -> a_gravity
physics.sum_forces(vessel, state, control_input) -> F_total
physics.sum_torques(vessel, state, control_input) -> τ_total
physics.integrate_translational(state, F, mass, dt) -> state_new
physics.integrate_rotational(attitude, τ, I, dt) -> attitude_new
physics.rk4_step(state, forces_fn, dt) -> state_new
physics.detect_collisions(vessel, terrain) -> collision_events
```

**Integrators:**
- **Explicit Euler** — Fastest, least accurate
- **RK2** — Moderate accuracy
- **RK4** — Good accuracy (KSP default)
- **Symplectic** — Energy-conserving for long orbital sims
- **Verlet** — Position-based (good for constraints)

**GPU Acceleration:**
```morphogen
// Option to run on GPU for N-body or particle systems
physics.integrate_gpu(states[], forces_fn, dt, backend="cuda") -> states_new[]
```

---

### 2.6 Destruction & Failure Domain

**KSP Feature:** Parts explode under excessive stress

**Morphogen Domain:** `FailureMechanicsDomain`

**What It Does:**
- Monitor G-forces, heat, pressure, torque
- Trigger failures when limits exceeded
- Model explosions and debris
- Handle cascading failures

**Key Operators:**
```morphogen
failure.g_force_limit(part_type) -> g_max
failure.thermal_limit(part_type) -> T_max
failure.pressure_limit(part_type) -> q_max  // Dynamic pressure
failure.check_failure(part, state) -> failure_reason?
failure.explode(part) -> debris[]
failure.cascade_check(vessel, failed_parts[]) -> additional_failures[]
```

**Failure Modes:**
- **G-force** — Excessive acceleration
- **Thermal** — Overheating from entry or engines
- **Aerodynamic** — Dynamic pressure (q) exceeds limit
- **Structural** — Torque or bending moment too high

---

### 2.7 Propulsion Experiments Domain

**KSP Feature:** Various engine types

**Morphogen Extension:** Real propulsion models from J-tube experiments!

**What It Enables:**
- **Liquid engines** — RP-1/LOX, LH2/LOX
- **Solid rockets** — Burn curves, thrust tailing
- **Hybrid motors** — Fuel regression rate
- **Electric propulsion** — Ion drives, Hall thrusters
- **Pulse jets** — Valveless combustion (J-tube inspired!)
- **Ramjets/Scramjets** — Air-breathing hypersonics
- **Acoustic modeling** — Engine sounds from AudioDomain

**Integration:**
```morphogen
// Define engine using combustion physics
let engine = propulsion.liquid_engine(
    propellants = ["RP-1", "LOX"],
    mixture_ratio = 2.56,
    chamber_pressure = 10 MPa,
    nozzle_expansion = 15,
    combustion_model = combustion.adiabatic_flame()
)

// Or use J-tube inspired pulse jet
let pulse_jet = propulsion.pulse_jet(
    tube_geometry = jtube_geometry,
    fuel_flow = 0.05 kg/s,
    combustion = combustion.periodic(freq=50 Hz)
)

// Acoustic simulation
let engine_sound = audio.engine_noise(
    thrust = engine.thrust,
    rpm = engine.turbopump_speed,
    spectral_model = "broadband + harmonics"
)
```

---

## 3. Complete KSP-Style Simulation Pipeline

Here's how all domains work together:

```morphogen
scene KerbalLaunch {
    // === PART ASSEMBLY ===

    // Load parts from library
    let engine_mainsail = parts.load("engine_mainsail")
    let tank_jumbo = parts.load("fuel_tank_jumbo_64")
    let capsule = parts.load("command_pod_mk1")
    let parachute = parts.load("parachute_mk16")

    // Assemble vessel (bottom to top)
    let vessel = assembly.create()
    vessel.attach(engine_mainsail, anchor="root")
    vessel.attach(tank_jumbo, engine_mainsail.anchor("top"))
    vessel.attach(capsule, tank_jumbo.anchor("top"))
    vessel.attach(parachute, capsule.anchor("top"))

    // Define staging
    vessel.add_stage(0, [parachute])              // Final stage: parachute
    vessel.add_stage(1, [capsule.rcs])            // RCS for deorbit
    vessel.add_stage(2, [engine_mainsail])        // Main engine

    // Compute vessel properties
    let mass_total = assembly.compute_mass(vessel)
    let com = assembly.compute_com(vessel)
    let moi = assembly.compute_moi(vessel)

    // === INITIAL CONDITIONS ===

    let body = celestial.kerbin()  // Launch from Kerbin
    let launch_site = body.surface_point(
        latitude = -0.09,   // KSC coordinates
        longitude = -74.56
    )

    let state = physics.initial_state(
        position = launch_site.position,
        velocity = launch_site.surface_velocity,  // Rotating with planet
        attitude = quaternion.from_euler(90deg, 0, 90deg),  // Vertical
        angular_velocity = Vec3(0, 0, 0)
    )

    // === FLIGHT PROGRAM ===

    // Gravity turn ascent profile
    fn pitch_program(t: Time, altitude: Length) -> Angle {
        if altitude < 1000m {
            return 90deg  // Vertical
        } else if altitude < 10000m {
            // Linear gravity turn
            return lerp(90deg, 45deg, (altitude - 1000m) / 9000m)
        } else if altitude < 45000m {
            return 45deg  // Continue at 45°
        } else {
            return 0deg  // Horizontal for orbital insertion
        }
    }

    // Control system
    fn control_law(state: VesselState, target_pitch: Angle) -> ControlInput {
        let current_pitch = state.attitude.pitch()
        let pitch_error = target_pitch - current_pitch

        // PID controller for pitch
        let torque_pitch = pid.compute(
            error = pitch_error,
            Kp = 0.5,
            Ki = 0.1,
            Kd = 0.2
        )

        // Throttle logic
        let throttle = if state.altitude < 70000m {
            // Full throttle until orbit
            1.0
        } else {
            // Throttle to circularize
            let target_speed = orbit.circular_velocity(state.altitude, body.μ)
            let speed_error = target_speed - state.velocity.magnitude()
            clamp(speed_error / 100, 0, 1)
        }

        return ControlInput(
            throttle = throttle,
            pitch = torque_pitch,
            yaw = 0,
            roll = 0
        )
    }

    // === SIMULATION LOOP ===

    let dt = 0.02s  // 50 Hz physics
    let sim = Simulation(initial_state = state, vessel = vessel)

    while sim.time < 600s {  // 10 minute flight
        // Current state
        let state = sim.state

        // Flight program
        let target_pitch = pitch_program(sim.time, state.altitude)
        let control = control_law(state, target_pitch)

        // === FORCE ACCUMULATION ===

        // 1. Gravity
        let F_gravity = physics.gravity_accel(state.position, [body]) * mass_total

        // 2. Thrust (if engine active)
        let F_thrust = if control.throttle > 0 {
            let altitude = state.altitude
            let Isp_eff = rocket.isp_altitude(
                Isp_vac = engine_mainsail.Isp_vac,
                Isp_sl = engine_mainsail.Isp_sl,
                P_amb = aero.atmosphere(altitude, body).pressure
            )

            let thrust_magnitude = control.throttle * engine_mainsail.thrust_max
            let thrust_vector = state.attitude.forward() * thrust_magnitude

            // Update fuel consumption
            let mass_flow = rocket.mass_flow(thrust_magnitude, Isp_eff, g0)
            vessel.consume_fuel(mass_flow * dt)

            thrust_vector
        } else {
            Vec3(0, 0, 0)
        }

        // 3. Aerodynamics (if in atmosphere)
        let F_aero = if state.altitude < 70000m {
            let (ρ, P, T) = aero.atmosphere(state.altitude, body)
            let v_rel = state.velocity  // Relative to atmosphere

            // Drag
            let F_drag = aero.drag_cube(
                parts = vessel.parts,
                velocity = v_rel,
                density = ρ
            )

            // Lift (if wings present)
            let F_lift = if vessel.has_wings() {
                aero.lift(vessel.wings, state.AoA, v_rel, ρ)
            } else {
                Vec3(0, 0, 0)
            }

            F_drag + F_lift
        } else {
            Vec3(0, 0, 0)
        }

        // Total force
        let F_total = F_gravity + F_thrust + F_aero

        // === TORQUE ACCUMULATION ===

        // Thrust vectoring (gimbal)
        let τ_thrust = if control.throttle > 0 {
            let gimbal_angle = control.pitch * engine_mainsail.gimbal_range
            let thrust_offset = engine_mainsail.position - com
            cross(thrust_offset, F_thrust.rotate(gimbal_angle))
        } else {
            Vec3(0, 0, 0)
        }

        // Aerodynamic torque
        let τ_aero = aero.compute_torque(vessel, state, com)

        // Reaction wheels
        let τ_reaction = vessel.reaction_wheels_torque(
            control.pitch,
            control.yaw,
            control.roll
        )

        // Total torque
        let τ_total = τ_thrust + τ_aero + τ_reaction

        // === PHYSICS INTEGRATION ===

        // Translational motion (RK4)
        let state_new = physics.rk4_step(
            state = state,
            force = F_total,
            mass = mass_total,
            dt = dt
        )

        // Rotational motion (quaternion integration)
        let attitude_new = physics.integrate_rotation(
            attitude = state.attitude,
            torque = τ_total,
            inertia = moi,
            dt = dt
        )

        sim.update(state_new, attitude_new)

        // === FAILURE CHECKS ===

        let g_force = F_total.magnitude() / mass_total / g0
        let q_pressure = 0.5 * ρ * state.velocity.magnitude()^2
        let temp_skin = aero.entry_heat(state.velocity, ρ) * dt  // Accumulated

        let failures = failure.check_all(
            vessel = vessel,
            g_force = g_force,
            q = q_pressure,
            temperature = temp_skin
        )

        if failures.any() {
            // Handle part failures (explosions!)
            for part in failures {
                let debris = failure.explode(part)
                vessel.remove_part(part)
                sim.add_debris(debris)
            }
        }

        // === STAGING ===

        // Auto-stage when fuel depleted
        if vessel.current_stage_fuel() < 0.01 * vessel.current_stage_capacity() {
            vessel.activate_next_stage()
        }

        // === ORBITAL STATE ===

        // Compute current orbit
        let orbit = orbit.state_to_elements(
            r = state.position,
            v = state.velocity,
            μ = body.μ
        )

        // Check for SOI transition
        if orbit.apoapsis > body.soi_radius {
            // Leaving Kerbin SOI, entering Sun SOI
            let parent = celestial.sun()
            state = orbit.soi_transition(state, body, parent)
        }

        // === TELEMETRY ===

        if sim.time % 1s == 0 {
            print(f"T+{sim.time}: Alt={state.altitude/1000:.1f}km, " +
                  f"Speed={state.velocity.magnitude():.0f}m/s, " +
                  f"Ap={orbit.apoapsis/1000:.0f}km, " +
                  f"Pe={orbit.periapsis/1000:.0f}km")
        }

        // Step simulation
        sim.step()
    }

    // === POST-FLIGHT ANALYSIS ===

    let final_orbit = orbit.state_to_elements(sim.state.position, sim.state.velocity, body.μ)

    export {
        apoapsis: final_orbit.apoapsis,
        periapsis: final_orbit.periapsis,
        inclination: final_orbit.inclination,
        total_delta_v: rocket.delta_v_stages(vessel.stages),
        max_g_force: sim.telemetry.max_g,
        max_q: sim.telemetry.max_dynamic_pressure,
        success: final_orbit.periapsis > 70000m  // Stable orbit achieved?
    }
}
```

---

## 4. Domain Specifications

### 4.1 OrbitalMechanicsDomain

**Purpose:** Keplerian orbits and patched conics

#### Core Types

```morphogen
type OrbitalElements {
    a: f32<m>,           // Semi-major axis
    e: f32,              // Eccentricity
    i: f32<rad>,         // Inclination
    Ω: f32<rad>,         // Longitude of ascending node
    ω: f32<rad>,         // Argument of periapsis
    ν: f32<rad>,         // True anomaly
    epoch: Time
}

type CelestialBody {
    name: String,
    μ: f32<m³/s²>,       // Gravitational parameter
    radius: f32<m>,
    atmosphere: Option<AtmosphereModel>,
    soi_radius: f32<m>,  // Sphere of influence
    rotation_period: Time
}

type Transfer {
    Δv1: Vec3<m/s>,
    Δv2: Vec3<m/s>,
    t_transfer: Time,
    total_Δv: f32<m/s>
}
```

#### Key Operators

**Kepler Solver:**
```json
{
  "name": "kepler_solve",
  "domain": "orbital_mechanics",
  "layer": 4,
  "inputs": [
    {"name": "M", "type": "f32<rad>", "description": "Mean anomaly"},
    {"name": "e", "type": "f32", "description": "Eccentricity"}
  ],
  "outputs": [
    {"name": "E", "type": "f32<rad>", "description": "Eccentric anomaly"}
  ],
  "determinism": "strict",
  "description": "Solve Kepler's equation M = E - e·sin(E) for E"
}
```

**State Vector to Orbital Elements:**
```json
{
  "name": "state_to_elements",
  "domain": "orbital_mechanics",
  "layer": 4,
  "inputs": [
    {"name": "r", "type": "Vec3<m>"},
    {"name": "v", "type": "Vec3<m/s>"},
    {"name": "μ", "type": "f32<m³/s²>"}
  ],
  "outputs": [
    {"name": "elements", "type": "OrbitalElements"}
  ],
  "determinism": "strict",
  "description": "Convert state vectors to Keplerian elements"
}
```

**Hohmann Transfer:**
```json
{
  "name": "hohmann_transfer",
  "domain": "orbital_mechanics",
  "layer": 4,
  "inputs": [
    {"name": "r1", "type": "f32<m>"},
    {"name": "r2", "type": "f32<m>"},
    {"name": "μ", "type": "f32<m³/s²>"}
  ],
  "outputs": [
    {"name": "transfer", "type": "Transfer"}
  ],
  "determinism": "strict",
  "description": "Compute Hohmann transfer delta-v and time"
}
```

**Lambert's Problem:**
```json
{
  "name": "lambert_solve",
  "domain": "orbital_mechanics",
  "layer": 4,
  "inputs": [
    {"name": "r1", "type": "Vec3<m>"},
    {"name": "r2", "type": "Vec3<m>"},
    {"name": "t_transfer", "type": "Time"},
    {"name": "μ", "type": "f32<m³/s²>"}
  ],
  "outputs": [
    {"name": "v1", "type": "Vec3<m/s>"},
    {"name": "v2", "type": "Vec3<m/s>"}
  ],
  "determinism": "repro",
  "description": "Solve Lambert's problem for rendezvous trajectory"
}
```

**Use Cases:**
- Orbital simulations (KSP, spaceflight games)
- Mission planning (NASA, SpaceX trajectory design)
- Education (orbital mechanics teaching tool)
- Satellite constellations (Starlink, etc.)

---

### 4.2 AerodynamicsDomain

**Purpose:** Atmospheric flight physics

#### Core Types

```morphogen
type AtmosphereModel {
    density_curve: Fn(altitude: f32<m>) -> f32<kg/m³>,
    pressure_curve: Fn(altitude: f32<m>) -> f32<Pa>,
    temperature_curve: Fn(altitude: f32<m>) -> f32<K>,
    scale_height: f32<m>
}

type DragCube {
    directions: [Vec3; 6],     // +X, -X, +Y, -Y, +Z, -Z
    coefficients: [f32; 6],    // Cd in each direction
    areas: [f32<m²>; 6],       // Projected area
    centers: [Vec3<m>; 6]      // Center of pressure
}

type AeroForces {
    drag: Vec3<N>,
    lift: Vec3<N>,
    torque: Vec3<N·m>,
    heating: f32<W>
}
```

#### Key Operators

**Atmosphere Model:**
```json
{
  "name": "atmosphere",
  "domain": "aerodynamics",
  "layer": 4,
  "inputs": [
    {"name": "altitude", "type": "f32<m>"},
    {"name": "body", "type": "Ref<CelestialBody>"}
  ],
  "outputs": [
    {"name": "density", "type": "f32<kg/m³>"},
    {"name": "pressure", "type": "f32<Pa>"},
    {"name": "temperature", "type": "f32<K>"}
  ],
  "determinism": "strict",
  "description": "Get atmospheric properties at altitude"
}
```

**Drag Cube:**
```json
{
  "name": "drag_cube",
  "domain": "aerodynamics",
  "layer": 4,
  "inputs": [
    {"name": "parts", "type": "List<PartRef>"},
    {"name": "velocity", "type": "Vec3<m/s>"},
    {"name": "density", "type": "f32<kg/m³>"}
  ],
  "outputs": [
    {"name": "drag_force", "type": "Vec3<N>"},
    {"name": "center_of_pressure", "type": "Vec3<m>"}
  ],
  "determinism": "strict",
  "description": "Compute drag using drag cube model with occlusion"
}
```

**Entry Heating:**
```json
{
  "name": "entry_heat",
  "domain": "aerodynamics",
  "layer": 4,
  "inputs": [
    {"name": "velocity", "type": "f32<m/s>"},
    {"name": "density", "type": "f32<kg/m³>"}
  ],
  "outputs": [
    {"name": "heat_flux", "type": "f32<W/m²>"}
  ],
  "determinism": "strict",
  "description": "Compute aerodynamic heating rate"
}
```

---

### 4.3 RocketEquationDomain

**Purpose:** Propulsion and delta-v calculations

#### Core Types

```morphogen
type Engine {
    thrust_vac: f32<N>,
    thrust_sl: f32<N>,
    Isp_vac: f32<s>,
    Isp_sl: f32<s>,
    mass: f32<kg>,
    gimbal_range: f32<deg>,
    throttle_min: f32,
    propellants: List<String>
}

type FuelTank {
    capacity: f32<kg>,
    fuel_type: String,
    dry_mass: f32<kg>,
    current_fuel: f32<kg>
}

type Stage {
    parts: List<PartRef>,
    m_wet: f32<kg>,
    m_dry: f32<kg>,
    Δv: f32<m/s>
}
```

#### Key Operators

**Delta-V:**
```json
{
  "name": "delta_v",
  "domain": "rocket_equation",
  "layer": 4,
  "inputs": [
    {"name": "m_wet", "type": "f32<kg>"},
    {"name": "m_dry", "type": "f32<kg>"},
    {"name": "Isp", "type": "f32<s>"},
    {"name": "g0", "type": "f32<m/s²>"}
  ],
  "outputs": [
    {"name": "Δv", "type": "f32<m/s>"}
  ],
  "determinism": "strict",
  "description": "Tsiolkovsky rocket equation: Δv = Isp·g0·ln(m_wet/m_dry)"
}
```

**Thrust-to-Weight Ratio:**
```json
{
  "name": "twr",
  "domain": "rocket_equation",
  "layer": 4,
  "inputs": [
    {"name": "thrust", "type": "f32<N>"},
    {"name": "mass", "type": "f32<kg>"},
    {"name": "g_local", "type": "f32<m/s²>"}
  ],
  "outputs": [
    {"name": "TWR", "type": "f32"}
  ],
  "determinism": "strict",
  "description": "Thrust-to-weight ratio"
}
```

**Staging Optimizer:**
```json
{
  "name": "staging_optimize",
  "domain": "rocket_equation",
  "layer": 7,
  "inputs": [
    {"name": "parts", "type": "List<Part>"},
    {"name": "target_Δv", "type": "f32<m/s>"}
  ],
  "outputs": [
    {"name": "optimal_staging", "type": "List<Stage>"},
    {"name": "total_Δv", "type": "f32<m/s>"}
  ],
  "determinism": "repro",
  "description": "Find optimal staging sequence for target delta-v"
}
```

---

### 4.4 PartsAssemblyDomain

**Purpose:** Modular vessel construction

#### Key Operators

**Create Vessel:**
```json
{
  "name": "create_vessel",
  "domain": "parts_assembly",
  "layer": 6,
  "inputs": [
    {"name": "parts", "type": "List<Part>"},
    {"name": "connections", "type": "List<Connection>"}
  ],
  "outputs": [
    {"name": "vessel", "type": "Vessel"}
  ],
  "determinism": "strict",
  "description": "Assemble parts into vessel"
}
```

**Compute Mass Properties:**
```json
{
  "name": "mass_properties",
  "domain": "parts_assembly",
  "layer": 6,
  "inputs": [
    {"name": "vessel", "type": "Vessel"}
  ],
  "outputs": [
    {"name": "mass_total", "type": "f32<kg>"},
    {"name": "center_of_mass", "type": "Vec3<m>"},
    {"name": "moment_of_inertia", "type": "Mat3<kg·m²>"}
  ],
  "determinism": "strict",
  "description": "Compute mass, COM, and MOI for vessel"
}
```

**Staging:**
```json
{
  "name": "activate_stage",
  "domain": "parts_assembly",
  "layer": 6,
  "inputs": [
    {"name": "vessel", "type": "Vessel"},
    {"name": "stage_num", "type": "i32"}
  ],
  "outputs": [
    {"name": "vessel_after", "type": "Vessel"},
    {"name": "debris", "type": "List<Part>"}
  ],
  "determinism": "strict",
  "description": "Activate stage, separate parts, return debris"
}
```

---

## 5. Integration with Existing Morphogen Domains

### 5.1 Geometry (TiaCAD)

**Use:** Part geometry and collision meshes

```morphogen
// Load part geometry
let engine = tiacad.load("parts/engine_vector.step")
let tank = tiacad.load("parts/tank_s3_14400.step")

// Attach with anchors
let vessel = assembly.create()
vessel.attach(engine, anchor="root")
vessel.attach(tank, engine.anchor("top_node"))

// Generate drag cube from geometry
let drag_cube = aero.drag_cube_from_geometry(engine.geometry)
```

---

### 5.2 Audio Domain

**Use:** Engine sounds, aerodynamic noise, explosions

```morphogen
// Engine sound synthesis
let engine_audio = audio.synthesize(
    type = "broadband_noise",
    frequency_range = [100 Hz, 2000 Hz],
    amplitude = engine.thrust / engine.thrust_max,
    modulation = "turbulence"
)

// Aerodynamic noise (proportional to velocity)
let aero_audio = audio.synthesize(
    type = "wind_noise",
    velocity = state.velocity.magnitude(),
    density = aero.atmosphere(state.altitude).density
)

// Explosion (when parts fail)
let explosion_audio = audio.explosion(
    intensity = part.mass * 10,  // Bigger parts = louder
    decay = 2s
)

// Mix all audio sources
let total_audio = audio.mix([engine_audio, aero_audio, explosion_audio])
```

---

### 5.3 Visualization Domain

**Use:** Render orbits, vessels, trajectories

```morphogen
// Render orbit prediction
let orbit_viz = visual.orbit_path(
    elements = current_orbit,
    body = kerbin,
    time_range = [0, orbit.period],
    color = "cyan"
)

// Render vessel
let vessel_viz = visual.render_3d(
    geometry = vessel.combined_geometry(),
    position = state.position,
    attitude = state.attitude,
    camera = camera.orbital_follow(distance=100m)
)

// Render velocity vector
let velocity_viz = visual.vector(
    origin = state.position,
    vector = state.velocity,
    scale = 10,
    color = "yellow"
)

// Maneuver node visualization
let maneuver_viz = visual.maneuver_node(
    position = orbit_position_at(t_maneuver),
    Δv = planned_burn.Δv,
    prograde = planned_burn.prograde,
    normal = planned_burn.normal,
    radial = planned_burn.radial
)
```

---

### 5.4 Combustion + Propulsion (J-Tube Integration!)

**Use:** Realistic engine modeling

```morphogen
// Model a liquid rocket engine with actual combustion
let engine = propulsion.liquid_engine(
    propellants = ["RP-1", "LOX"],
    mixture_ratio = 2.56,
    chamber_pressure = 10 MPa,
    nozzle_geometry = tiacad.load("nozzles/bell_nozzle_15.step"),
    combustion = combustion.adiabatic_flame(
        fuel = "RP-1",
        oxidizer = "LOX",
        phi = 1.0  // Stoichiometric
    )
)

// Compute thrust at current altitude
let thrust = propulsion.thrust(
    engine = engine,
    throttle = control.throttle,
    P_ambient = aero.atmosphere(state.altitude).pressure
)

// Pulse jet (J-tube style!)
let pulse_jet = propulsion.pulse_jet(
    tube = tiacad.load("engines/jtube_engine.step"),
    fuel_flow = 0.1 kg/s,
    ignition_freq = 50 Hz,
    combustion = combustion.periodic_flame()
)
```

---

## 6. Why Morphogen Excels at KSP-Style Simulations

### 6.1 Declarative Assembly

**KSP Problem:** Building rockets in code is tedious

**Morphogen Solution:** YAML-based part assembly

```yaml
# rocket.morphogen.yaml
vessel:
  name: "Kerbal X"
  parts:
    - id: engine1
      type: engine_mainsail
      position: [0, 0, 0]

    - id: tank1
      type: fuel_tank_jumbo
      attach_to: engine1.top_node

    - id: capsule
      type: command_pod_mk1
      attach_to: tank1.top_node

  staging:
    - stage: 0
      parts: [parachute]
    - stage: 1
      parts: [engine1]
```

---

### 6.2 GPU Acceleration

**KSP Problem:** Physics bottleneck with large vessels

**Morphogen Solution:** Automatic GPU offload

```morphogen
// Option 1: Explicit GPU backend
physics.integrate_gpu(
    states = vessel_states,
    forces_fn = compute_forces,
    dt = 0.02s,
    backend = "cuda"
)

// Option 2: Automatic backend selection
physics.integrate(
    state = state,
    forces_fn = compute_forces,
    dt = 0.02s,
    backend = "auto"  // Uses GPU if available
)
```

---

### 6.3 Level-of-Detail Physics

**KSP Problem:** N-body gravity too slow, patched conics inaccurate

**Morphogen Solution:** Switchable physics models

```morphogen
// Far from planets: cheap patched conics
let gravity = if distance_to_nearest_body > 100 * body.radius {
    orbit.gravity_patched_conics(state.position, bodies)
} else if distance_to_nearest_body > 10 * body.radius {
    // Medium distance: 2-body + perturbations
    orbit.gravity_perturbed(state.position, primary_body, other_bodies)
} else {
    // Close: full N-body
    orbit.gravity_nbody(state.position, bodies)
}
```

---

### 6.4 Cross-Domain Composability

**Example:** Audio + Physics + Visualization all integrated

```morphogen
scene LaunchWithFullExperience {
    // Physics
    let sim = kerbal_launch_simulation()

    // Audio (engine + aerodynamics)
    let audio = audio.mix([
        audio.engine_noise(sim.vessel.engines),
        audio.aero_noise(sim.state.velocity, sim.atmosphere),
        audio.background_ambience("launch_pad")
    ])

    // Visuals (3D render + HUD)
    let visual = visual.composite([
        visual.render_3d(sim.vessel, sim.state),
        visual.orbit_overlay(sim.orbit),
        visual.hud(sim.telemetry)
    ])

    // All synchronized
    export {
        simulation: sim,
        audio_stream: audio,
        video_stream: visual
    }
}
```

---

## 7. Educational Use Cases

### 7.1 Teaching Orbital Mechanics

**Scenario:** Students learn Hohmann transfers

```morphogen
lesson HohmannTransfer {
    // Setup: spacecraft in 200 km circular orbit
    let orbit_initial = orbit.circular(
        altitude = 200 km,
        body = earth
    )

    // Target: 400 km circular orbit
    let r1 = earth.radius + 200 km
    let r2 = earth.radius + 400 km

    // Compute transfer
    let transfer = orbit.hohmann_transfer(r1, r2, earth.μ)

    // Visualize
    visual.show([
        visual.orbit(orbit_initial, color="blue", label="Initial orbit"),
        visual.orbit_from_elements(transfer.elements, color="green", label="Transfer orbit"),
        visual.orbit(orbit_final, color="red", label="Target orbit"),
        visual.maneuver_node(transfer.burn1, label=f"Δv = {transfer.Δv1:.1f} m/s"),
        visual.maneuver_node(transfer.burn2, label=f"Δv = {transfer.Δv2:.1f} m/s")
    ])

    // Interactive: let students adjust target altitude
    let target_altitude = param(400 km, range=[200 km, 1000 km])
    // ... transfer recalculates automatically
}
```

---

### 7.2 Rocket Design Challenge

**Scenario:** Students design rocket to reach orbit with minimum fuel

```morphogen
challenge ReachOrbit {
    // Constraints
    let max_cost = 50000  // Budget limit
    let target_altitude = 80 km
    let target_periapsis = 75 km  // Stable orbit

    // Student designs vessel
    let vessel = assembly.create_from_parts(
        selected_parts = student_selection,
        budget = max_cost
    )

    // Simulate launch
    let result = simulate_launch(vessel)

    // Grade
    let score = if result.orbit.periapsis > target_periapsis {
        // Success! Score based on efficiency
        100 - (vessel.total_cost / max_cost) * 20  // Cheaper = better
    } else {
        // Failed to orbit
        (result.max_altitude / target_altitude) * 50  // Partial credit
    }

    export {
        success: result.orbit.periapsis > target_periapsis,
        score: score,
        delta_v_used: result.total_Δv,
        final_orbit: result.orbit
    }
}
```

---

## 8. Performance Considerations

### 8.1 Real-Time Requirements

**Target:** 50 Hz physics (0.02s timestep)

**Bottlenecks:**
- Force accumulation (many parts)
- Aerodynamics (drag cubes + occlusion)
- Integration (RK4 is 4x cost of Euler)

**Optimizations:**

```morphogen
// 1. Batch force computation
let forces = physics.batch_forces_gpu(
    vessels = [vessel1, vessel2, ...],
    states = [state1, state2, ...],
    backend = "cuda"
)

// 2. Simplified aero model when far from atmosphere
let aero_forces = if altitude > 100 km {
    Vec3(0, 0, 0)  // Skip aero entirely
} else if altitude > 70 km {
    aero.drag_simple(velocity, altitude)  // Simplified model
} else {
    aero.drag_cube_full(vessel.parts, velocity, atmosphere)  // Full model
}

// 3. Adaptive timestep
let dt = if in_atmosphere {
    0.02s  // Fine timestep for aero
} else {
    0.1s   // Coarse timestep in space
}
```

---

### 8.2 Large Vessel Handling

**Problem:** 1000-part space station = expensive

**Solution:** Hierarchical rigid bodies

```morphogen
// Treat docked vessels as single rigid body
let station = assembly.merge_rigid_bodies([
    vessel1,
    vessel2,
    vessel3
])

// Only compute inter-part forces when needed (docking, undocking)
```

---

### 8.3 Parallel Simulation

**Use Case:** Many vessels in physics range

```morphogen
// Simulate multiple vessels in parallel
let vessels = [vessel1, vessel2, ..., vessel_n]
let states = [state1, state2, ..., state_n]

let states_new = physics.simulate_parallel(
    vessels = vessels,
    states = states,
    dt = 0.02s,
    backend = "cuda",  // All on GPU
    threads = 16       // CPU fallback
)
```

---

## 9. Extensions & Future Work

### 9.1 Multiplayer / Distributed Simulation

**Idea:** Morphogen as server for multiplayer KSP-like game

```morphogen
server MultiplayerOrbit {
    // Each player's vessel
    let vessels = players.vessels()

    // Simulate all in parallel
    for vessel in vessels {
        let state_new = physics.step(vessel.state, vessel.vessel, dt)
        vessel.update(state_new)
    }

    // Broadcast state updates
    network.broadcast(vessels.states())
}
```

---

### 9.2 Mission Planning Tools

**Idea:** Morphogen as mission design software (like STK, GMAT)

```morphogen
mission MarsTransfer {
    // Earth departure
    let earth_orbit = orbit.circular(300 km, earth)

    // Mars arrival
    let mars_orbit = orbit.circular(500 km, mars)

    // Find launch window
    let window = orbit.porkchop_plot(
        body_from = earth,
        body_to = mars,
        departure_range = [2026-01-01, 2026-12-31],
        arrival_range = [2026-06-01, 2027-06-01],
        objective = "minimize_Δv"
    )

    // Plot results
    visual.porkchop(window)

    export {
        launch_date: window.optimal.departure,
        arrival_date: window.optimal.arrival,
        total_Δv: window.optimal.Δv
    }
}
```

---

### 9.3 Procedural Planet Generation

**Idea:** Generate planets with realistic properties

```morphogen
planet ProceduralEarthLike {
    // Physical parameters
    let radius = param(6371 km, range=[1000 km, 20000 km])
    let mass = param(5.972e24 kg, range=[1e23 kg, 1e26 kg])
    let rotation_period = param(24 hours, range=[1 hour, 100 hours])

    // Derived
    let μ = G * mass
    let surface_gravity = μ / radius^2

    // Atmosphere
    let atmosphere = atmosphere.from_template(
        type = "earth_like",
        scale_height = 8500 m,
        surface_pressure = 101325 Pa
    )

    // Terrain (using noise domain!)
    let terrain = noise.fractal_terrain(
        resolution = 1000 m,
        octaves = 8,
        persistence = 0.5,
        lacunarity = 2.0,
        seed = random()
    )

    export CelestialBody {
        name: "Procedural Earth-like",
        μ: μ,
        radius: radius,
        atmosphere: atmosphere,
        terrain: terrain
    }
}
```

---

## 10. Comparison: Morphogen vs. Unity Physics (KSP's Engine)

| Feature | Unity Physics (KSP) | Morphogen |
|---------|---------------------|-------|
| **Physics Backend** | PhysX (closed source) | Open, modular operators |
| **Part Assembly** | GameObject hierarchy | YAML + operator graph |
| **Orbital Mechanics** | Custom C# scripts | Native OrbitalMechanicsDomain |
| **Aerodynamics** | Drag cubes (custom) | AerodynamicsDomain + optional CFD |
| **GPU Acceleration** | Limited (mainly rendering) | Full physics on GPU |
| **Determinism** | Non-deterministic (PhysX) | Deterministic (repro/strict) |
| **Extensibility** | Unity plugins (C#) | Operator registry (any backend) |
| **Audio** | Unity Audio (sample-based) | AudioDomain (synthesis + samples) |
| **Visuals** | Unity renderer | VisualizationDomain (flexible) |
| **Multi-domain** | Hard (separate systems) | Native (operator graph) |
| **Educational Use** | Mod-based (limited) | First-class (declarative) |

**Conclusion:** Morphogen offers more flexibility, better multi-domain integration, GPU acceleration, and determinism — perfect for serious simulation and education.

---

## 11. Summary

### What We've Demonstrated

1. **Complete Physics Stack** — Orbital mechanics, aerodynamics, rocket equation, assembly, integration, failures
2. **Domain Specifications** — Detailed operator definitions for each physics domain
3. **Full Pipeline Example** — End-to-end launch simulation with staging, control, and telemetry
4. **Cross-Domain Integration** — Geometry, audio, visuals, combustion all work together
5. **Educational Applications** — Teaching orbital mechanics and rocket design
6. **Performance Strategy** — GPU acceleration, LOD physics, adaptive timesteps
7. **Future Extensions** — Multiplayer, mission planning, procedural generation

### Why This Validates Morphogen

- **KSP is the perfect benchmark** — Complex, multi-domain, real-time physics
- **Morphogen handles it naturally** — Operator graphs map directly to game systems
- **Beyond games** — Education, aerospace research, mission planning
- **Reusable domains** — OrbitalMechanics, Aerodynamics, RocketEquation are broadly useful
- **Integration wins** — Geometry (TiaCAD) + Audio + Combustion (J-tube!) all compose

### Next Steps

1. **Prototype OrbitalMechanics domain** — Start with Kepler solver, state conversions
2. **Add Aerodynamics operators** — Drag cube, atmosphere model
3. **Build example mission** — Simple orbital insertion
4. **Integration tests** — Cross-domain flows (geometry → aero, physics → visuals)
5. **Educational tools** — Interactive lessons on orbital mechanics
6. **Performance benchmarks** — Compare Morphogen vs. Unity/Unreal for physics

---

## 12. Related Documentation

### Morphogen Documentation

- **[architecture/domain-architecture.md](../architecture/domain-architecture.md)** — Complete domain vision
- **[ADR-002: Cross-Domain Architectural Patterns](../adr/002-cross-domain-architectural-patterns.md)** — Reference systems and operator composition
- **[specifications/physics-domains.md](../specifications/physics-domains.md)** — Physics operator specifications
- **[specifications/geometry.md](../specifications/geometry.md)** — TiaCAD geometry integration
- **[examples/j-tube-firepit-multiphysics.md](./j-tube-firepit-multiphysics.md)** — Similar multi-physics example

### External References

- **Kerbal Space Program** — https://www.kerbalspaceprogram.com/
- **Orbital Mechanics** — Curtis, "Orbital Mechanics for Engineering Students"
- **Rocket Propulsion** — Sutton & Biblarz, "Rocket Propulsion Elements"
- **Patched Conics** — https://en.wikipedia.org/wiki/Patched_conic_approximation
- **Lambert's Problem** — Izzo, "Revisiting Lambert's Problem" (2015)

---

## Conclusion

**Kerbal Space Program proves that Morphogen's operator graph paradigm extends to real-time game physics.**

By implementing domains like OrbitalMechanics, Aerodynamics, RocketEquation, and PartsAssembly, Morphogen becomes a **powerful platform for aerospace simulation** — useful for:

- **Games** (KSP-like spaceflight sims)
- **Education** (teaching orbital mechanics)
- **Research** (mission planning, trajectory optimization)
- **Industry** (satellite constellation design, launch analysis)

**And here's the kicker:** With Morphogen's cross-domain integration, you can add:
- **Realistic combustion** (from J-tube domain!)
- **Engine acoustics** (from AudioDomain)
- **Procedural planets** (from NoiseDomain)
- **Part CAD** (from TiaCAD/GeometryDomain)

**Morphogen isn't just a simulation framework — it's a multi-physics platform that can build KSP... and so much more.**

---

**End of Document**
