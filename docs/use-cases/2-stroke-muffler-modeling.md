# 2-Stroke Engine & Muffler Acoustic Modeling in Morphogen

**Date:** 2025-11-15
**Status:** Vision / Use Case Specification
**Complexity:** Advanced Multi-Domain
**Domains:** FluidDynamics, Acoustics, Geometry, Audio, Optimization, Thermal

---

## Executive Summary

**2-stroke exhaust system modeling** is a **perfect showcase** for Morphogen's multi-domain, operator-driven, geometry-aware simulation architecture.

A 2-stroke exhaust system is not just a pipe — it is:
- An **acoustic device** (resonance, filtering, sound shaping)
- An **engine tuning device** (backpressure timing, scavenging)
- A **coupled nonlinear fluid system** (pressure waves, thermal effects)
- A **geometric structure** (exact pipe shapes, expansion chambers, cones)

This problem **uniquely demonstrates Morphogen's core strengths**:
- ✅ **Multi-domain coupling** — fluid → acoustic → thermal → audio
- ✅ **Geometry-aware physics** — shape directly influences simulation
- ✅ **Time & frequency domain** — pressure waves + spectral filtering
- ✅ **Operator composition** — complex physics from simple building blocks
- ✅ **Real-world outputs** — WAV files, torque curves, visualizations

**No mainstream system unifies geometry + acoustics + fluid dynamics + signal processing in one composable operator graph.**

Morphogen does.

---

## Why This Problem Is Perfect for Morphogen

### Multi-Domain Nature

A 2-stroke muffler simulation spans **at least 6 domains**:

1. **FluidDynamicsDomain** — Compressible gas flow, pressure pulses, backpressure
2. **AcousticsDomain** — Waveguide propagation, resonance, impedance matching
3. **GeometryDomain** — Exact pipe shapes (cones, chambers, bends)
4. **ThermalDomain** — Temperature gradients, viscosity changes
5. **AudioDomain** — Sound synthesis, microphone modeling, WAV export
6. **OptimizationDomain** — Inverse design, parameter tuning

These are **not separate tools** — they are **unified operators** in Morphogen's graph IR.

### Operator Graph Architecture

Every component becomes an operator:

```
EnginePressurePulse
        ↓
PulseGeneratorOperator (combustion cycle)
        ↓
WaveguideNetwork (geometry → discretized pipes)
        ↓
ReflectionOperators (expansion chambers, cones)
        ↓
AbsorptionOperators (fiberglass muffler packing)
        ↓
RadiationOperator (exhaust → open air)
        ↓
AudioDomain (microphone model, stereo, WAV)
        ↓
Output: .wav file, torque curve, pressure visualization
```

This is **identical** to:
- Circuit simulation (SPICE)
- Audio signal chains (RiffStack)
- 3D spatial resolvers (TiaCAD)
- Physics solvers (field operators)

Just applied to **gas + acoustics**.

---

## Physical Components & Morphogen Operators

### 1. Pressure Wave Propagation

**Physics:**
- Combustion creates pressure pulses
- Pulses travel down exhaust pipe
- Wave speed depends on temperature, gas composition

**Morphogen Operators:**
```morphogen
# FluidDynamicsDomain
pulse_generator(rpm, cylinder_volume, compression_ratio) -> PressurePulse
pressure_wave_propagate(pulse, pipe_geometry, dt) -> Field1D<Pa>
```

**Mathematical Model:**
- 1D compressible Euler equations
- Non-linear wave equation with viscosity
- CFL-stable time integration

---

### 2. Expansion Chamber (The "Bulge")

**Physics:**
- The iconic 2-stroke expansion chamber is a **Helmholtz-like resonator**
- Diverging cone creates negative reflection wave
- Converging cone sends positive reflection wave back to cylinder
- Timing is critical: wave must arrive during scavenging phase

**Morphogen Operators:**
```morphogen
# AcousticsDomain + GeometryDomain
expansion_chamber = geom.expansion_chamber(
    inlet_diameter = 40mm,
    diverge_angle = 12deg,
    max_diameter = 120mm,
    converge_angle = 8deg,
    outlet_diameter = 50mm
)

# Discretize geometry into waveguide segments
waveguide = acoustic.waveguide_from_geometry(
    geometry = expansion_chamber,
    segments = 100,
    sample_rate = 44100Hz  # or physics dt
)

# Compute reflection coefficients at each segment boundary
reflections = acoustic.reflection_coefficients(waveguide)
```

**Why This Needs Geometry:**
- Wave behavior **depends on exact cone angles**
- Morphogen's `GeometryDomain` provides precise shape definitions
- Operators sample geometry to build acoustic network

---

### 3. Resonance & Tuning

**Physics:**
- Expansion chamber acts as **bandpass filter**
- Resonant frequency depends on chamber volume and neck dimensions
- Tuned to specific RPM for peak power

**Morphogen Operators:**
```morphogen
# AcousticsDomain
helmholtz_resonator(
    volume = chamber_volume,
    neck_length = stinger_length,
    neck_area = stinger_area
) -> ResonantFrequency

# Find peak efficiency RPM
resonant_rpm = acoustic.resonance_to_rpm(
    resonator,
    engine_config
)
```

**Output:**
- Transfer function (frequency domain)
- Peak power RPM prediction
- Spectral response curve

---

### 4. Backpressure Timing

**Physics:**
- Reflected wave arrives at cylinder exactly during scavenging
- Pushes fresh fuel-air mixture back into cylinder
- Prevents mixture from escaping out exhaust port

**Morphogen Operators:**
```morphogen
# FluidDynamicsDomain + AcousticsDomain coupling
backpressure_wave = acoustic.reflected_wave(
    incident_pulse,
    reflection_coefficients,
    pipe_lengths
)

# Check timing alignment with engine cycle
scavenge_timing = engine.scavenge_window(rpm, port_timing)
efficiency = timing.alignment(backpressure_wave, scavenge_timing)
```

**Multi-Domain Coupling:**
- Fluid dynamics generates initial pulse
- Acoustics propagates and reflects wave
- Engine timing determines effectiveness

---

### 5. Thermal Effects

**Physics:**
- Hot exhaust gases (500-800°C)
- Temperature affects wave speed: \( c = \sqrt{\gamma R T} \)
- Viscosity changes with temperature

**Morphogen Operators:**
```morphogen
# ThermalDomain
gas_properties = thermal.gas_properties(
    temperature_field,
    composition = "exhaust"  # CO2, H2O, unburned HC
)

wave_speed = fluid.wave_speed(gas_properties)
viscosity = thermal.viscosity(temperature_field)

# Update waveguide with temperature-dependent properties
waveguide_thermal = acoustic.update_properties(
    waveguide,
    wave_speed,
    viscosity
)
```

---

### 6. Muffler (Sound Absorption)

**Physics:**
- Perforated pipes
- Fiberglass packing absorbs high frequencies
- Quarter-wave tubes (resonant absorbers)

**Morphogen Operators:**
```morphogen
# AcousticsDomain
perforated_pipe(
    hole_diameter = 5mm,
    hole_spacing = 20mm,
    open_area_ratio = 0.3
) -> AcousticImpedance

absorption_material(
    type = "fiberglass",
    density = 50 kg/m³,
    thickness = 25mm
) -> AbsorptionCoefficient

# Apply to waveguide
muffler_segment = acoustic.absorptive_segment(
    waveguide,
    perforated_pipe,
    absorption_material
)
```

**Transfer Function:**
- Frequency-dependent attenuation
- Phase response
- Backpressure contribution

---

### 7. Radiation to Open Air

**Physics:**
- Exhaust exits pipe into infinite half-space
- Radiation impedance (frequency-dependent)
- Directivity pattern

**Morphogen Operators:**
```morphogen
# AcousticsDomain
radiation = acoustic.radiation_impedance(
    pipe_diameter = 50mm,
    type = "unflanged"  # or "flanged"
)

sound_pressure_level = acoustic.radiate_to_air(
    exhaust_pressure,
    radiation,
    distance = 1.0m,
    angle = 90deg  # perpendicular to pipe axis
)
```

---

### 8. Audio Output

**Final Stage:**
- Convert acoustic pressure to audio signal
- Apply microphone response
- Stereo positioning
- Export WAV

**Morphogen Operators:**
```morphogen
# AudioDomain
microphone = audio.microphone_model(
    type = "measurement",  # flat response
    position = Vec3(1.0m, 0.5m, 0.0m),
    orientation = facing_exhaust
)

audio_signal = audio.pressure_to_signal(
    sound_pressure_level,
    microphone,
    sample_rate = 48000Hz
)

# Stereo mix
stereo = audio.stereo_mix(
    left = audio_signal * 0.7,
    right = audio_signal * 0.3
)

# Export
audio.export_wav("2stroke_exhaust.wav", stereo)
```

**Result:**
- Realistic engine sound at different RPMs
- Different microphone positions
- Effect of muffler modifications on sound

---

## Complete Morphogen Operator Graph

Here's how the full simulation composes:

```morphogen
scene TwoStrokeExhaust {
    // ==================
    // 1. GEOMETRY
    // ==================

    # Define exhaust geometry
    let header_pipe = geom.cylinder(
        diameter = 40mm,
        length = 200mm
    )

    let expansion_chamber = geom.expansion_chamber(
        inlet_diameter = 40mm,
        diverge_length = 150mm,
        diverge_angle = 12deg,
        belly_diameter = 120mm,
        belly_length = 100mm,
        converge_length = 180mm,
        converge_angle = 8deg,
        outlet_diameter = 50mm
    )

    let stinger = geom.cylinder(
        diameter = 50mm,
        length = 100mm
    )

    let muffler = geom.muffler(
        inlet_diameter = 50mm,
        body_diameter = 100mm,
        length = 300mm,
        packing = "fiberglass"
    )

    let tailpipe = geom.cylinder(
        diameter = 45mm,
        length = 150mm
    )

    # Compose exhaust system
    let exhaust_system = geom.pipe_chain([
        header_pipe,
        expansion_chamber,
        stinger,
        muffler,
        tailpipe
    ])

    // ==================
    // 2. FLUID DYNAMICS
    // ==================

    # Engine parameters
    const RPM: f32 = 8000.0
    const DISPLACEMENT: f32 [cm³] = 125.0
    const COMPRESSION_RATIO: f32 = 12.0

    # Generate combustion pressure pulse
    @state pulse: PressurePulse = engine.combustion_pulse(
        rpm = RPM,
        displacement = DISPLACEMENT,
        compression_ratio = COMPRESSION_RATIO,
        fuel = "gasoline_2stroke"
    )

    # Temperature field (hot exhaust gases)
    @state temperature: Field1D<K> = field.initialize(
        exhaust_system.length_discretization(dx = 1mm),
        value = 600K  # typical 2-stroke exhaust temp
    )

    # Gas properties
    let gas_props = thermal.exhaust_gas_properties(temperature)
    let wave_speed = fluid.sound_speed(gas_props)

    // ==================
    // 3. ACOUSTICS
    // ==================

    # Convert geometry to acoustic waveguide network
    let waveguide = acoustic.waveguide_from_geometry(
        geometry = exhaust_system,
        discretization = 1mm,
        properties = gas_props
    )

    # Compute reflection coefficients at discontinuities
    let reflections = acoustic.compute_reflections(waveguide)

    # Absorption in muffler
    let absorption = acoustic.fiberglass_absorption(
        density = 50 kg/m³,
        thickness = 25mm,
        frequency_range = (100Hz, 10000Hz)
    )

    # Radiation impedance at tailpipe exit
    let radiation = acoustic.radiation_impedance(
        diameter = tailpipe.diameter,
        type = "unflanged"
    )

    // ==================
    // 4. TIME-STEPPING
    // ==================

    @state pressure_field: Field1D<Pa> = zeros(waveguide.length)
    @state velocity_field: Field1D<m/s> = zeros(waveguide.length)

    const DT: Time = 1.0 / 44100.0 s  # audio sample rate
    const CYCLE_TIME: Time = 60.0 / RPM s

    @state time: Time = 0.0s
    @state cycle_count: i32 = 0

    # Audio buffer for recording
    @state audio_buffer: Vec<f32> = []

    flow(dt = DT, steps = 44100 * 3) {  # 3 seconds of audio

        # Inject combustion pulse at start of each cycle
        if time % CYCLE_TIME < DT {
            pressure_field[0] = pulse.peak_pressure
            cycle_count = cycle_count + 1
        }

        # Propagate pressure wave using 1D waveguide
        (pressure_field, velocity_field) = acoustic.waveguide_step(
            pressure_field,
            velocity_field,
            waveguide,
            reflections,
            absorption,
            radiation,
            dt
        )

        # Thermal coupling (optional, for high fidelity)
        # temperature = thermal.convection_step(temperature, velocity_field, dt)
        # waveguide = acoustic.update_properties(waveguide, thermal.gas_properties(temperature))

        # Sample pressure at microphone location (1m from tailpipe)
        let mic_pressure = acoustic.radiate_to_point(
            pressure_field[-1],  # tailpipe exit
            distance = 1.0m,
            angle = 90deg,
            radiation
        )

        # Convert to audio sample
        let audio_sample = audio.pressure_to_sample(
            mic_pressure,
            reference_pressure = 20e-6 Pa  # 0 dB SPL
        )

        audio_buffer.push(audio_sample)

        time = time + dt
    }

    // ==================
    // 5. OUTPUTS
    // ==================

    # Export audio
    audio.export_wav(
        "2stroke_8000rpm.wav",
        audio_buffer,
        sample_rate = 44100Hz,
        bit_depth = 16
    )

    # Compute transfer function (frequency response)
    let transfer_function = transform.fft(audio_buffer)
    visual.plot_spectrum(
        transfer_function,
        title = "Exhaust Sound Spectrum",
        output = "spectrum.png"
    )

    # Compute backpressure timing analysis
    let backpressure_timing = engine.analyze_backpressure_timing(
        pressure_field,
        cycle_time = CYCLE_TIME,
        scavenge_window = engine.scavenge_timing(RPM)
    )

    print("Backpressure efficiency: {backpressure_timing.efficiency * 100}%")
    print("Resonant RPM: {backpressure_timing.resonant_rpm}")
    print("Peak power RPM: {backpressure_timing.peak_power_rpm}")
}
```

---

## What Morphogen Can Produce

### Outputs

1. **Audio:**
   - Realistic engine sound at any RPM
   - Different microphone positions (near/far, on-axis/off-axis)
   - Effect of pipe modifications on sound
   - Full-throttle vs idle audio renders

2. **Acoustic Analysis:**
   - Transfer function (frequency response)
   - Resonant frequencies
   - Sound pressure level at various distances
   - Directivity patterns

3. **Performance Metrics:**
   - Backpressure timing efficiency
   - Optimal RPM for peak power
   - Effect of geometry changes on torque curve
   - Scavenging effectiveness

4. **Visualizations:**
   - Pressure wave animations inside pipe
   - Temperature distribution
   - Resonance mode shapes
   - Spectrum waterfall (RPM sweep)

5. **Design Optimization:**
   - Inverse design: "find chamber dimensions for peak power at 9000 RPM"
   - Multi-objective: "maximize power, minimize noise"
   - Parameter sensitivity analysis

---

## Technical Approaches (Multiple Fidelity Levels)

Morphogen's domain system supports **multiple methods** depending on fidelity needs:

### Method 1: 1D Waveguide Simulation (Fast, Accurate)

**Approach:**
- Discretize pipe into segments
- Compute reflection/transmission coefficients
- Digital waveguide propagation (sample-by-sample)

**Pros:**
- Very fast (real-time capable)
- Accurate for 1D wave propagation
- Used in commercial exhaust tuning software
- Used in Yamaha VL synthesizers

**Cons:**
- 1D only (no radial modes)
- Assumes planar wave propagation

**Morphogen Implementation:**
```morphogen
# AcousticsDomain
waveguide_network_1d(geometry, sample_rate)
```

---

### Method 2: Lumped Acoustic Networks (Very Fast)

**Approach:**
- Model as **acoustic circuit**:
  - Chambers = capacitors (compliance)
  - Pipes = inductors (inertance)
  - Absorption = resistors
  - Helmholtz resonators = LC circuits

**Pros:**
- Extremely fast (analytical)
- Perfect for quick simulation
- Directly analogous to SPICE circuits

**Cons:**
- Less accurate for complex geometries
- Lumped approximation breaks down at high frequencies

**Morphogen Implementation:**
```morphogen
# AcousticsDomain + CircuitDomain
acoustic_network = circuit.from_acoustic_geometry(exhaust_system)
response = circuit.simulate(acoustic_network, pulse)
```

This is **exactly like SPICE** but for acoustics.

---

### Method 3: FDTD Acoustics (High Fidelity)

**Approach:**
- Finite Difference Time Domain
- Solve 3D wave equation on grid
- Captures complex 3D effects

**Pros:**
- High fidelity
- Captures radial modes, turbulence
- Full 3D geometry

**Cons:**
- Slower (but GPU-accelerated)
- More complex boundary conditions

**Morphogen Implementation:**
```morphogen
# AcousticsDomain + FieldDomain
acoustic_field_3d = field.acoustic_fdtd(
    geometry,
    source = pulse,
    resolution = 1mm,
    dt = CFL_stable_timestep
)
```

---

### Method 4: CFD Coupling (Optional, Very High Fidelity)

**Approach:**
- Full Navier-Stokes simulation
- Compressible flow with chemistry
- Turbulence modeling

**Pros:**
- Maximum fidelity
- Captures all nonlinear effects

**Cons:**
- Very slow (hours to days)
- Overkill for most use cases

**Morphogen Implementation:**
```morphogen
# FluidDynamicsDomain (advanced)
cfd_solver = fluid.navier_stokes_compressible(
    geometry,
    turbulence_model = "LES",
    chemistry = "2stroke_combustion"
)
```

Only needed for **racing-level** optimization.

---

## Required Morphogen Domains & Operators

### FluidDynamicsDomain (NEW)

**Purpose:** Compressible gas flow, pressure waves, thermodynamics

**Operators:**
```morphogen
# Pulse generation
engine.combustion_pulse(rpm, displacement, compression_ratio) -> PressurePulse

# 1D compressible flow
fluid.euler_1d(pressure, velocity, density, dt) -> (pressure', velocity', density')

# Gas properties
fluid.sound_speed(gas_properties) -> m/s
fluid.gas_properties(temperature, composition) -> GasProperties

# Backpressure analysis
engine.backpressure_timing(pressure_field, engine_timing) -> Efficiency
```

**Dependencies:**
- IntegratorsDomain (time-stepping)
- ThermalDomain (temperature effects)

---

### AcousticsDomain (NEW)

**Purpose:** Wave propagation, resonance, impedance, sound radiation

**Operators:**
```morphogen
# Waveguide construction
acoustic.waveguide_from_geometry(geometry, discretization) -> WaveguideNetwork
acoustic.reflection_coefficients(waveguide) -> Vec<f32>

# Propagation
acoustic.waveguide_step(pressure, velocity, waveguide, dt) -> (pressure', velocity')
acoustic.fdtd_step(acoustic_field, dt) -> acoustic_field'

# Components
acoustic.helmholtz_resonator(volume, neck_length, neck_area) -> Resonator
acoustic.perforated_pipe(hole_diameter, spacing, open_ratio) -> AcousticImpedance
acoustic.absorption_material(type, density, thickness) -> AbsorptionCoefficient

# Radiation
acoustic.radiation_impedance(diameter, type) -> Impedance
acoustic.radiate_to_point(source_pressure, distance, angle, impedance) -> Pa

# Analysis
acoustic.transfer_function(waveguide, freq_range) -> FrequencyResponse
acoustic.resonant_frequencies(waveguide) -> Vec<Hz>
```

**Dependencies:**
- GeometryDomain (pipe shapes)
- FieldDomain (3D acoustics via FDTD)
- TransformDomain (FFT for frequency analysis)

---

### GeometryDomain (Additions)

**New Primitives:**
```morphogen
# Exhaust-specific shapes
geom.expansion_chamber(inlet, diverge_angle, belly, converge_angle, outlet)
geom.pipe_chain(segments) -> CompositeGeometry
geom.perforated_pipe(diameter, length, hole_pattern)
geom.muffler(inlet, body, packing_material)

# Meshing for acoustics
geom.axis_symmetric_mesh(geometry, radial_resolution, axial_resolution)
```

---

### AudioDomain (Additions)

**New Operators:**
```morphogen
# Acoustic → Audio conversion
audio.pressure_to_sample(pressure_pa, reference_pressure) -> f32
audio.microphone_model(type, position, orientation) -> Microphone

# Spatial audio
audio.stereo_from_position(source_position, listener_position) -> (left, right)
audio.room_response(audio_signal, room_geometry) -> audio_signal'

# Export
audio.export_wav(filename, samples, sample_rate, bit_depth)
```

---

### OptimizationDomain (Future)

**Use Cases:**
```morphogen
# Find optimal expansion chamber dimensions
optimizer.minimize(
    objective = |params| -1.0 * engine_power(params),  # maximize power
    constraints = [
        noise_level(params) < 95 dB,
        backpressure(params) < 0.5 bar
    ],
    params = {
        diverge_angle: (8deg, 15deg),
        belly_diameter: (100mm, 150mm),
        converge_angle: (5deg, 12deg)
    },
    method = "nelder_mead"
)

# Multi-objective
optimizer.pareto_front(
    objectives = [maximize(power), minimize(noise)],
    params = chamber_dimensions
)
```

---

## Why This Showcases Morphogen Better Than Almost Anything

### 1. Truly Multi-Domain

Most examples use **one or two domains**.
This uses **six domains** in deep coupling:
- Geometry → Acoustics (shape determines wave behavior)
- Acoustics → Audio (pressure → sound)
- FluidDynamics → Acoustics (pulse generation)
- Thermal → FluidDynamics (temperature affects waves)
- Optimization → Everything (inverse design)

### 2. Real-World, Non-Trivial

This is not a toy problem.
**Professional exhaust designers** use expensive specialized software for this.
Morphogen unifies what currently requires multiple tools:
- CAD (SolidWorks, OnShape)
- CFD (ANSYS Fluent)
- Acoustics (Ricardo WAVE, GT-Power)
- Audio (custom tools)

### 3. Geometry-Physics Coupling

The **exact pipe shape** directly determines **acoustic behavior**.
This is the TiaCAD lesson applied to physics:
- Geometry is not just visualization
- Geometry **drives** simulation
- Operators sample geometry for discretization

### 4. Time-Domain + Frequency-Domain

Morphogen naturally handles both:
- **Time-domain**: Pressure wave propagation (integration)
- **Frequency-domain**: Transfer function analysis (FFT)

Switching between them is **seamless**.

### 5. Novel

**No system does this.**
- CAD tools don't do acoustics
- CFD tools don't do audio synthesis
- Audio tools don't do fluid dynamics
- No one unifies all of them

Morphogen's **operator graph architecture** makes this natural.

---

## Implementation Phases

### Phase 1: Foundation (v0.9)
- ✅ GeometryDomain with pipe primitives
- ✅ Basic 1D FluidDynamics operators
- ✅ 1D waveguide AcousticsDomain
- ✅ Audio export to WAV

**Milestone:** Simple pipe resonance simulation

---

### Phase 2: Expansion Chamber (v1.0)
- ✅ Reflection coefficient calculation
- ✅ Expansion chamber geometry primitives
- ✅ Backpressure timing analysis
- ✅ Transfer function computation

**Milestone:** Full 2-stroke expansion chamber tuning

---

### Phase 3: Muffler & Absorption (v1.1)
- ✅ Perforated pipe impedance
- ✅ Fiberglass absorption models
- ✅ Multi-segment waveguide networks
- ✅ Radiation impedance

**Milestone:** Complete exhaust system with muffler

---

### Phase 4: Advanced (v1.2+)
- ✅ 3D FDTD acoustics
- ✅ CFD coupling (optional)
- ✅ Thermal effects
- ✅ Optimization loops

**Milestone:** Inverse design of optimal exhaust

---

## Example Use Cases

### 1. Exhaust Sound Design
**Task:** "Make it sound like a vintage Yamaha RD350"
**Approach:**
- Record target audio
- Run inverse optimization to match spectrum
- Output chamber dimensions

### 2. Performance Tuning
**Task:** "Maximize power at 9000 RPM"
**Approach:**
- Sweep expansion chamber parameters
- Compute backpressure timing efficiency
- Find optimal geometry

### 3. Noise Reduction
**Task:** "Reduce noise below 90 dB without losing power"
**Approach:**
- Add muffler components
- Optimize packing density and chamber volume
- Validate against constraints

### 4. Educational Tool
**Task:** "Teach students how 2-stroke tuning works"
**Approach:**
- Interactive parameter sweeps
- Visualize pressure waves in real-time
- Show effect of geometry changes

---

## Comparison to Existing Tools

| Tool | Domain | Morphogen Advantage |
|------|--------|-----------------|
| **Ricardo WAVE** | 1D gas dynamics, acoustics | ❌ Proprietary, expensive<br>❌ No geometry integration<br>❌ No audio synthesis<br>✅ Morphogen: Open, unified, audio output |
| **GT-Power** | Engine simulation | ❌ Black-box solver<br>❌ No geometry modeling<br>✅ Morphogen: Transparent operators, geometry-aware |
| **ANSYS Fluent** | 3D CFD | ❌ Overkill for 1D waves<br>❌ No acoustic analysis<br>✅ Morphogen: Right fidelity for the problem |
| **COMSOL** | Multi-physics | ❌ General-purpose (not optimized)<br>❌ No audio output<br>✅ Morphogen: Domain-specific operators |
| **Pure Data / Max/MSP** | Audio synthesis | ❌ No physics<br>❌ No geometry<br>✅ Morphogen: Physics-driven audio |

**Morphogen is the only system that unifies all of these.**

---

## Documentation & Examples

When implemented, this should include:

### Documentation
- `docs/domains/FLUID_DYNAMICS.md` — FluidDynamics domain spec
- `docs/domains/ACOUSTICS.md` — Acoustics domain spec
- `docs/examples/2stroke-expansion-chamber.md` — Walkthrough

### Examples
- `examples/acoustics/01_pipe_resonance.morph` — Simple resonance
- `examples/acoustics/02_helmholtz_resonator.morph` — Resonator tuning
- `examples/acoustics/03_expansion_chamber.morph` — 2-stroke chamber
- `examples/acoustics/04_full_exhaust_system.morph` — Complete system
- `examples/acoustics/05_inverse_design.morph` — Optimization

### Tests
- Reflection coefficient accuracy
- Waveguide stability (CFL condition)
- Audio output SNR
- Determinism (same geometry → same audio)

---

## Conclusion

**2-stroke muffler modeling is a perfect Morphogen showcase** because it:

1. ✅ **Demonstrates multi-domain strength** — 6 domains, deeply coupled
2. ✅ **Shows geometry-physics integration** — TiaCAD principles applied to acoustics
3. ✅ **Produces compelling outputs** — Realistic engine sounds, performance metrics
4. ✅ **Solves a real problem** — Professional exhaust designers need this
5. ✅ **Is uniquely suited to Morphogen** — No other system unifies these domains

This is not just "can Morphogen do this?" — this is **"Morphogen does this better than anything else."**

---

## References

### Academic
- **"Design of Racing and High-Performance Engines"** — Blair (1999)
- **"Two-Stroke Engine Exhaust Systems"** — Gorr, Benson
- **"Acoustic Wave Propagation in Ducts"** — Munjal (2014)
- **"Digital Waveguide Networks for Acoustic Modeling"** — Van Duyne, Smith

### Software
- **Ricardo WAVE** — 1D engine simulation (commercial)
- **Yamaha VL1** — Physical modeling synthesizer (uses waveguides)
- **Python-acoustics** — Acoustic modeling library

### Morphogen Docs
- `docs/architecture/domain-architecture.md` — Domain overview
- `docs/specifications/geometry.md` — Geometry domain (TiaCAD patterns)
- `docs/specifications/transform.md` — Transform operations (FFT, STFT)
- `docs/adr/002-cross-domain-architectural-patterns.md` — Cross-domain patterns

---

**Status:** Vision document — ready for domain design phase
**Next Steps:**
1. Design FluidDynamicsDomain operators
2. Design AcousticsDomain operators
3. Prototype 1D waveguide implementation
4. Create simple pipe resonance example

---

**End of Document**
