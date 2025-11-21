# ADR 003: Circuit Modeling Domain - EE/Analog/Digital/PCB Simulation

**Status:** IN PROGRESS (Phase 1 Complete)
**Date:** 2025-11-15
**Last Updated:** 2025-11-20
**Authors:** Morphogen Architecture Team
**Supersedes:** N/A
**Related:** ADR-002 (Cross-Domain Patterns), ../specifications/circuit.md

---

## Context

The circuit modeling domain represents one of the most natural fits for Morphogen's multi-domain architecture. Electrical engineering inherently combines:

- **Differential equations** (transient analysis, state-space models)
- **Spatial geometry** (PCB traces, component placement, routing)
- **Physics** (electromagnetic fields, inductance, capacitance, thermal coupling)
- **Audio/analog modeling** (tubes, op-amps, guitar pedals, synthesizers)
- **Discrete-time simulation** (SPICE-like solvers, AC/DC analysis)
- **Nonlinear systems** (diodes, BJTs, MOSFETs, saturation)
- **Constraints solving** (Kirchhoff's laws, component value ranges)
- **Multi-domain coupling** (heat, power, electromagnetics, mechanical)

Current tools in this space are fragmented:
- **SPICE simulators** (LTspice, ngspice) - excellent at circuit simulation, but no geometry integration
- **PCB CAD tools** (KiCad, Altium) - excellent at layout, but weak physics modeling
- **EM solvers** (HFSS, Sonnet) - excellent at fields, but disconnected from circuit simulation
- **Audio modeling** (JUCE, Max/MSP) - excellent at DSP, but no physical circuit modeling

**No existing tool unifies**: circuit simulation + PCB geometry + electromagnetics + audio modeling + pattern generation in a single declarative framework.

Morphogen can fill this gap by treating circuits as **typed operator graphs** with **multi-physics coupling**, **geometry integration**, and **cross-domain flows**.

---

## Decision

We will implement a **Circuit Modeling Domain** as a Layer 4/5 hybrid domain with the following architecture:

### 1. Core Domain Position

**Circuit Domain Placement:**
```
Layer 4: Physics/Fields (base layer)
  ├── Field operators (EM field solvers, inductance, capacitance)
  ├── PDE/ODE solvers (differential equation kernels)
  └── Constraint solvers (KCL, KVL enforcement)

Layer 5: Circuit/DSP (application layer)
  ├── Circuit operators (R, L, C, diodes, transistors, op-amps)
  ├── Audio operators (oscillators, filters, effects)
  └── PCB operators (trace routing, impedance calculation)
```

**Rationale:** Circuit modeling sits at the intersection of physics (differential equations) and DSP (audio signals), making it a natural bridge domain.

---

### 2. Unified Reference System: `CircuitRef`

Following ADR-002's pattern (one reference type per domain), the Circuit domain uses **`CircuitRef`** as its primary reference.

```python
class CircuitRef:
    """Unified reference to circuit nodes, components, and nets."""

    # Reference target types
    RefType = Enum("node", "component", "net", "port")

    # Auto-generated anchors (examples):
    # - component.port["input"]
    # - component.port["output"]
    # - component.port["ground"]
    # - component.param["resistance"]
    # - net.voltage
    # - net.current
    # - node.connections
```

**Auto-Anchors for Circuit Components:**

| Component | Auto-Anchors | Example Usage |
|-----------|--------------|---------------|
| **Resistor** | `.port["p"]`, `.port["n"]`, `.param["R"]` | `r1.port["p"].voltage` |
| **Capacitor** | `.port["p"]`, `.port["n"]`, `.param["C"]`, `.energy` | `c1.param["C"] = 10e-6` |
| **Op-Amp** | `.port["in+"]`, `.port["in-"]`, `.port["out"]`, `.port["vcc"]`, `.port["vee"]` | `opamp.port["out"] >> load.port["in"]` |
| **Transistor (BJT)** | `.port["base"]`, `.port["collector"]`, `.port["emitter"]`, `.beta`, `.vbe` | `bjt.port["collector"].current` |
| **Net** | `.voltage`, `.current`, `.nodes[]`, `.impedance` | `net.voltage.at_time(1e-3)` |
| **PCB Trace** | `.start`, `.end`, `.path[]`, `.width`, `.inductance`, `.capacitance` | `trace.inductance.compute()` |

**Example Usage:**
```python
# Create components
r1 = Resistor(value=1000)  # 1kΩ
c1 = Capacitor(value=10e-6)  # 10µF
opamp = OpAmp(model="tl072")

# Connect using auto-anchors
input_net = Net(name="input")
output_net = Net(name="output")

input_net.connect(r1.port["p"])
r1.port["n"].connect(opamp.port["in+"])
opamp.port["out"].connect(output_net)

# Access derived properties via anchors
gain = opamp.param["gain"]  # Auto-computed from feedback network
bandwidth = opamp.bandwidth  # Auto-anchor for -3dB point
```

**Benefits:**
- Eliminates manual node numbering (SPICE netlist style)
- Type-safe connections (can't connect incompatible ports)
- Auto-computed derived properties (impedance, gain, etc.)
- Seamless integration with PCB geometry domain

---

### 3. Circuit Operator Catalog (4-Layer Model)

Following ADR-002's multi-layer complexity model:

#### **Layer 1: Atomic Circuit Operators**

Low-level circuit primitives (no dependencies):

**Linear Components:**
```
resistor(R: f32<Ω>) -> Component
capacitor(C: f32<F>) -> Component
inductor(L: f32<H>) -> Component
voltage_source(V: f32<V>) -> Component
current_source(I: f32<A>) -> Component
```

**Nonlinear Components:**
```
diode(Is: f32<A>, n: f32) -> Component  # Shockley equation
bjt_npn(beta: f32, Vbe: f32<V>) -> Component
mosfet_n(Vth: f32<V>, kn: f32) -> Component
```

**Circuit Analysis Primitives:**
```
nodal_analysis(circuit: Circuit) -> LinearSystem
modified_nodal_analysis(circuit: Circuit) -> LinearSystem
newton_raphson_step(system: NonlinearSystem, x: Vec<f32>) -> Vec<f32>
```

#### **Layer 2: Composite Circuit Operators**

Combine 2-5 atomic ops:

```
voltage_divider(R1: f32<Ω>, R2: f32<Ω>) -> Component
rc_filter(R: f32<Ω>, C: f32<F>, type: Enum["lpf", "hpf"]) -> Component
rlc_resonator(R: f32<Ω>, L: f32<H>, C: f32<F>) -> Component
dc_bias_network(Vcc: f32<V>, Vbias: f32<V>) -> Component
```

**Solver Integrators:**
```
transient_solve(circuit: Circuit, duration: f32<s>, method: Enum["euler", "rk4", "backward_euler"]) -> TimeSeries
ac_sweep(circuit: Circuit, freq_start: f32<Hz>, freq_end: f32<Hz>, points: i32) -> FrequencyResponse
dc_operating_point(circuit: Circuit) -> StateVector
```

#### **Layer 3: Circuit Constructs/DSL**

Domain-specific patterns (10-50 ops):

**Audio Circuits:**
```
opamp_inverting_amp(gain: f32, input_impedance: f32<Ω>) -> Circuit
opamp_non_inverting_amp(gain: f32) -> Circuit
sallen_key_filter(fc: f32<Hz>, Q: f32, type: Enum["lpf", "hpf", "bpf"]) -> Circuit
twin_t_notch(fc: f32<Hz>, depth: f32) -> Circuit
```

**Power Circuits:**
```
buck_converter(Vin: f32<V>, Vout: f32<V>, Iout: f32<A>, fsw: f32<Hz>) -> Circuit
ldo_regulator(Vin: f32<V>, Vout: f32<V>, Imax: f32<A>) -> Circuit
```

**RF Circuits:**
```
pi_matching_network(Zin: Complex<Ω>, Zout: Complex<Ω>, freq: f32<Hz>) -> Circuit
bandpass_filter_butterworth(fc: f32<Hz>, bw: f32<Hz>, order: i32) -> Circuit
```

**Tube Amplifiers:**
```
triode_stage(tube_model: str, bias: f32<V>, load: f32<Ω>) -> Circuit
pentode_output_stage(tube_model: str, class: Enum["A", "AB", "B"]) -> Circuit
```

#### **Layer 4: Circuit Presets**

Pre-configured systems (50+ ops):

```
guitar_pedal_overdrive(drive: f32, tone: f32, level: f32) -> Circuit
tube_amp_preamp(channels: i32, gain_stages: i32) -> Circuit
modular_synth_vcf(type: Enum["moog", "arp", "roland"]) -> Circuit
audio_mixer_channel_strip(eq_bands: i32, dynamics: bool) -> Circuit
```

**PCB + Circuit Combined:**
```
pcb_circuit_coupled(
    schematic: Circuit,
    layout: PCBLayout,
    coupling_mode: Enum["parasitic_extraction", "full_em_solve"]
) -> CoupledSystem
```

---

### 4. Circuit Domain Passes

Domain-specific optimization and lowering passes:

#### **Validation Passes**

```python
class KirchhoffLawsPass(CircuitPass):
    """Validate KCL (current) and KVL (voltage) at all nodes."""

    def validate_kcl(self, node: CircuitRef):
        # Sum of currents into node = 0
        pass

    def validate_kvl(self, loop: List[CircuitRef]):
        # Sum of voltages around loop = 0
        pass
```

```python
class ComponentValueRangePass(CircuitPass):
    """Ensure component values are physically reasonable."""

    def validate_component(self, comp: Component):
        # R > 0, C > 0, L > 0
        # Voltage sources finite
        # No divide-by-zero in gain calculations
        pass
```

#### **Optimization Passes**

```python
class SeriesParallelReductionPass(CircuitPass):
    """Simplify series/parallel R, L, C combinations."""

    def visit_resistor_chain(self, resistors: List[Resistor]):
        if all_in_series(resistors):
            return Resistor(R=sum(r.R for r in resistors))
        elif all_in_parallel(resistors):
            return Resistor(R=1 / sum(1/r.R for r in resistors))
```

```python
class TheveninNortonEquivalentPass(CircuitPass):
    """Replace complex networks with Thevenin/Norton equivalents."""

    def simplify_two_port(self, network: Circuit):
        # Compute Vth, Rth (Thevenin)
        # Or In, Rn (Norton)
        pass
```

```python
class SymbolicSimplificationPass(CircuitPass):
    """Symbolic preprocessing before numeric solve."""

    def simplify_transfer_function(self, circuit: Circuit):
        # H(s) = Vout(s) / Vin(s)
        # Factor polynomials, cancel poles/zeros
        pass
```

#### **Lowering Passes**

```python
class CircuitToODESystemPass(LoweringPass):
    """Convert circuit netlist to state-space ODE system."""

    def lower(self, circuit: Circuit) -> ODESystem:
        # Modified nodal analysis (MNA)
        # dx/dt = Ax + Bu
        # y = Cx + Du
        return ode_system
```

```python
class ODEToMLIRPass(LoweringPass):
    """Lower ODE system to MLIR scf/linalg dialects."""

    def lower(self, ode: ODESystem) -> MLIRModule:
        # State-space to loops + linear algebra
        pass
```

```python
class PCBGeometryToEMFieldPass(LoweringPass):
    """Extract EM fields from PCB trace geometry."""

    def compute_trace_inductance(self, trace: PCBTrace):
        # FastHenry-like approximation or full EM solve
        pass

    def compute_trace_capacitance(self, trace: PCBTrace):
        # FastCap-like approximation
        pass
```

---

### 5. Multi-Domain Integration

Circuit domain integrates seamlessly with other Morphogen domains:

#### **Circuit ↔ Geometry (PCB Layout)**

```python
# Define PCB geometry
pcb = GeometryDomain.rectangle(width=100, height=60, units="mm")

# Define circuit schematic
circuit = CircuitDomain.opamp_amplifier(gain=10)

# Route traces using geometry domain
trace_input = GeometryDomain.trace(
    start=pcb.anchor["edge_left"].at(y=30),
    end=circuit.component["opamp"].port["in+"].location,
    width=0.254,  # 10 mil
    layer="top_copper"
)

# Compute parasitic inductance from geometry
trace_input.inductance = CircuitDomain.compute_trace_inductance(trace_input.path)

# Add inductance to circuit model
circuit.add_parasitic(
    component=Inductor(L=trace_input.inductance),
    at=circuit.net["input"]
)
```

#### **Circuit ↔ Audio (Analog Modeling)**

```python
# Circuit: Guitar pedal model
pedal = CircuitDomain.guitar_pedal_overdrive(
    drive=0.7,
    tone=0.5,
    level=0.8
)

# Audio: Input signal
guitar = AudioDomain.load_sample("guitar_riff.wav")

# Simulate circuit with audio input
output = CircuitDomain.transient_solve(
    circuit=pedal,
    input=guitar,
    sample_rate=96000,
    duration=guitar.duration
)

# Audio: Add post-processing
reverb = AudioDomain.reverb(mix=0.2)
final = output >> reverb

# Export
AudioDomain.export(final, "pedal_output.wav")
```

#### **Circuit ↔ Physics (Thermal Coupling)**

```python
# Circuit: Power amplifier
amp = CircuitDomain.class_ab_amplifier(power=100)  # 100W

# Compute power dissipation in transistors
power_loss = CircuitDomain.power_dissipation(amp.component["output_stage"])

# Physics: Thermal model
heatsink = PhysicsDomain.thermal_model(
    material="aluminum",
    geometry=heatsink_geometry,
    ambient_temp=25  # °C
)

# Couple power dissipation to heat source
PhysicsDomain.add_heat_source(
    model=heatsink,
    power=power_loss,
    location=amp.component["output_stage"].location
)

# Solve thermal distribution
temp_distribution = PhysicsDomain.steady_state_heat(heatsink)

# Feed temperature back to circuit (temperature-dependent parameters)
amp.component["output_stage"].temperature = temp_distribution.at(
    amp.component["output_stage"].location
)
```

#### **Circuit ↔ Pattern (Modulation)**

```python
# Pattern: PWM generator (from Strudel-like pattern domain)
pwm = PatternDomain.pulse_width_modulation(
    frequency=100e3,  # 100 kHz
    duty_cycle=PatternDomain.sine(freq=1000)  # 1 kHz modulation
)

# Circuit: Buck converter
buck = CircuitDomain.buck_converter(
    Vin=12,
    Vout=5,
    Iout=2,
    fsw=100e3
)

# Drive buck converter with PWM pattern
buck.switch_control = pwm

# Simulate
output_voltage = CircuitDomain.transient_solve(
    circuit=buck,
    duration=10e-3  # 10 ms
)
```

---

### 6. Solver Architecture

The circuit domain requires a sophisticated solver system:

#### **Solver Hierarchy**

```
Layer 1: Direct Solvers
├── LU decomposition (dense)
├── QR decomposition
└── Cholesky (symmetric positive definite)

Layer 2: Iterative Solvers
├── Conjugate gradient (sparse)
├── GMRES (nonsymmetric)
└── BiCGSTAB

Layer 3: Nonlinear Solvers
├── Newton-Raphson
├── Modified Newton
└── Damped Newton (for robustness)

Layer 4: Time Integration
├── Forward Euler (explicit, 1st order)
├── Backward Euler (implicit, stable)
├── Trapezoidal rule (implicit, 2nd order)
└── Runge-Kutta 4 (explicit, 4th order)
```

#### **Solver Selection Pass**

```python
class SolverSelectionPass(CircuitPass):
    """Automatically select best solver based on circuit characteristics."""

    def select_solver(self, circuit: Circuit):
        # Small circuit (< 100 nodes) → direct solver
        # Large circuit (> 1000 nodes) → iterative solver
        # Stiff system → implicit integrator
        # High-frequency → small timestep + RK4
        pass
```

#### **GPU Acceleration**

```python
class CircuitToGPUPass(LoweringPass):
    """Offload large circuits to GPU."""

    def can_gpu_accelerate(self, circuit: Circuit):
        # Large sparse matrices benefit from GPU
        # Small dense systems stay on CPU
        pass
```

---

### 7. Novel Capabilities (What Existing Tools Can't Do)

#### **1. Unified Analog/Digital/Audio/Geometry Simulation**

```yaml
# Morphogen circuit specification (YAML syntax)

pcb:
  shape: rect [100, 60]
  units: mm
  traces:
    - net: input
      route: auto
      width: 10mil
      impedance_target: 50Ω

circuit:
  components:
    - id: amp
      type: op_amp
      model: "tl072"

    - id: load
      type: speaker
      model: "eminence_12inch"
      impedance: 8Ω

  nets:
    input: jack.tip
    output: speaker.positive

simulate:
  transient:
    duration: 5ms
    input: audio_sample("guitar.wav")
    method: rk4

  coupling:
    - pcb_parasitics: true
    - thermal: true
    - em_fields: false  # Optional full-wave EM

audio:
  post_fx:
    - reverb: {mix: 0.2}
    - eq: {low: -3dB, high: +2dB}

export:
  - schematic: "amp.svg"
  - pcb: "amp.kicad_pcb"
  - audio: "amp_output.wav"
  - spice: "amp.cir"
```

**This simulates:**
- PCB geometry
- Circuit schematic
- Amplifier nonlinearities
- Speaker electrical response
- Audio signal processing
- Electromagnetic coupling
- Thermal effects

**No existing tool does this in one framework.**

#### **2. Gradient-Based Circuit Optimization**

```python
# ML domain integration
circuit = CircuitDomain.opamp_filter(fc=1000, Q=2.0)

# Define optimization target
target_response = FrequencyResponse(...)  # Desired frequency response

# Use ML domain to optimize component values via gradient descent
optimized_circuit = MLDomain.optimize(
    circuit=circuit,
    parameters=["R1", "R2", "C1", "C2"],
    objective=lambda c: loss(c.frequency_response(), target_response),
    optimizer="adam",
    learning_rate=0.01
)
```

**Novel:** Circuit design via differentiable programming (like JAX, but for circuits).

#### **3. Tube Amplifier Modeling with Audio Integration**

```python
# Triode amplifier stage
triode = CircuitDomain.triode_stage(
    tube_model="12ax7",
    bias_voltage=-1.5,
    plate_load=100e3
)

# Include cathode bypass capacitor
cathode_bypass = Capacitor(value=25e-6)

# Audio input
guitar = AudioDomain.load("guitar.wav")

# Simulate with oversampling (to capture harmonics)
output = CircuitDomain.transient_solve(
    circuit=triode,
    input=guitar,
    sample_rate=192000,  # 4x oversampling
    method="backward_euler"
)

# Downsample and export
final = AudioDomain.resample(output, target_rate=48000)
```

**Novel:** Physical tube modeling (not just lookup tables or waveshapers).

#### **4. PCB Trace Inductance → Circuit Co-Simulation**

```python
# PCB geometry
pcb = GeometryDomain.pcb_layout(layers=4)

# High-speed digital circuit
circuit = CircuitDomain.digital_driver(
    rise_time=1e-9,  # 1ns edges
    drive_strength=25e-3  # 25mA
)

# Route trace (manual or auto)
trace = GeometryDomain.trace(
    start=circuit.output,
    end=connector.pin[1],
    width=0.127,  # 5 mil
    length=50  # mm
)

# Compute trace inductance and capacitance
L_trace = CircuitDomain.compute_inductance(trace, method="fasthenry")
C_trace = CircuitDomain.compute_capacitance(trace, method="fastcap")

# Add transmission line model to circuit
circuit.add_transmission_line(
    L=L_trace,
    C=C_trace,
    length=trace.length
)

# Simulate signal integrity
waveform = CircuitDomain.transient_solve(circuit, duration=10e-9)

# Check for ringing, reflections
assert waveform.overshoot < 0.1  # < 10% overshoot
```

**Novel:** Seamless geometry → EM → circuit flow in one framework.

---

## Implementation Roadmap

### Phase 1: Foundation (Months 1-2) ✅ COMPLETE

**Goal:** Basic circuit simulation (SPICE-like)

- [x] ~~Define `CircuitRef` and auto-anchors~~ (Deferred to Phase 2 - using node indices for now)
- [x] Implement Layer 1 operators (R, L, C, voltage source, current source)
- [x] Implement modified nodal analysis (MNA)
- [x] Implement direct solvers (LU decomposition)
- [x] Implement basic time integrators (backward Euler)
- [ ] Write validation passes (KCL, KVL) - TODO
- [ ] Write unit tests for linear circuits (RC, RL, RLC) - TODO

**Deliverable:** ✅ Working RC filter example (`examples/circuit/01_rc_filter_basic.py`)

**Implementation:** See `morphogen/stdlib/circuit.py` and [CIRCUIT_DOMAIN_IMPLEMENTATION.md](../CIRCUIT_DOMAIN_IMPLEMENTATION.md)

---

### Phase 2: Nonlinear Components (Months 3-4)

**Goal:** Diodes, transistors, op-amps

- [ ] Implement Layer 1 nonlinear operators (diode, BJT, MOSFET)
- [ ] Implement Newton-Raphson solver
- [ ] Implement op-amp models (ideal, macromodel, SPICE-based)
- [ ] Write optimization passes (Thevenin/Norton equivalents)
- [ ] Write unit tests for nonlinear circuits (rectifier, amplifier, oscillator)

**Deliverable:** Working op-amp amplifier, BJT amplifier examples

---

### Phase 3: Audio Integration (Months 5-6)

**Goal:** Analog audio modeling

- [ ] Implement Layer 3 audio circuit constructs (filters, preamps)
- [ ] Implement tube models (triode, pentode)
- [ ] Integrate with AudioDomain (audio input/output)
- [ ] Implement oversampling for harmonic accuracy
- [ ] Write examples: guitar pedal, tube preamp, speaker model

**Deliverable:** Guitar pedal simulator, tube amp simulator

---

### Phase 4: PCB Geometry Integration (Months 7-9)

**Goal:** Couple circuit simulation with PCB layout

- [ ] Define PCB geometry operators (traces, pads, vias)
- [ ] Implement parasitic extraction (inductance, capacitance, resistance)
- [ ] Implement FastHenry/FastCap-like approximations
- [ ] Write cross-domain passes (geometry → circuit)
- [ ] Write examples: PCB trace inductance, signal integrity

**Deliverable:** PCB trace parasitic extraction, coupled simulation

---

### Phase 5: Advanced Features (Months 10-12)

**Goal:** Multi-physics, optimization, RF

- [ ] Implement AC sweep analysis (frequency response)
- [ ] Implement harmonic balance (for RF circuits)
- [ ] Implement thermal coupling (circuit → heat)
- [ ] Implement ML-based optimization (gradient descent on component values)
- [ ] Implement S-parameter import/export
- [ ] Write examples: buck converter, RF matching network, thermal analysis

**Deliverable:** Full-featured circuit simulator with multi-physics

---

## Consequences

### Positive

1. **Unique Positioning**
   - Only tool to unify circuit + PCB + audio + EM in one framework
   - Enables workflows impossible in SPICE, CAD, or audio tools alone
   - Natural fit for Morphogen's operator graph architecture

2. **Cross-Domain Synergy**
   - Audio domain benefits from physical circuit modeling
   - Geometry domain benefits from electrical constraints (impedance, trace width)
   - Physics domain benefits from circuit examples (EM fields, thermal)
   - ML domain benefits from circuit optimization examples

3. **Community Expansion**
   - EE community is large and underserved by modern tools
   - Analog audio community (pedal builders, tube amp designers) eager for better tools
   - PCB designers frustrated by disconnected tools

4. **Technical Rigor**
   - Proven SPICE algorithms (MNA, Newton-Raphson)
   - Validated against existing simulators (ngspice, LTspice)
   - Type safety and unit checking prevent common errors

5. **Progressive Complexity**
   - Layer 4 presets (guitar_pedal_overdrive) for beginners
   - Layer 1 atomic ops (resistor, capacitor) for experts
   - Smooth learning curve

### Negative

1. **Implementation Complexity**
   - Circuit simulation is a mature, complex field
   - Nonlinear solvers require careful tuning for robustness
   - EM field solvers are computationally expensive
   - Validation against existing simulators is time-consuming

2. **Performance Expectations**
   - Users will compare performance to highly optimized SPICE engines
   - Large circuits (> 10,000 nodes) require sparse matrix optimization
   - GPU acceleration non-trivial for sparse linear systems

3. **Component Model Library**
   - Need comprehensive library of SPICE models (diodes, BJTs, MOSFETs, op-amps)
   - Tube models require empirical data (plate curves, transconductance)
   - Speaker models require electromechanical parameters (Thiele-Small)

4. **Standards Compliance**
   - Need to support standard formats (SPICE netlist, Touchstone S2P, IBIS)
   - KiCad/Altium import/export for PCB integration
   - Audio export (WAV, FLAC) for analog modeling

### Mitigations

1. **Leverage Existing Libraries**
   - Use ngspice as reference implementation
   - Import SPICE model library (diodes, transistors, op-amps)
   - Use FastHenry/FastCap algorithms for EM approximations

2. **Start Simple, Expand Incrementally**
   - Phase 1: Linear circuits only (validation against known solutions)
   - Phase 2: Nonlinear components (validation against SPICE)
   - Phase 3: Audio integration (validation against physical pedals/amps)
   - Phase 4+: Advanced features (PCB, EM, thermal)

3. **Benchmark and Optimize**
   - Profile solver performance (CPU, GPU)
   - Implement sparse matrix optimizations (CSR/CSC format)
   - Use adaptive timestep for stiff systems

4. **Build Community**
   - Release early with clear limitations
   - Invite EE/audio community to contribute models
   - Provide migration path from SPICE netlists

---

## Success Criteria

The Circuit domain will be considered successful if:

1. **Functional Parity:**
   - Can simulate all circuits that ngspice can (RC, RLC, op-amp, BJT, MOSFET)
   - Results match SPICE within 1% for linear circuits, 5% for nonlinear

2. **Unique Capabilities:**
   - Demonstrates at least 3 workflows impossible in existing tools:
     - Circuit + PCB parasitic co-simulation
     - Analog circuit → audio export
     - ML-based circuit optimization

3. **Performance:**
   - Small circuits (< 100 nodes): < 1 second for transient analysis
   - Medium circuits (100-1000 nodes): < 10 seconds
   - Large circuits (> 1000 nodes): GPU acceleration available

4. **Community Adoption:**
   - 10+ community-contributed circuit examples (pedals, amps, power supplies)
   - 5+ third-party component model libraries
   - 100+ users in EE/audio community

5. **Documentation:**
   - Complete ../specifications/circuit.md
   - 20+ tutorial examples (RC filter → full guitar amp)
   - Migration guide from SPICE

---

## References

- **ADR-002:** Cross-Domain Architectural Patterns
- **../architecture/domain-architecture.md:** Morphogen domain vision
- **../specifications/operator-registry.md:** Operator metadata schema
- **../specifications/circuit.md:** Circuit domain specification (to be written)
- **ngspice:** Open-source SPICE simulator (reference implementation)
- **FastHenry:** MIT tool for inductance extraction
- **FastCap:** MIT tool for capacitance extraction

---

## Appendix: Why EE/Circuit Modeling is Perfect for Morphogen

### 1. Circuits Are Operator Graphs

Electrical circuits are fundamentally **typed operator graphs**:

```
Voltage Source → Resistor → Capacitor → Op Amp → Speaker Load
```

Each component:
- Has **typed ports** (positive, negative, signal, ground)
- Has **constraints** (KCL: sum of currents = 0, KVL: sum of voltages = 0)
- Has **internal relations** (I = C dV/dt, V = L di/dt, nonlinear transfer functions)

This is **identical** to Morphogen's operator registry design:
- Operators have typed inputs/outputs
- Operators have constraints (unit compatibility)
- Operators have internal transformations

### 2. Circuit Simulation is Multi-Domain Physics

Circuit simulation requires:
- **ODE solving** (transient analysis: capacitor charge, inductor current)
- **Linear system solving** (DC operating point: Ax = b)
- **Nonlinear system solving** (Newton-Raphson for diodes, transistors)
- **Multi-physics coupling** (thermal, EM fields, mechanical vibration)

Morphogen's Layer 4 (Physics/Fields) provides:
- PDE/ODE solvers
- Integrators (Euler, RK4, implicit methods)
- Constraint solvers
- Multi-domain coupling infrastructure

**Circuit simulation is just physics with electrical components.**

### 3. PCB Geometry = Geometry Domain

PCB layout involves:
- **2D/3D geometry** (board shape, component placement)
- **Path planning** (trace routing, via placement)
- **Spatial constraints** (clearance, trace width)
- **Derived properties** (trace inductance, capacitance, impedance)

TiaCAD's `SpatialRef` system is **perfect** for:
- PCB shape definition
- Trace routing with anchors (start, end, waypoints)
- Ground pour generation
- Impedance-controlled routing
- Coupling/inductance analysis

**GeometryDomain + CircuitDomain = full PCB CAD.**

### 4. Audio Modeling = Circuit + DSP

Analog audio circuits (guitar pedals, tube amps, synthesizers) require:
- **Circuit simulation** (op-amp stages, clipping, biasing)
- **Nonlinear modeling** (tube saturation, diode clipping)
- **DSP** (filtering, modulation, reverb)
- **Real-time processing** (audio-rate sample generation)

Morphogen can unify:
- **CircuitDomain:** Physical circuit modeling (SPICE-like)
- **AudioDomain:** DSP effects, filters, oscillators
- **Cross-domain flow:** Circuit output → Audio post-processing

**Example:**
```python
pedal = CircuitDomain.tube_screamer(drive=0.8)
guitar = AudioDomain.load("riff.wav")
output = pedal.process(guitar)  # Circuit simulation
reverb = AudioDomain.reverb(output, mix=0.3)  # DSP post-processing
```

**Only Morphogen can do this seamlessly.**

### 5. Exploration + Optimization

Morphogen's cross-domain flows enable **novel workflows**:

**Circuit → ML Optimization:**
```python
# Optimize RC filter for target frequency response
circuit = CircuitDomain.rc_filter(R=1000, C=1e-6)
target = FrequencyResponse(...)

optimized = MLDomain.gradient_descent(
    circuit=circuit,
    params=["R", "C"],
    objective=lambda c: loss(c.freq_response(), target)
)
```

**Circuit → Physics (Thermal):**
```python
# Couple power transistor heat to heatsink
amp = CircuitDomain.class_ab_amplifier(power=100)
power_loss = amp.component["output_stage"].power_dissipation

heatsink = PhysicsDomain.thermal_model(...)
PhysicsDomain.add_heat_source(heatsink, power=power_loss)
temp = PhysicsDomain.solve()

# Feed temperature back to circuit (temperature-dependent beta, Vbe)
amp.update_temperature(temp)
```

**Pattern → Circuit (Modulation):**
```python
# PWM pattern from Strudel-like sequencer
pwm = PatternDomain.pwm(freq=100e3, duty=PatternDomain.sine(1000))

# Drive buck converter switch
buck = CircuitDomain.buck_converter(...)
buck.switch_control = pwm
```

**These workflows don't exist in SPICE, CAD tools, or audio software.**

---

**Conclusion:**

Circuit modeling is **the most natural domain for Morphogen** outside of geometry and audio, because it:
1. Perfectly matches Morphogen's operator graph architecture
2. Requires multi-domain physics (ODEs, linear systems, nonlinear solvers)
3. Integrates seamlessly with geometry (PCB layout)
4. Integrates seamlessly with audio (analog modeling)
5. Enables novel cross-domain workflows (ML optimization, thermal coupling, pattern-driven modulation)

**No existing tool can compete with Morphogen's unified architecture for circuit modeling.**

This is Morphogen's **killer app** for the EE and analog audio communities.
