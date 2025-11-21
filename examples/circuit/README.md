# Kairo Circuit Modeling Examples

This directory contains example circuit specifications demonstrating Kairo's circuit modeling capabilities.

## Overview

Kairo.Circuit enables **unified circuit simulation, PCB layout, analog audio modeling, and multi-physics coupling** in a single declarative framework. These examples showcase capabilities that **don't exist in any other tool** (SPICE, LTspice, KiCad, etc.).

---

## Examples

### 01. RC Low-Pass Filter (`01_rc_filter.kairo`)

**Demonstrates:**
- Basic passive circuit simulation (R, L, C)
- DC operating point
- AC frequency sweep (Bode plot)
- Transient analysis (step response)
- Auto-computed properties (cutoff frequency, time constant)
- Assertions for validation

**Key concepts:**
- Reference-based connections (no manual node numbering)
- Auto-anchors (`.port["p"]`, `.port["n"]`)
- Unit-safe parameters (`Ω`, `F`, `Hz`, `s`)

**Run:**
```bash
kairo simulate 01_rc_filter.kairo
```

---

### 02. Op-Amp Inverting Amplifier (`02_opamp_amplifier.kairo`)

**Demonstrates:**
- Nonlinear component modeling (op-amp)
- Multi-stage circuits
- Power supply modeling
- DC bias analysis
- Frequency response (gain, phase, bandwidth)
- Saturation and clipping checks

**Key concepts:**
- Op-amp models with realistic parameters (gain, bandwidth, slew rate)
- Computed parameters (feedback resistor from gain)
- Validation (no clipping, DC offset near zero)

**Run:**
```bash
kairo simulate 02_opamp_amplifier.kairo
```

---

### 03. Guitar Pedal (Tube Screamer) (`03_guitar_pedal.kairo`)

**Demonstrates:**
- **Circuit → Audio domain integration** (unique to Kairo!)
- Asymmetric diode clipping (harmonic generation)
- Nonlinear modeling (diodes, op-amps)
- Audio input/output
- Oversampling for harmonic accuracy
- Audio post-processing (reverb, resampling)
- Harmonic distortion analysis (THD, FFT)

**Key concepts:**
- Load audio sample as circuit input
- 4x oversampling to capture harmonics
- Export audio output (WAV)
- Cross-domain flow: Circuit → Audio

**Run:**
```bash
kairo simulate 03_guitar_pedal.kairo --audio-input guitar_riff.wav
```

**Output:**
- `pedal_output.wav` - Processed audio
- `tube_screamer.cir` - SPICE netlist
- `tube_screamer.svg` - Schematic

---

### 04. PCB Trace Inductance (`04_pcb_trace_inductance.kairo`)

**Demonstrates:**
- **Circuit + Geometry domain integration** (unique to Kairo!)
- PCB trace parasitic extraction (L, C, R)
- Transmission line modeling
- Signal integrity analysis (overshoot, undershoot, rise time)
- Eye diagram generation
- FastHenry/FastCap algorithms

**Key concepts:**
- Define PCB geometry (board, trace, ground plane)
- Extract parasitics from geometry
- Add parasitics to circuit model
- Signal integrity metrics (Z0, overshoot, eye diagram)
- Export to KiCad, Touchstone S2P

**Run:**
```bash
kairo simulate 04_pcb_trace_inductance.kairo
```

**Output:**
- `pcb_trace.cir` - SPICE netlist
- `trace.s2p` - S-parameters (Touchstone)
- `pcb_layout.kicad_pcb` - KiCad PCB layout

---

### 05. Unified Multi-Domain Example (`05_unified_example.kairo`)

**Demonstrates:**
- **ALL domains integrated** (Circuit + PCB + Audio + EM + Thermal + Pattern)
- Complete guitar amplifier (preamp + power amp + speaker)
- PCB layout with auto-routing
- Parasitic extraction (geometry → circuit)
- Audio input/output (guitar sample → WAV)
- Thermal coupling (power dissipation → heatsink temperature → circuit)
- EM field simulation (mutual inductance, EMI)
- Pattern-driven modulation (tremolo effect)

**This is the KILLER DEMO - no other tool can do this!**

**Key concepts:**
- Multi-domain system specification
- Cross-domain data flows:
  - **Geometry → Circuit:** Parasitic extraction
  - **Circuit → Audio:** Analog modeling
  - **Circuit → Physics:** Thermal coupling
  - **Physics → Circuit:** Temperature feedback
  - **Pattern → Audio:** Modulation
- Complete workflow: PCB design → circuit simulation → audio export

**Run:**
```bash
kairo simulate 05_unified_example.kairo --audio-input guitar_solo.wav --multi-domain
```

**Output:**
- `guitar_amp.cir` - SPICE netlist
- `guitar_amp.kicad_pcb` - PCB layout
- `guitar_amp_gerbers/` - Gerber files (fabrication)
- `guitar_amp_output.wav` - Processed audio
- `guitar_amp_report.pdf` - Comprehensive report (circuit, thermal, SI, audio)

---

## Why This Matters

### What Existing Tools Can't Do

| Tool | Limitation | Kairo Solution |
|------|------------|----------------|
| **SPICE (ngspice, LTspice)** | Circuit simulation only, no PCB integration | Circuit + PCB + parasitics in one framework |
| **KiCad, Altium** | PCB layout only, weak physics modeling | Seamless geometry → circuit flow |
| **HFSS, Sonnet** | EM simulation only, disconnected from circuit | EM fields → circuit parasitics |
| **JUCE, Max/MSP** | Audio DSP, no physical circuit modeling | Circuit → audio with physical modeling |
| **All of the above** | No multi-domain coupling | Circuit + PCB + Audio + Thermal + EM unified |

---

## Running Examples

### Prerequisites

```bash
# Install Kairo
pip install kairo

# Verify installation
kairo --version
```

### Simulate a Circuit

```bash
# Basic simulation
kairo simulate 01_rc_filter.kairo

# With audio input
kairo simulate 03_guitar_pedal.kairo --audio-input guitar.wav

# Multi-domain simulation
kairo simulate 05_unified_example.kairo --multi-domain

# With visualization
kairo simulate 01_rc_filter.kairo --plot
```

### Export Formats

```bash
# SPICE netlist
kairo export 01_rc_filter.kairo --format spice

# KiCad PCB
kairo export 04_pcb_trace_inductance.kairo --format kicad

# Audio (WAV)
kairo export 03_guitar_pedal.kairo --format wav

# All formats
kairo export 05_unified_example.kairo --format all
```

---

## Tutorial: Build Your Own Circuit

### Step 1: Define Components

```kairo
circuit MyFilter {
    components:
        r1 = resistor(R=1kΩ)
        c1 = capacitor(C=100nF)
}
```

### Step 2: Define Connections

```kairo
    nets:
        input, output, ground

    connections:
        input.connect(r1.port["p"])
        r1.port["n"].connect(output)
        output.connect(c1.port["p"])
        c1.port["n"].connect(ground)
```

### Step 3: Add Analysis

```kairo
    analysis:
        ac: ac_sweep(freq_start=10Hz, freq_end=100kHz, points=100)
```

### Step 4: Run Simulation

```bash
kairo simulate my_filter.kairo --plot
```

---

## Cross-Domain Flows

### Circuit → Audio (Analog Modeling)

```kairo
audio_input:
    guitar = AudioDomain.load_sample("riff.wav")

analysis:
    output = CircuitDomain.transient(
        circuit=pedal,
        input=guitar,
        sample_rate=96kHz
    )

audio_output:
    AudioDomain.export(output, "pedal_output.wav")
```

### Geometry → Circuit (Parasitic Extraction)

```kairo
pcb_geometry:
    trace = GeometryDomain.pcb_trace(...)

parasitic_extraction:
    L = CircuitDomain.compute_trace_inductance(trace)
    C = CircuitDomain.compute_trace_capacitance(trace)

circuit:
    trace_model = transmission_line(L=L, C=C)
```

### Circuit → Physics (Thermal Coupling)

```kairo
circuit_simulation:
    power_loss = amplifier.power_dissipation

thermal_analysis:
    heatsink = PhysicsDomain.thermal_model(...)
    PhysicsDomain.add_heat_source(heatsink, power=power_loss)
    temp = PhysicsDomain.solve()

    # Feed temperature back to circuit
    amplifier.temperature = temp
```

---

## Next Steps

1. **Study the examples** - Start with `01_rc_filter.kairo` and work up
2. **Modify parameters** - Change component values, see how results change
3. **Add your own circuits** - Use examples as templates
4. **Combine domains** - Circuit + Audio, Circuit + PCB, Circuit + Thermal
5. **Contribute** - Submit your circuits to the Kairo example library!

---

## References

- **SPEC-CIRCUIT.md** - Complete circuit domain specification
- **ADR-003** - Circuit domain architecture decision record
- **DOMAIN_ARCHITECTURE.md** - Multi-domain architecture vision
- **Kairo Documentation** - https://kairo.dev/docs

---

## Community

- **GitHub:** https://github.com/kairo-lang/kairo
- **Discord:** https://discord.gg/kairo
- **Forum:** https://forum.kairo.dev

**Questions? Ask in the #circuit-modeling channel on Discord!**
