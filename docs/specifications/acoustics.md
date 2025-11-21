# Acoustics Domain

**Version:** 1.0 (Phase 1)
**Status:** Implemented
**Last Updated:** 2025-11-15

---

## Overview

The Acoustics Domain provides tools for modeling sound wave propagation, resonance, and impedance in 1D, 2D, and 3D acoustic systems. This is a key differentiator for Morphogen, bridging physics simulation (fields, fluids) to audio output.

**Phase 1 Status (v1.0):** 1D waveguide acoustics - COMPLETE ✅

---

## Key Applications

- **Musical Instruments** — Physical modeling synthesis (strings, pipes, brass, woodwinds)
- **Exhaust Systems** — 2-stroke expansion chambers, muffler design, sound prediction
- **Architectural Acoustics** — Room modes, reverberation, absorption
- **Noise Control** — HVAC silencers, industrial mufflers, active cancellation
- **Speaker Design** — Bass reflex ports, transmission lines, directivity control

---

## Phase 1: 1D Waveguide Acoustics

### Implemented Operators

#### Waveguide Construction

**`waveguide_from_geometry(geometry, discretization, sample_rate, speed_of_sound)`**

Build digital waveguide from pipe geometry.

```python
from morphogen.stdlib.acoustics import acoustics, create_pipe

# Simple pipe
pipe = create_pipe(diameter=0.025, length=1.0)  # 25mm x 1m
wg = acoustics.waveguide_from_geometry(pipe, discretization=0.01)

# Expansion chamber (muffler)
from morphogen.stdlib.acoustics import create_expansion_chamber
chamber = create_expansion_chamber(
    inlet_diameter=0.04,
    belly_diameter=0.12,
    outlet_diameter=0.05,
    total_length=1.0
)
wg = acoustics.waveguide_from_geometry(chamber)
```

**`reflection_coefficients(waveguide, end_condition)`**

Compute reflection coefficients at area discontinuities.

```python
# Open end (R ≈ -1.0)
reflections = acoustics.reflection_coefficients(wg, end_condition="open")

# Closed end (R ≈ +1.0)
reflections = acoustics.reflection_coefficients(wg, end_condition="closed")

# Matched impedance (R = 0.0, no reflection)
reflections = acoustics.reflection_coefficients(wg, end_condition="matched")
```

#### Waveguide Propagation

**`waveguide_step(pressure_forward, pressure_backward, waveguide, reflections, excitation, excitation_pos)`**

Single time step of waveguide simulation (digital waveguide algorithm).

```python
import numpy as np

# Initialize
p_fwd = np.zeros(wg.num_segments)
p_bwd = np.zeros(wg.num_segments)

# Simulate impulse
for t in range(num_steps):
    excitation = np.array([1.0]) if t == 0 else None
    p_fwd, p_bwd = acoustics.waveguide_step(
        p_fwd, p_bwd, wg, reflections,
        excitation=excitation, excitation_pos=0
    )
```

**`total_pressure(pressure_forward, pressure_backward)`**

Compute total pressure from bidirectional waves.

```python
p_total = acoustics.total_pressure(p_fwd, p_bwd)
```

#### Helmholtz Resonators

**`helmholtz_frequency(volume, neck_length, neck_area, speed_of_sound)`**

Compute resonant frequency of Helmholtz resonator.

```python
# Quarter-wave resonator for muffler
f_res = acoustics.helmholtz_frequency(
    volume=500e-6,    # 500 cm³
    neck_length=0.05, # 50 mm
    neck_area=20e-4   # 20 cm²
)
# f_res ≈ 690 Hz
```

**`helmholtz_impedance(frequency, volume, neck_length, neck_area, damping, speed_of_sound, air_density)`**

Compute acoustic impedance of Helmholtz resonator.

```python
Z = acoustics.helmholtz_impedance(
    frequency=100.0,
    volume=500e-6,
    neck_length=0.05,
    neck_area=20e-4,
    damping=0.1
)
# Returns complex impedance: Z = R + jX
```

#### Radiation Impedance

**`radiation_impedance_unflanged(diameter, frequency, speed_of_sound, air_density)`**

Radiation impedance for unflanged circular pipe.

```python
Z_rad = acoustics.radiation_impedance_unflanged(
    diameter=0.05,
    frequency=1000.0
)
# Complex impedance for radiation to free space
```

#### Transfer Functions & Frequency Analysis

**`transfer_function(waveguide, reflections, freq_range, resolution, excitation_pos, measurement_pos)`**

Compute acoustic transfer function (input → output).

```python
response = acoustics.transfer_function(
    wg, reflections,
    freq_range=(50, 5000),
    resolution=10
)
# Returns FrequencyResponse with magnitude (dB) and phase (rad)
```

**`resonant_frequencies(frequency_response, threshold_db)`**

Find resonant frequencies (peaks in transfer function).

```python
resonances = acoustics.resonant_frequencies(response, threshold_db=-3.0)
# Returns array of peak frequencies
```

---

## Data Structures

### PipeGeometry

Represents pipe geometry for acoustic modeling.

```python
from morphogen.stdlib.acoustics import PipeGeometry

# Uniform pipe
pipe = PipeGeometry(diameter=0.025, length=1.0)

# Variable diameter pipe
segments = [(0.0, 0.04), (0.5, 0.12), (1.0, 0.05)]
pipe = PipeGeometry(diameter=0.04, length=1.0, segments=segments)
```

### WaveguideNetwork

Digital waveguide network for 1D propagation.

```python
wg = acoustics.waveguide_from_geometry(pipe)
print(f"Segments: {wg.num_segments}")
print(f"Length: {wg.total_length}m")
print(f"Delay: {wg.delay_samples} samples")
```

### ReflectionCoeff

Reflection coefficient at acoustic discontinuity.

```python
@dataclass
class ReflectionCoeff:
    position: int      # Segment index
    coefficient: float # -1.0 (open) to +1.0 (closed)
```

### FrequencyResponse

Frequency response of acoustic system.

```python
response = acoustics.transfer_function(...)
print(f"Frequencies: {response.frequencies}")
print(f"Magnitude: {response.magnitude} (dB)")
print(f"Phase: {response.phase} (rad)")
```

---

## Examples

### Example 1: Open Pipe Resonance

```python
from morphogen.stdlib.acoustics import acoustics, create_pipe

# Create 1m open pipe
pipe = create_pipe(diameter=0.025, length=1.0)
wg = acoustics.waveguide_from_geometry(pipe)
reflections = acoustics.reflection_coefficients(wg, end_condition="open")

# Compute resonances
response = acoustics.transfer_function(wg, reflections)
resonances = acoustics.resonant_frequencies(response)

# Open-open pipe: f_n = n * c / (2L)
# Expected: 171.5 Hz, 343 Hz, 514.5 Hz, ...
print(f"Resonances: {resonances}")
```

### Example 2: Expansion Chamber (Muffler)

```python
from morphogen.stdlib.acoustics import acoustics, create_expansion_chamber

# Create expansion chamber
chamber = create_expansion_chamber(
    inlet_diameter=0.04,
    belly_diameter=0.12,
    outlet_diameter=0.05,
    total_length=1.0
)

# Build waveguide and analyze
wg = acoustics.waveguide_from_geometry(chamber)
reflections = acoustics.reflection_coefficients(wg)

# Simulate impulse response
import numpy as np
p_fwd = np.zeros(wg.num_segments)
p_bwd = np.zeros(wg.num_segments)

output = []
for t in range(500):
    exc = np.array([1.0]) if t == 0 else None
    p_fwd, p_bwd = acoustics.waveguide_step(
        p_fwd, p_bwd, wg, reflections,
        excitation=exc
    )
    p_total = acoustics.total_pressure(p_fwd, p_bwd)
    output.append(p_total[-1])

# Analyze frequency response
response = acoustics.transfer_function(wg, reflections)
```

### Example 3: Complete Simulation

See `examples/acoustics_waveguide_demo.py` for complete demonstrations including:
- Simple pipe resonance
- Expansion chamber simulation
- Helmholtz resonator calculations
- Radiation impedance analysis

---

## Testing

Comprehensive test suite in `tests/test_acoustics.py`:

```bash
pytest tests/test_acoustics.py -v
```

Tests cover:
- Pipe geometry creation
- Waveguide construction (uniform and variable diameter)
- Reflection coefficient calculation
- Waveguide propagation
- Helmholtz resonator formulas
- Radiation impedance
- Transfer function analysis
- Resonant frequency detection
- Complete integration tests

---

## Future Phases

### Phase 2: Enhanced 1D Features (Planned)

- **Coupling from fluid fields** → acoustic propagation
- **Perforated pipes** & absorption materials
- **Lumped acoustic networks** (circuit analogy)
- **Temperature-dependent properties**

### Phase 3: 2D/3D Acoustics (Planned)

- **FDTD acoustics** for complex geometries
- **Room acoustics** simulation
- **Modal analysis**
- **Boundary conditions** (absorbing, reflecting)

### Phase 4: Audio Integration (Planned)

- **Acoustic pressure → audio samples**
- **Microphone models**
- **Real-time synthesis** from physical models

---

## Cross-Domain Integration

The Acoustics domain is designed to integrate seamlessly with other Morphogen domains:

**Geometry → Acoustics:**
```python
# Pipe geometry defines acoustic behavior
pipe = geom.pipe(...)
waveguide = acoustics.waveguide_from_geometry(pipe)
```

**Acoustics → Audio:**
```python
# Acoustic pressure → audio signal (Phase 4)
audio_signal = audio.pressure_to_sample(acoustic_pressure)
```

**FluidDynamics → Acoustics:**
```python
# Combustion pulse → acoustic propagation (Phase 2)
pulse = engine.combustion_pulse(...)
tailpipe_pressure = acoustics.propagate(pulse, exhaust_waveguide)
```

---

## References

### Academic

- **"Acoustics: An Introduction"** — Kinsler, Frey, Coppens, Sanders
- **"Acoustic Wave Propagation in Ducts and Mufflers"** — Munjal (2014)
- **"Digital Waveguide Networks for Acoustic Modeling"** — Van Duyne, Smith (1993)
- **"Physical Audio Signal Processing"** — Julius O. Smith III (online book)

### Commercial Systems

- **Yamaha VL1** — Physical modeling synthesizer (uses digital waveguides)
- **Ricardo WAVE** — 1D engine/exhaust simulation software
- **COMSOL Acoustics Module** — Finite element acoustics
- **GT-POWER** — Engine acoustics simulation

---

## Implementation Notes

### Numerical Stability

- Reflection coefficients slightly below ±1.0 for numerical stability
- Bilinear interpolation for smooth diameter transitions
- Energy-conserving waveguide updates

### Performance

- 1D waveguides: O(N) per timestep where N = segments
- Transfer function: O(F × T) where F = frequencies, T = timestep
- Optimized for real-time when segment count < 1000

### Accuracy

- Digital waveguide accurate to ~20 kHz at 44.1 kHz sample rate
- Discretization should be < λ/10 for accuracy
- Low-frequency approximations for radiation impedance (ka < 2)

---

## Contributing

To extend the Acoustics domain:

1. **Add operators** to `morphogen/stdlib/acoustics.py`
2. **Add tests** to `tests/test_acoustics.py`
3. **Update documentation** in this file
4. **Create examples** in `examples/`

See `docs/guides/domain-implementation.md` for detailed guidelines.

---

**Status Summary:**

✅ Phase 1 Complete (1D Waveguides)
🔲 Phase 2 Planned (Enhanced 1D)
🔲 Phase 3 Planned (2D/3D)
🔲 Phase 4 Planned (Audio Integration)
