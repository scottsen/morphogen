# SPEC: Morphogen Type System & Units

**Version:** 1.0 Draft
**Status:** RFC
**Last Updated:** 2025-11-13

---

## Overview

The **Morphogen Type System** is the cornerstone of all semantic correctness in Morphogen. Every value in the system must have:

1. **Value type** — Scalar, vector, complex, etc.
2. **Domain** — time, frequency, space, k-space, etc.
3. **Rate** — audio, control, visual, sim
4. **Units** — Hz, dB, m, m/s, etc.
5. **Shape** — Dimensionality (0D, 1D, 2D, 3D)
6. **Determinism tier** — strict, repro, live

**Design Principle:** Errors in types, units, or domains must be caught at compile time. Silent mismatches are not allowed.

---

## Core Type Constructors

### 1. Stream<T, Domain, Rate>

**Definition:** A time-varying signal or data stream.

**Parameters:**
- `T` — Value type (f32, f64, Vec2<f32>, Complex<f32>, etc.)
- `Domain` — Domain annotation (time, frequency, space, k-space)
- `Rate` — Execution rate (audio, control, visual, sim)

**Examples:**
```morphogen
Stream<f32, time, audio>        // Audio signal (1D, time domain, 48kHz)
Stream<f64, time, control>      // Control signal (0D, time domain, 1kHz)
Stream<Complex<f32>, frequency, audio>  // Frequency spectrum
Stream<Vec2<f32>, space, sim>   // 2D velocity field
```

**Invariants:**
- Rate determines minimum update frequency
- Domain must match operation semantics
- Cross-domain operations require explicit transforms

---

### 2. Field<T, Domain>

**Definition:** A spatial field over a regular grid.

**Parameters:**
- `T` — Element type (f32, f64, Vec2<f32>, Vec3<f32>)
- `Domain` — Domain annotation (space, k-space)

**Examples:**
```morphogen
Field<f32, space>              // Scalar field (temperature, pressure)
Field<Vec2<f32>, space>        // 2D vector field (velocity)
Field<Vec3<f32>, space>        // 3D vector field
Field<Complex<f32>, k-space>   // Fourier-space field
```

**Invariants:**
- Fields have explicit grid metadata (spacing, centering)
- Boundary conditions must be specified
- Field operations must preserve grid compatibility

---

### 3. Evt<A>

**Definition:** A timestamped event stream.

**Parameters:**
- `A` — Event payload type

**Examples:**
```morphogen
Evt<Note>        // Musical note events (pitch, velocity, duration)
Evt<Control>     // Control change events (CC number, value)
Evt<Trigger>     // Bang/trigger events (no payload)
Evt<f32>         // Timestamped scalar values
```

**Invariants:**
- Events are sorted by time (monotonically increasing)
- Event times are sample-accurate
- Replay is deterministic

---

### 4. Grid<Dim, Spacing, Centering>

**Definition:** Grid metadata for fields.

**Parameters:**
- `Dim` — Dimensionality (1D, 2D, 3D)
- `Spacing` — Grid spacing (uniform or non-uniform)
- `Centering` — Node-centered or cell-centered

**Examples:**
```morphogen
Grid<2D, 0.1m, node>        // 2D uniform grid, 0.1m spacing, node-centered
Grid<3D, 0.05m, cell>       // 3D uniform grid, 0.05m spacing, cell-centered
Grid<2D, nonuniform, node>  // 2D non-uniform grid
```

**Invariants:**
- Spacing must have spatial units
- Stencil operations must respect centering
- Grid boundaries must be well-defined

---

## Scalar Types

### Primitive Types

| Type | Description | Size | Range |
|------|-------------|------|-------|
| `f16` | Half-precision float | 16 bits | ±65504, 3-4 decimal digits |
| `f32` | Single-precision float | 32 bits | ±3.4e38, 7-8 decimal digits |
| `f64` | Double-precision float | 64 bits | ±1.8e308, 15-16 decimal digits |
| `i32` | Signed 32-bit integer | 32 bits | -2^31 to 2^31-1 |
| `i64` | Signed 64-bit integer | 64 bits | -2^63 to 2^63-1 |
| `u32` | Unsigned 32-bit integer | 32 bits | 0 to 2^32-1 |
| `u64` | Unsigned 64-bit integer | 64 bits | 0 to 2^64-1 |
| `bool` | Boolean | 1 bit | true, false |

### Complex Types

```morphogen
Complex<T>   // Complex number (real, imaginary)
```

**Examples:**
```morphogen
Complex<f32>  // Single-precision complex
Complex<f64>  // Double-precision complex
```

**Operations:** `+`, `-`, `*`, `/`, `conj`, `abs`, `arg`, `exp`, `log`

---

## Vector Types

### Fixed-Size Vectors

```morphogen
Vec2<T>   // 2D vector
Vec3<T>   // 3D vector
Vec4<T>   // 4D vector
```

**Examples:**
```morphogen
Vec2<f32>   // 2D float vector
Vec3<f64>   // 3D double vector
```

**Operations:** `+`, `-`, `*` (scalar), `dot`, `cross` (Vec3 only), `norm`, `normalize`

---

## Domain Annotations

Domains describe the semantic space in which values exist. Transforms change domains.

| Domain | Description | Typical Units |
|--------|-------------|---------------|
| `time` | Time-domain signals | s, ms, samples |
| `frequency` | Frequency-domain signals | Hz, kHz, bins |
| `space` | Spatial fields | m, cm, px |
| `k-space` | Spatial frequency domain | 1/m, 1/cm |
| `cepstral` | Cepstral domain (DCT) | quefrency |
| `wavelet` | Wavelet domain | scale, time |

**Invariants:**
- Operations within a domain do not change the domain
- Cross-domain operations require explicit `transform.to()` or `transform.from()`

**Examples:**
```morphogen
// ERROR: Cannot add time-domain and frequency-domain signals
let x: Stream<f32, time, audio> = sine(440Hz)
let y: Stream<Complex<f32>, frequency, audio> = fft(sine(880Hz))
let z = x + y  // ERROR: Domain mismatch (time vs frequency)

// CORRECT: Transform to same domain first
let y_time = ifft(y)
let z = x + y_time  // OK
```

---

## Rate System

Rates define execution frequencies and scheduling groups.

| Rate | Description | Typical Frequency | Block Size |
|------|-------------|-------------------|------------|
| `audio` | Audio-rate signals | 44.1kHz, 48kHz, 96kHz | 64-512 samples |
| `control` | Control-rate signals | 100Hz, 1kHz | 1-10 samples |
| `visual` | Visual-rate updates | 30Hz, 60Hz, 120Hz | 1 frame |
| `sim` | Simulation timestep | Variable | dt-dependent |

**Invariants:**
- Cross-rate connections require explicit resampling
- Rates form a partial order: audio ≥ control ≥ visual ≥ sim
- Events must be scheduled at rate boundaries

**Examples:**
```morphogen
// ERROR: Direct connection across rates
let audio_sig: Stream<f32, time, audio> = sine(440Hz)
let control_sig: Stream<f32, time, control> = audio_sig  // ERROR: Rate mismatch

// CORRECT: Explicit resampling
let control_sig = resample(audio_sig, to_rate=control, mode=linear)  // OK
```

---

## Units System

All numeric values with physical meaning must have units. Unit mismatches are compile errors.

### SI Base Units

| Unit | Dimension | Symbol |
|------|-----------|--------|
| meter | length | `m` |
| kilogram | mass | `kg` |
| second | time | `s` |
| ampere | current | `A` |
| kelvin | temperature | `K` |
| mole | amount | `mol` |
| candela | luminous intensity | `cd` |

### Derived Units

| Unit | Dimension | Formula | Symbol |
|------|-----------|---------|--------|
| hertz | frequency | 1/s | `Hz` |
| newton | force | kg⋅m/s² | `N` |
| pascal | pressure | N/m² | `Pa` |
| joule | energy | N⋅m | `J` |
| watt | power | J/s | `W` |
| volt | voltage | W/A | `V` |

### Audio-Specific Units

| Unit | Description | Conversion |
|------|-------------|------------|
| `Hz` | Frequency | Base unit |
| `kHz` | Kilohertz | 1000 Hz |
| `MHz` | Megahertz | 1e6 Hz |
| `dB` | Decibels | 20⋅log10(amp) |
| `cents` | Musical cents | 1200⋅log2(freq_ratio) |
| `midi` | MIDI note number | 69 @ 440Hz |
| `rad` | Radians (phase) | Base unit |
| `deg` | Degrees | π/180 rad |
| `turns` | Full rotations | 2π rad |

### Visual-Specific Units

| Unit | Description |
|------|-------------|
| `px` | Pixels |
| `sr` | Steradians (solid angle) |

### Physics-Specific Units

| Unit | Description | Formula |
|------|-------------|---------|
| `m` | Meters | Base |
| `m/s` | Velocity | m/s |
| `m/s²` | Acceleration | m/s² |
| `N` | Newtons (force) | kg⋅m/s² |
| `kg` | Kilograms (mass) | Base |

---

## Unit Arithmetic

### Rules

1. **Addition/Subtraction:** Units must match exactly
   ```morphogen
   let x: f32<Hz> = 440Hz
   let y: f32<Hz> = 880Hz
   let z = x + y  // OK: 1320Hz

   let a: f32<Hz> = 440Hz
   let b: f32<s> = 1s
   let c = a + b  // ERROR: Unit mismatch (Hz vs s)
   ```

2. **Multiplication:** Units multiply
   ```morphogen
   let t: f32<s> = 2s
   let v: f32<m/s> = 10m/s
   let d = t * v  // OK: 20<m>
   ```

3. **Division:** Units divide
   ```morphogen
   let d: f32<m> = 100m
   let t: f32<s> = 10s
   let v = d / t  // OK: 10<m/s>
   ```

4. **Exponentiation:** Units raise to power
   ```morphogen
   let r: f32<m> = 5m
   let area = r * r  // OK: 25<m²>
   ```

5. **Transcendental Functions:** Require unitless arguments
   ```morphogen
   let x: f32<rad> = 1.57rad
   let y = sin(x)  // OK: sin expects <rad>, returns unitless

   let f: f32<Hz> = 440Hz
   let z = sin(f)  // ERROR: sin expects unitless or <rad>
   ```

---

## Unit Conversions

### Implicit Conversions (Allowed)

Safe conversions that don't lose information:

```morphogen
Hz → kHz → MHz (frequency scaling)
s → ms → us (time scaling)
m → cm → mm (length scaling)
```

### Explicit Conversions (Required)

Conversions that change representation:

```morphogen
linear → dB:
  let amp: f32<linear> = 0.5
  let db = to_dB(amp)  // -6.02dB

Hz → midi:
  let freq: f32<Hz> = 440Hz
  let note = to_midi(freq)  // 69

rad → deg:
  let angle: f32<rad> = 3.14rad
  let degrees = to_deg(angle)  // 180deg
```

### Forbidden Conversions (Compile Error)

```morphogen
let f: f32<Hz> = 440Hz
let t: f32<s> = f  // ERROR: Cannot convert Hz to s
```

---

## Determinism Tiers

Every operation is classified by its determinism guarantee.

| Tier | Description | Examples |
|------|-------------|----------|
| **strict** | Bit-exact across devices/runs | FFT (reference), RNG (Philox), field.diffuse |
| **repro** | Deterministic within FP precision | Iterative solvers, vendor FFTs |
| **live** | Replayable but not bit-exact | Adaptive algorithms, live input |

**Invariants:**
- Operators declare their tier in the registry
- Graphs inherit the weakest tier of their operators
- Users can enforce minimum tier with profile

**Examples:**
```morphogen
// Strict profile: All ops must be tier=strict
profile strict {
  let x = sine(440Hz)  // OK: tier=strict
  let y = fft(x, method=reference)  // OK: tier=strict
  let z = fft(x, method=fftw)  // ERROR: tier=repro, profile requires strict
}
```

---

## Type Inference Rules

### Rule 1: Literal Inference

```morphogen
let x = 440  // Inferred: i32 (unitless)
let y = 440Hz  // Inferred: f32<Hz>
let z = 1.5s  // Inferred: f32<s>
```

### Rule 2: Operator Output Types

```morphogen
let osc = sine(440Hz)
// Inferred: Stream<f32, time, audio>

let spec = fft(osc)
// Inferred: Stream<Complex<f32>, frequency, audio>
```

### Rule 3: Binary Operations

```morphogen
let a: Stream<f32, time, audio> = sine(440Hz)
let b: Stream<f32, time, audio> = sine(880Hz)
let c = a + b
// Inferred: Stream<f32, time, audio>
```

### Rule 4: Cross-Rate Inference

```morphogen
let audio_sig: Stream<f32, time, audio> = sine(440Hz)
let control_param: Stream<f32, time, control> = lfo(1Hz)
let modulated = audio_sig * control_param
// Inferred rate: audio (higher of the two)
// Automatic upsampling of control_param
```

---

## Validation Rules

### 1. Type Compatibility

```python
def types_compatible(from_type, to_type):
    """Check if types are compatible for connection."""
    if from_type.value_type != to_type.value_type:
        return False  # Value type must match
    if from_type.domain != to_type.domain:
        return False  # Domain must match
    if from_type.rate != to_type.rate:
        return False  # Rate must match (or explicit resample)
    return True
```

### 2. Unit Compatibility

```python
def units_compatible(unit1, unit2):
    """Check if units are compatible for addition."""
    return unit1.dimension == unit2.dimension and \
           unit1.scale_factor == unit2.scale_factor
```

### 3. Domain Transform Validation

```python
def validate_transform(op, input_domain, output_domain):
    """Validate domain transform is legal."""
    legal_transforms = {
        ("time", "frequency"): ["fft", "stft"],
        ("frequency", "time"): ["ifft", "istft"],
        ("space", "k-space"): ["fft2d", "fft3d"],
        ("k-space", "space"): ["ifft2d", "ifft3d"],
    }

    key = (input_domain, output_domain)
    if key not in legal_transforms:
        raise ValueError(f"No transform from {input_domain} to {output_domain}")

    if op not in legal_transforms[key]:
        raise ValueError(f"{op} cannot transform {input_domain} → {output_domain}")
```

---

## Examples

### Example 1: Audio Signal Processing

```morphogen
scene SimpleSynth {
  // Types inferred from operators
  let osc = sine(440Hz)
  // osc: Stream<f32, time, audio>

  let env = adsr(attack=0.01s, decay=0.1s, sustain=0.7, release=0.3s)
  // env: Stream<f32, time, control>

  let modulated = osc * env
  // modulated: Stream<f32, time, audio> (env auto-upsampled)

  out mono = modulated
}
```

### Example 2: Frequency-Domain Processing

```morphogen
scene SpectralFilter {
  let sig = sine(440Hz) + sine(880Hz)
  // sig: Stream<f32, time, audio>

  let spec = transform.to(sig, domain=frequency, method=fft)
  // spec: Stream<Complex<f32>, frequency, audio>

  let filtered = spec * lowpass_mask(cutoff=600Hz)
  // filtered: Stream<Complex<f32>, frequency, audio>

  let back = transform.from(filtered, domain=frequency, method=ifft)
  // back: Stream<f32, time, audio>

  out mono = back
}
```

### Example 3: Spatial Field

```morphogen
scene FluidSim {
  let velocity: Field<Vec2<f32>, space> = field.init(128, 128, 0.1m)
  let density: Field<f32, space> = field.init(128, 128, 0.1m)

  // Grid metadata automatically attached
  // velocity.grid = Grid<2D, 0.1m, node>

  velocity = field.advect(velocity, velocity, dt=0.01s, method=bfecc)
  density = field.advect(density, velocity, dt=0.01s, method=bfecc)

  out visual = colorize(density, palette=viridis)
}
```

---

## Summary

The Morphogen Type System provides:

✅ **Strong static typing** — Catch errors at compile time
✅ **Physical unit tracking** — Prevent unit mismatches
✅ **Domain annotations** — Explicit semantic spaces
✅ **Rate system** — Multirate scheduling support
✅ **Determinism tiers** — Explicit guarantees
✅ **Type inference** — Concise syntax without boilerplate

This is the foundation that makes Morphogen safe, composable, and correct.

---

## References

- `graph-ir.md` — Graph IR uses these types
- `transform.md` — Transform operations change domains
- `scheduler.md` — Scheduler uses rate annotations
- `profiles.md` — Profiles affect type behavior (precision, determinism)
