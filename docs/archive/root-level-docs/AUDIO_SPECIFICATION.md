# 🎶 Morphogen.Audio Specification v0.2

**A compositional and deterministic audio language built on the Morphogen kernel.**

---

## 0. Overview

Morphogen.Audio is a typed, stream-based audio computation dialect layered on the Morphogen kernel.
It provides deterministic semantics, physical and procedural synthesis primitives, and composable timing constructs.
It is the intermediate layer between:

- **RiffStack** — a live, stack-based performance shell
- **Morphogen Core** — the deterministic MLIR-based execution kernel

---

## 1. Language Philosophy

| Principle | Meaning |
|-----------|---------|
| **Fun-first defaults** | Simple expressions should sound good immediately. |
| **Deterministic semantics** | Same code, same sound — across machines and runs. |
| **Typed composition** | Every audio/control stream is a typed entity with known rate and units. |
| **Declarative structure** | Compositions describe what to play, not how to tick. |
| **Cross-domain extensibility** | Integrates with Field, Agent, and Visual dialects via shared scheduler. |

---

## 2. Core Types

All audio data types are defined in terms of the kernel's `Stream<T, Domain, Rate>` model.

| Alias | Underlying Type | Description |
|-------|----------------|-------------|
| `Sig` | `Stream<f32, 1D, audio>` | Audio-rate sample stream |
| `Ctl` | `Stream<f32, 0D, control>` | Control-rate continuous stream |
| `Evt<A>` | `Evt<A>` | Timestamped event stream carrying payload A |
| `Env` | `Stream<f32, 0D, control>` | Envelope or automation curve |
| `Stereo` | `(Sig, Sig)` | Two-channel output pair |
| `Note` | `{ pitch: Ctl[Hz], vel: Ctl, dur: Ctl[s] }` | Musical note descriptor |

**Units**: Hz, dB, s, ms, beats, ratio.

Lossy unit coercions are compile-time errors unless annotated `@allow_unit_cast`.

---

## 3. Structural Constructs

### 3.1 scene

Defines a self-contained audio composition block.

```morphogen
scene Ambient {
  let tone = sine(220Hz) |> reverb(0.15)
  out stereo = pan(0.1) tone
}
```

- Compiles to a Morphogen flow with `@rate=audio`.
- `out stereo` defines the primary output stream.
- Scenes may depend on shared modules.

### 3.2 module

Reusable synthesis or effect component.

```morphogen
module Pluck(freq: Ctl, vel: Ctl=1.0): Sig {
  let exc = noise(seed=2) |> lpf(8kHz) |> envexp(10ms)
  let sig = string(freq, 1.2s) exc * vel
  sig |> reverb(0.05)
}
```

- Pure, deterministic functions from inputs to outputs.
- Module calls are inlined at compile time.

### 3.3 import / export

Optional interoperability hooks.

```morphogen
import riffstack("patch.yaml")   # Convert a YAML patch to Morphogen.Audio IR
export yaml("scene.yaml")        # Serialize back out
```

---

## 4. Rate Model

The kernel supports multiple rate domains under one scheduler.

| Rate | Domain | Typical Ops | Nominal Frequency |
|------|--------|-------------|-------------------|
| **audio** | 1D continuous | oscillators, filters, FX | 44.1–96 kHz |
| **control** | 0D continuous | envelopes, modulations | 100–1,000 Hz |
| **visual** | 2D frame | rendering, metering | 30–120 Hz |

All cross-rate communication is explicit via resample operators:

- `upsample(x, factor)`
- `downsample(x, factor, mode="mean|hold|interp")`

---

## 5. Signal Operators

Each operator is a pure function from one or more streams to one or more streams.
Default parameters are musical and perceptually tuned.

### 5.1 Oscillators

```morphogen
sine(freq=440Hz, phase=0)
saw(freq=440Hz, blep=true)
square(freq=440Hz, pwm=0.5)
tri(freq=440Hz)
noise(type="white", seed=0)
impulse(rate=1Hz)
```

- Band-limited via BLEP or PolyBLEP unless disabled.
- **Determinism**: strict (Philox RNG for noise).

### 5.2 Filters

```morphogen
lpf(cutoff=2kHz, q=0.707)
hpf(cutoff=120Hz)
bpf(center=1kHz, q=1.0)
svf(mode="lp", cutoff, q)
eq3(bass=0dB, mid=0dB, treble=0dB)
```

- Linear time-invariant filters with stable coefficients.

### 5.3 Envelopes

```morphogen
adsr(a=5ms, d=80ms, s=0.7, r=200ms) (gate: Evt<void>)
ar(a=5ms, r=300ms) (gate)
envexp(t=50ms)
linseg(points=[(0,0),(0.1,1),(1,0.5)])
```

- Control-rate by default.
- `adsr` auto-scales by note velocity if connected.

### 5.4 Effects

```morphogen
delay(time=300ms, feedback=0.3, mix=0.25)
reverb(mix=0.12, size=0.8)
chorus(rate=0.3Hz, depth=8ms, mix=0.25)
flanger(rate=0.2Hz, depth=3ms, fb=0.25)
conv(ir="room.ir", mix=0.3)
drive(amount=0.5, shape="tanh")
limiter(threshold=-1dB, release=50ms)
```

- Effects are stable and deterministic within floating precision.
- Oversampling and quality governed by profile.

### 5.5 Utilities

```morphogen
mix(a,b,...)          # gain-compensated sum
pan(pos=-0.5..0.5)
mono(sig)
db2lin(x)
clip(limit=0.98)
normalize(sig)
```

---

## 6. Event and Score System

Events define timed discrete messages driving gates, notes, or parameter changes.

### 6.1 Event creation

```morphogen
let seq = score [
  at 0s note(440Hz, 1.0, 0.5s),
  at 0.5s note(660Hz, 0.8, 0.5s)
] |> loop(2s)
```

- `score` defines ordered `Evt<Note>` stream.
- `loop` repeats deterministically with sample-accurate edges.

### 6.2 Event mapping

```morphogen
let voice = (n: Note) => Pluck(n.pitch, n.vel)
let poly  = spawn(seq, voice, max_voices=12)
```

- `spawn` schedules multiple instances with deterministic voice allocation (quietest steal by default).

---

## 7. Physical Modeling Extensions

### 7.1 Waveguide Models

```morphogen
string(freq, t60=1.5s, damp=0.3) (exc: Sig)
membrane(size=0.4, tension=0.8) (exc: Sig)
```

### 7.2 Resonant Bodies

```morphogen
bodyIR(path="acoustic.ir", mix=0.9) (sig: Sig)
pickup(type="humbucker", pos=0.25) (sig: Sig)
```

### 7.3 Amplification

```morphogen
amp(model="brown", drive=0.6)
cab(ir="4x12.ir", mic="sm57")
```

All nonlinearities are anti-aliased (BLEP or oversampled) per profile.

---

## 8. Expressive Controls

```morphogen
let vib = sine(6Hz) * cents(7)
let bend = ctl.curve(2s, shape="easein")
let tone = string(note("A3") + vib + bend)
```

All parameters are modulatable by `Ctl` or `Sig`.

**Expressive mappings:**
- **Vibrato** → pitch mod
- **Dynamics** → amplitude scaling
- **Articulation** → subgraph selection

---

## 9. Output and Composition

### 9.1 Mixing

```morphogen
scene Mix {
  out stereo = mix(
    Bass.out * -6dB,
    Lead.out,
    Pad.out * -3dB
  ) |> limiter(-1dB)
}
```

### 9.2 Profiles

Profiles control precision, oversampling, and solver choice.

```morphogen
profile live:   precision=f32; oversample=1; block=64
profile render: precision=f64; oversample=2; block=256
profile strict: precision=f64; oversample=4; block=128
```

Precedence: per-op > module > profile > global.

---

## 10. Determinism Model

| Tier | Definition | Examples |
|------|------------|----------|
| **strict** | Bit-identical across devices | sine, lpf, delay |
| **repro** | Deterministic within FP precision | reverb, drive |
| **live** | Deterministic replayable | live loop, external control |

`@nondeterministic` annotations mark known exceptions (e.g., external inputs).

---

## 11. MLIR Lowering Map

| Op | Target Dialect | Notes |
|----|---------------|-------|
| `sine`, `saw`, etc. | `morphogen.signal` → `linalg.vector` | BLEP templates |
| `filter.*` | `linalg` | Small stencil kernels |
| `delay`, `reverb`, `conv` | `linalg` + `async` | Partitioned FFT conv |
| `spawn` | `scf.for` | Polyphonic scheduling |
| `score`, `loop` | `Evt` + fences | Deterministic event timing |
| `profile` | metadata only | Controls lowering params |

---

## 12. Examples

### 12.1 Simple Pluck

```morphogen
scene PluckDemo {
  let note = note("D3")
  let env  = adsr(5ms, 60ms, 0.6, 200ms)
  let exc  = noise(seed=1) |> lpf(6kHz) |> envexp(10ms)
  out stereo = string(note, 1.2s) exc |> reverb(0.1)
}
```

### 12.2 Polyphonic Sequence

```morphogen
scene Poly {
  let seq = score [
    at 0s note("A3",1,0.5s),
    at 0.5s note("C4",0.8,0.5s),
    at 1s note("E4",0.7,1s)
  ] |> loop(2s)
  let voice = (n: Note) => Pluck(n.pitch, n.vel)
  out stereo = spawn(seq, voice, max_voices=8)
}
```

### 12.3 Live Modulation

```morphogen
scene Mod {
  let base = note("A2")
  let vib  = sine(6Hz)*cents(8)
  let wob  = sine(0.25Hz)*1kHz + 1.5kHz
  out stereo = sine(base+vib) |> lpf(wob) |> drive(0.4) |> reverb(0.08)
}
```

---

## 13. Implementation Notes & Tradeoffs

| Topic | Decision | Rationale |
|-------|----------|-----------|
| **Rate inference** | Implicit (Sig, Ctl, Evt) with explicit override | Fun defaults, explicit power |
| **Unit system** | Strict; no silent casts | Prevents hidden scaling bugs |
| **Polyphony** | Deterministic allocation | Reproducible renders |
| **Oversampling** | Profile-based | Tunable quality/perf tradeoff |
| **Noise RNG** | Philox 4×32-10 | Cross-platform determinism |
| **Operator registry** | JSON metadata shared with RiffStack | Cross-DSL extensibility |
| **YAML I/O** | Optional import/export | Bridge with RiffStack patches |
| **Diagnostics** | NaN, DC offset, RMS lints | Safe defaults for musicians |
| **MLIR lowering** | Targets linalg, scf, async | Integrates into Morphogen toolchain |

---

## 14. Conformance Tests

| Test | Expected Result |
|------|----------------|
| Deterministic noise seed | identical render |
| Event timing | sample-accurate edge |
| Polyphonic voice allocation | reproducible voice IDs |
| Filter coefficient stability | no NaN/Inf |
| Profile switching | identical output within tolerance |
| RiffStack import/export | structural equivalence |

---

## 15. Future Work

- Granular synthesis and spectral processing ops
- Adaptive latency scheduler for live mode
- Integration with Luma (visuals) for audiovisual composition
- MIDI / OSC bridge for real controllers
- Interactive REPL shell (`morphogen.audio play`)

---

## One-line Summary

> **Morphogen.Audio is the compositional audio language of the Morphogen ecosystem — a deterministic, fun-first, extensible DSL where musical structure meets computational clarity.**

---

**Document Information**

- **Version**: 0.2
- **Date**: 2025-11-12
- **Status**: Formal Specification
- **Authors**: Scott Sen
- **Related**: [RiffStack](https://github.com/scottsen/riffstack), [Morphogen Core](../SPECIFICATION.md)

---

**End of Morphogen.Audio Specification v0.2**
