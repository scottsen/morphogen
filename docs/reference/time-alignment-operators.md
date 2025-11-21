# Morphogen Time Alignment Operators

**Version:** 1.0
**Date:** 2025-11-15
**Status:** Design Document
**Domain:** AudioDomain
**Related:** ../specifications/operator-registry.md, AUDIO_SPECIFICATION.md, OPERATOR_REGISTRY_EXPANSION.md

---

## Overview

**Time alignment** is one of the cleanest, most impactful DSP tasks to express in a Morphogen pipeline. This document specifies operators for per-speaker time alignment, a critical workflow in professional audio (car audio, studio monitors, live sound) that follows Morphogen's natural **measurement → analysis → operator output** pattern.

**Why Time Alignment Matters:**

Every speaker in a multi-way system has:
- Different physical distance from the listener
- Different crossover filters with different phase characteristics
- Different inherent latency (DSP, amplifiers, crossovers)
- Different mechanical response and group delay

**Time Alignment Solves:**

1. **Arrival time difference** — Distance compensation (speed of sound ≈ 0.343 ms per 10 cm)
2. **Phase alignment** — Crossover frequency coherence (subwoofer ↔ midbass ↔ midrange ↔ tweeter)
3. **Group delay differences** — Frequency-dependent delay correction
4. **Stereo imaging** — Proper spatial localization and coherent wavefront arrival
5. **Bass integration** — Tight, phase-coherent low-frequency response

**Results:**
- ✅ Coherent wavefront arrival
- ✅ Correct stereo imaging
- ✅ Tighter transients
- ✅ Better bass integration
- ✅ Proper spatial localization

---

## Why Morphogen Excels at Time Alignment

Time alignment is an **ideal Morphogen workflow** because:

1. ✅ **Same operators, multiple domains**
   - Auto-EQ uses FFT, IR extraction, smoothing
   - Time alignment uses the same FFT/IR operators + new analysis
   - Guitar modal modeling uses the same IR analysis
   - Room correction uses the same measurement pipeline

2. ✅ **MLIR/GPU-friendly**
   - FFT, deconvolution, cross-correlation → trivially vectorizable
   - Perfect for GPU acceleration

3. ✅ **Extends naturally into physics**
   - Crossover phase matching = modal excitation matching (same math)
   - Group delay analysis applies to mechanical systems

4. ✅ **Highly reusable operators**
   - Time alignment operators apply to ALL audio DSP
   - Same cross-correlation used for: audio sync, beamforming, echo detection

5. ✅ **Fits domain separation**
   - `AudioMeasurementDomain` — Generate test signals, record responses
   - `AudioAnalysisDomain` — FFT, IR extraction, peak detection, group delay
   - `FilterDesignDomain` — Delay designer, crossover matching
   - `AlignmentDesignDomain` (optional) — High-level calibration workflows

---

## Operator Specifications

### Layer 2: Transform Operators (Extended)

These operators extend the existing Transform layer (Layer 2) from `../specifications/operator-registry.md`.

| Operator | Category | Description | Already Exists |
|----------|----------|-------------|----------------|
| `fft` | transform | Time → frequency domain | ✅ YES |
| `ifft` | transform | Frequency → time domain | ✅ YES |
| `stft` | transform | Time → time-frequency | ✅ YES |

---

### Layer 5: Audio/DSP Operators (Extended)

#### 5a. Measurement Operators (NEW SUBCATEGORY)

| Operator | Signature | Description | Determinism |
|----------|-----------|-------------|-------------|
| `sine_sweep` | `(start_freq: Hz, end_freq: Hz, duration: s, method: linear\|log) → AudioSignal` | Generate exponential or linear sine sweep | DETERMINISTIC |
| `impulse_train` | `(interval: s, duration: s) → AudioSignal` | Generate periodic impulse train for time alignment | DETERMINISTIC |
| `white_noise_burst` | `(duration: s, seed: int) → AudioSignal` | White noise burst for MLS analysis | DETERMINISTIC |
| `mls_sequence` | `(order: int, seed: int) → AudioSignal` | Maximum Length Sequence for impulse response | DETERMINISTIC |

**Example:**
```json
{
  "name": "sine_sweep",
  "category": "measurement",
  "layer": 5,
  "description": "Generate exponential or linear sine sweep for impulse response measurement",
  "inputs": [],
  "outputs": [
    {"name": "sweep", "type": "Stream<f32,time,audio>", "description": "Sweep signal"}
  ],
  "params": [
    {"name": "start_freq", "type": "f32<Hz>", "default": "20Hz", "description": "Start frequency"},
    {"name": "end_freq", "type": "f32<Hz>", "default": "20000Hz", "description": "End frequency"},
    {"name": "duration", "type": "f32<s>", "default": "10s", "description": "Sweep duration"},
    {"name": "method", "type": "string", "default": "log", "enum": ["linear", "log"], "description": "Sweep method"}
  ],
  "determinism": "strict",
  "rate": "audio",
  "implementation": {
    "python": "morphogen.stdlib.measurement.sine_sweep",
    "mlir": "morphogen.audio.measurement.sine_sweep"
  }
}
```

---

#### 5b. Analysis Operators (NEW SUBCATEGORY)

| Operator | Signature | Description | Determinism |
|----------|-----------|-------------|-------------|
| `impulse_response_extractor` | `(sweep: AudioSignal, recording: AudioSignal) → ImpulseResponse` | Extract IR via deconvolution (Farina method) | DETERMINISTIC |
| `ir_peak_detect` | `(ir: ImpulseResponse, method: max\|threshold) → DelayTime` | Find arrival time (peak detection) | DETERMINISTIC |
| `cross_correlation` | `(signal_a: AudioSignal, signal_b: AudioSignal) → CrossCorrResult` | Cross-correlation for time offset detection | DETERMINISTIC |
| `group_delay` | `(fft_mag: Spectrum, fft_phase: Spectrum) → GroupDelaySpectrum` | Compute frequency-dependent group delay: gd(f) = -dφ/dω | DETERMINISTIC |
| `phase_difference` | `(spectrum_a: Spectrum, spectrum_b: Spectrum) → PhaseSpectrum` | Compute phase difference between two signals | DETERMINISTIC |
| `windowed_ir` | `(ir: ImpulseResponse, window_start: ms, window_length: ms) → ImpulseResponse` | Extract windowed portion of IR (isolate early reflections) | DETERMINISTIC |

**Example:**
```json
{
  "name": "impulse_response_extractor",
  "category": "analysis",
  "layer": 5,
  "description": "Extract impulse response from sweep and recording using Farina deconvolution",
  "inputs": [
    {"name": "sweep", "type": "Stream<f32,time,audio>", "description": "Original sweep signal"},
    {"name": "recording", "type": "Stream<f32,time,audio>", "description": "Recorded response"}
  ],
  "outputs": [
    {"name": "ir", "type": "ImpulseResponse", "description": "Extracted impulse response"},
    {"name": "metadata", "type": "IRMetadata", "description": "Peak sample, SNR, etc."}
  ],
  "params": [
    {"name": "normalize", "type": "bool", "default": true, "description": "Normalize output IR"}
  ],
  "determinism": "strict",
  "rate": "audio",
  "transform_metadata": {
    "input_domain": "time",
    "output_domain": "time",
    "transform_type": "deconvolution"
  },
  "lowering_hints": {
    "prefer_fft": true,
    "vectorize": true
  },
  "implementation": {
    "python": "morphogen.stdlib.analysis.impulse_response_extractor",
    "mlir": "morphogen.audio.analysis.ir_extract"
  }
}
```

**Example:**
```json
{
  "name": "group_delay",
  "category": "analysis",
  "layer": 5,
  "description": "Compute frequency-dependent group delay from FFT magnitude and phase",
  "inputs": [
    {"name": "fft_mag", "type": "Spectrum", "description": "FFT magnitude"},
    {"name": "fft_phase", "type": "Spectrum", "description": "FFT phase (unwrapped)"}
  ],
  "outputs": [
    {"name": "gd_curve", "type": "GroupDelaySpectrum", "description": "Group delay vs frequency"}
  ],
  "params": [],
  "determinism": "strict",
  "rate": "control",
  "numeric_properties": {
    "requires_unwrapped_phase": true
  },
  "implementation": {
    "python": "morphogen.stdlib.analysis.group_delay",
    "mlir": "morphogen.audio.analysis.group_delay"
  }
}
```

---

#### 5c. Alignment Operators (NEW SUBCATEGORY)

| Operator | Signature | Description | Determinism |
|----------|-----------|-------------|-------------|
| `delay_designer` | `(arrival_times: List[DelayTime], reference: string) → DelayMap` | Compute per-channel delays from arrival times | DETERMINISTIC |
| `crossover_phase_aligner` | `(woofer_ir: IR, mid_ir: IR, xo_freq: Hz) → PhaseCorrection` | Compute phase correction at crossover frequency | DETERMINISTIC |
| `allpass_delay` | `(target_delay: ms, sample_rate: Hz) → AllpassCoeffs` | Design allpass filter for fractional-sample delay | DETERMINISTIC |
| `delay_compensation` | `(signal: AudioSignal, delay: ms) → AudioSignal` | Apply delay compensation to signal | DETERMINISTIC |

**Example:**
```json
{
  "name": "delay_designer",
  "category": "alignment",
  "layer": 5,
  "description": "Compute per-channel delay offsets from measured arrival times",
  "inputs": [
    {"name": "arrival_times", "type": "List[DelayTime]", "description": "Measured arrival times per channel"}
  ],
  "outputs": [
    {"name": "delay_map", "type": "DelayMap", "description": "Per-channel delay settings"}
  ],
  "params": [
    {"name": "reference", "type": "string", "default": "earliest", "enum": ["earliest", "latest", "named"], "description": "Reference point for alignment"}
  ],
  "determinism": "strict",
  "rate": "control",
  "implementation": {
    "python": "morphogen.stdlib.alignment.delay_designer",
    "mlir": "morphogen.audio.alignment.delay_designer"
  }
}
```

---

#### 5d. Export Operators (NEW SUBCATEGORY)

| Operator | Signature | Description | Determinism |
|----------|-----------|-------------|-------------|
| `export_delays` | `(delay_map: DelayMap, format: minidsp\|json\|csv) → File` | Export delay settings to hardware DSP format | DETERMINISTIC |
| `export_ir` | `(ir: ImpulseResponse, format: wav\|flac) → File` | Export impulse response as audio file | DETERMINISTIC |
| `export_report` | `(alignment_result: AlignmentResult, format: pdf\|html) → File` | Generate alignment report with plots | DETERMINISTIC |

**Example:**
```json
{
  "name": "export_delays",
  "category": "export",
  "layer": 5,
  "description": "Export delay settings in hardware DSP format (miniDSP, JSON, CSV)",
  "inputs": [
    {"name": "delay_map", "type": "DelayMap", "description": "Per-channel delay settings"}
  ],
  "outputs": [
    {"name": "file", "type": "File", "description": "Exported configuration file"}
  ],
  "params": [
    {"name": "format", "type": "string", "default": "json", "enum": ["minidsp", "json", "csv"], "description": "Output format"},
    {"name": "path", "type": "string", "description": "Output file path"}
  ],
  "determinism": "strict",
  "rate": "control",
  "implementation": {
    "python": "morphogen.stdlib.export.export_delays"
  }
}
```

---

### New Data Types

These types are introduced to support time alignment workflows:

| Type | Description | Fields |
|------|-------------|--------|
| `ImpulseResponse` | Time-domain impulse response | `samples: Array[f32]`, `sample_rate: Hz`, `peak_sample: int`, `peak_time: ms` |
| `DelayTime` | Measured delay time | `time_ms: f32`, `confidence: f32`, `source: string` |
| `DelayMap` | Per-channel delay settings | `channels: Map[string, DelayTime]`, `reference: string` |
| `CrossCorrResult` | Cross-correlation result | `offset_samples: int`, `offset_ms: f32`, `correlation: f32` |
| `GroupDelaySpectrum` | Frequency-dependent group delay | `frequencies: Array[f32]`, `delays_ms: Array[f32]` |
| `PhaseSpectrum` | Phase vs frequency | `frequencies: Array[f32]`, `phase_rad: Array[f32]` |
| `PhaseCorrection` | Phase correction at crossover | `delay_offset: ms`, `allpass_coeffs: AllpassCoeffs` |
| `AlignmentResult` | Complete alignment result | `delay_map: DelayMap`, `group_delay: GroupDelaySpectrum`, `phase_alignment: PhaseSpectrum` |

---

## Complete Morphogen Workflow: Car Audio Time Alignment

This is a **real-world Morphogen pipeline** for time-aligning a 3-way car audio system (front left, front right, subwoofer).

```morphogen
# ============================================================
# Time Alignment Calibration Pipeline
# ============================================================

# 1. MEASUREMENT PHASE
# Generate test signal
measurement:
  - id: sweep
    operator: sine_sweep
    params:
      start_freq: 20Hz
      end_freq: 20000Hz
      duration: 10s
      method: log

# 2. RECORDING PHASE
# Record each speaker separately with reference mic at listening position
recording:
  - id: front_left_rec
    mic: ref_mic
    channel: front_left_output
    description: "Front left tweeter + midbass + woofer"

  - id: front_right_rec
    mic: ref_mic
    channel: front_right_output
    description: "Front right tweeter + midbass + woofer"

  - id: subwoofer_rec
    mic: ref_mic
    channel: subwoofer_output
    description: "Subwoofer (trunk mounted)"

# 3. ANALYSIS PHASE
# Extract impulse responses
analysis:
  - id: ir_left
    operator: impulse_response_extractor
    inputs: [sweep, front_left_rec]
    params:
      normalize: true

  - id: ir_right
    operator: impulse_response_extractor
    inputs: [sweep, front_right_rec]
    params:
      normalize: true

  - id: ir_sub
    operator: impulse_response_extractor
    inputs: [sweep, subwoofer_rec]
    params:
      normalize: true

  # Detect arrival times (peak detection)
  - id: delay_left
    operator: ir_peak_detect
    inputs: [ir_left]
    params:
      method: max

  - id: delay_right
    operator: ir_peak_detect
    inputs: [ir_right]
    params:
      method: max

  - id: delay_sub
    operator: ir_peak_detect
    inputs: [ir_sub]
    params:
      method: max

  # Optional: Cross-correlation for phase alignment
  - id: crosscorr_lr
    operator: cross_correlation
    inputs: [front_left_rec, front_right_rec]
    description: "Left-right phase alignment"

  # Optional: Group delay analysis for subwoofer
  - id: gd_sub
    operator: group_delay
    inputs: [ir_sub.fft_mag, ir_sub.fft_phase]
    description: "Subwoofer group delay (phase vs frequency)"

# 4. ALIGNMENT DESIGN PHASE
# Compute optimal delays
alignment:
  - id: delay_settings
    operator: delay_designer
    inputs: [delay_left, delay_right, delay_sub]
    params:
      reference: earliest  # Align to earliest arrival (usually tweeter)

  # Optional: Crossover phase matching (sub + midbass)
  - id: phase_correction_sub
    operator: crossover_phase_aligner
    inputs: [ir_sub, ir_left]
    params:
      xo_freq: 80Hz  # Subwoofer crossover frequency

# 5. EXPORT PHASE
# Export to miniDSP or JSON
export:
  - id: export_minidsp
    operator: export_delays
    inputs: [delay_settings]
    params:
      format: minidsp
      path: "car_alignment_minidsp.xml"

  - id: export_json
    operator: export_delays
    inputs: [delay_settings]
    params:
      format: json
      path: "car_alignment.json"

  - id: export_report
    operator: export_report
    inputs: [delay_settings, gd_sub, crosscorr_lr]
    params:
      format: html
      path: "alignment_report.html"

# 6. VALIDATION PHASE (Optional)
# Verify alignment by measuring again
validation:
  - id: verify_sweep
    operator: sine_sweep
    params:
      start_freq: 20Hz
      end_freq: 20000Hz
      duration: 5s

  - id: verify_recording
    mic: ref_mic
    channel: all_speakers_with_alignment

  - id: verify_ir
    operator: impulse_response_extractor
    inputs: [verify_sweep, verify_recording]

  - id: verify_phase
    operator: phase_difference
    inputs: [verify_ir.fft_phase, ir_left.fft_phase]
    description: "Phase alignment quality check"
```

**Press "Run" → Morphogen outputs:**

```
============================================================
Time Alignment Results
============================================================
Reference point: Driver headrest (earliest arrival)

Recommended delays:
------------------------------------------------------------
Front Left    :  1.37 ms  (tweeter arrival)
Front Right   :  0.00 ms  (REFERENCE - earliest)
Subwoofer     :  7.85 ms  (includes phase alignment at 80Hz)
------------------------------------------------------------

Cross-correlation (L-R): 0.98  (excellent stereo coherence)
Group delay @ 80Hz:      4.2 ms (corrected)

Exported:
  - car_alignment_minidsp.xml
  - car_alignment.json
  - alignment_report.html
============================================================
```

---

## Example Output: What Morphogen Would Generate

### 1. Delay Map (JSON Export)

```json
{
  "version": "1.0",
  "reference": "front_right",
  "reference_point": "driver_headrest",
  "sample_rate": 48000,
  "channels": {
    "front_left": {
      "delay_ms": 1.37,
      "delay_samples": 66,
      "confidence": 0.95,
      "arrival_time_ms": 2.41
    },
    "front_right": {
      "delay_ms": 0.00,
      "delay_samples": 0,
      "confidence": 0.98,
      "arrival_time_ms": 1.04
    },
    "subwoofer": {
      "delay_ms": 7.85,
      "delay_samples": 377,
      "confidence": 0.89,
      "arrival_time_ms": 8.89,
      "phase_correction": {
        "crossover_freq_hz": 80,
        "additional_delay_ms": 4.2,
        "allpass_coeffs": [0.95, -0.31, 0.87]
      }
    }
  }
}
```

### 2. Group Delay Plot (HTML Report)

**Frequency (Hz) → Group Delay (ms)**

```
20Hz   : 12.5 ms  ████████████
40Hz   :  8.3 ms  ████████
80Hz   :  4.2 ms  ████  ← Crossover (corrected)
160Hz  :  1.8 ms  ██
500Hz  :  1.2 ms  █
1kHz   :  1.1 ms  █
5kHz   :  1.0 ms  █
10kHz  :  1.0 ms  █
20kHz  :  1.0 ms  █
```

### 3. Phase Alignment Quality

**Cross-correlation Results:**

| Pair | Correlation | Delay Offset | Quality |
|------|-------------|--------------|---------|
| L-R | 0.98 | -1.37 ms | ✅ Excellent |
| L-Sub | 0.87 | +7.85 ms | ✅ Good |
| R-Sub | 0.91 | +7.85 ms | ✅ Good |

---

## Integration with Existing Morphogen Operators

Time alignment **reuses** many operators already in Morphogen:

### Already Exists (From ../specifications/operator-registry.md)

| Operator | Layer | Use in Time Alignment |
|----------|-------|----------------------|
| `fft` | 2 (Transform) | Deconvolution, group delay, phase analysis |
| `ifft` | 2 (Transform) | Reconstruct time-domain IR after processing |
| `lpf`, `hpf` | 5 (Audio) | Bandlimit analysis for specific crossover regions |
| `delay` | 5 (Audio) | Apply computed delays to signals |

### New Operators (This Document)

| Operator | Layer | Category | Purpose |
|----------|-------|----------|---------|
| `sine_sweep` | 5 | measurement | Test signal generation |
| `impulse_response_extractor` | 5 | analysis | IR extraction (Farina deconvolution) |
| `ir_peak_detect` | 5 | analysis | Arrival time detection |
| `cross_correlation` | 5 | analysis | Phase alignment detection |
| `group_delay` | 5 | analysis | Frequency-dependent delay |
| `delay_designer` | 5 | alignment | Compute optimal delays |
| `crossover_phase_aligner` | 5 | alignment | Crossover phase matching |
| `export_delays` | 5 | export | Hardware DSP export |

---

## Operator Registry Integration

These operators fit into the existing **7-layer operator architecture** from `../specifications/operator-registry.md`:

| Layer | Description | Time Alignment Operators |
|-------|-------------|-------------------------|
| **1. Core** | Foundational ops | (uses existing `cast`, `rate.change`) |
| **2. Transforms** | FFT, domain changes | ✅ **Already has:** `fft`, `ifft` |
| **3. Stochastic** | RNG, processes | (not used) |
| **4. Physics/Fields** | Integrators, PDEs | (not used) |
| **5. Audio/DSP** | Oscillators, filters, FX | ✅ **NEW:** `measurement`, `analysis`, `alignment`, `export` subcategories |
| **6. Fractals/Visuals** | Iteration, rendering | (not used) |
| **7. Finance** | Models, pricing | (not used) |

**New Subcategories in Layer 5 (Audio):**

```json
{
  "layer": 5,
  "domain": "audio",
  "subcategories": {
    "oscillator": ["sine", "saw", "square", "triangle", "noise"],
    "filter": ["lpf", "hpf", "bpf", "svf", "peq"],
    "envelope": ["adsr", "ar", "envexp"],
    "effect": ["delay", "reverb", "chorus", "compressor", "limiter"],
    "spectral": ["spectral.sharpen", "spectral.morph"],
    "measurement": ["sine_sweep", "impulse_train", "mls_sequence"],       // NEW
    "analysis": ["impulse_response_extractor", "ir_peak_detect",           // NEW
                 "cross_correlation", "group_delay", "phase_difference"],
    "alignment": ["delay_designer", "crossover_phase_aligner"],           // NEW
    "export": ["export_delays", "export_ir", "export_report"]            // NEW
  }
}
```

---

## Reference Types for Time Alignment

Following the unified reference architecture from `ADR-002`, time alignment introduces:

### Primary: `ImpulseResponseRef`

**Purpose:** Reference to an extracted impulse response.

**Auto-Anchors:**
```python
ImpulseResponseRef.peak → SampleRef               # Peak sample location
ImpulseResponseRef.peak_time → f32<ms>            # Peak arrival time
ImpulseResponseRef.duration → f32<ms>             # IR duration
ImpulseResponseRef.early_window → ImpulseResponseRef   # First 50ms (direct + early reflections)
ImpulseResponseRef.late_window → ImpulseResponseRef    # Late reflections (after 50ms)
ImpulseResponseRef.fft_mag → Spectrum             # FFT magnitude
ImpulseResponseRef.fft_phase → Spectrum           # FFT phase
ImpulseResponseRef.snr → f32<dB>                  # Signal-to-noise ratio
```

**Example:**
```python
ir = impulse_response_extractor(sweep, recording)

# Access auto-generated anchors
print(f"Peak arrival: {ir.peak_time} ms")
print(f"SNR: {ir.snr} dB")

# Extract early reflections
early = ir.early_window  # First 50ms

# Compute group delay
gd = group_delay(ir.fft_mag, ir.fft_phase)
```

### Secondary: `DelayMapRef`

**Purpose:** Reference to a complete delay configuration.

**Auto-Anchors:**
```python
DelayMapRef.channel[name: str] → DelayTime         # Delay for named channel
DelayMapRef.reference → str                        # Reference channel name
DelayMapRef.max_delay → f32<ms>                    # Maximum delay value
DelayMapRef.min_delay → f32<ms>                    # Minimum delay value
```

**Example:**
```python
delays = delay_designer([delay_left, delay_right, delay_sub], reference="earliest")

# Access per-channel delays
left_delay = delays.channel["front_left"]  # → 1.37 ms
sub_delay = delays.channel["subwoofer"]    # → 7.85 ms

# Export
export_delays(delays, format="minidsp", path="alignment.xml")
```

---

## Domain Architecture Integration

Time alignment fits into Morphogen's domain architecture (from `../architecture/domain-architecture.md`):

```
AudioMeasurementDomain
│
├── Operators: sine_sweep, impulse_train, mls_sequence
├── Output: Test signals
│
AudioAnalysisDomain
│
├── Operators: impulse_response_extractor, ir_peak_detect,
│              cross_correlation, group_delay
├── Input: Test signals + recordings
├── Output: ImpulseResponseRef, DelayTime, GroupDelaySpectrum
│
AlignmentDesignDomain (NEW)
│
├── Operators: delay_designer, crossover_phase_aligner
├── Input: DelayTime[], ImpulseResponseRef[]
├── Output: DelayMapRef, PhaseCorrection
│
ExportDomain
│
├── Operators: export_delays, export_ir, export_report
├── Input: DelayMapRef, AlignmentResult
├── Output: Files (JSON, XML, HTML, WAV)
```

---

## Passes for Time Alignment

Following the pass architecture from `../specifications/operator-registry.md`:

### Validation Passes

| Pass | Description | Error Conditions |
|------|-------------|------------------|
| `SampleRateConsistency` | Ensure all IRs have same sample rate | Mismatched sample rates |
| `DelayBoundsCheck` | Ensure delays are positive and reasonable | Negative delays, delays > 100ms (likely error) |
| `CrossCorrelationQuality` | Warn if correlation < 0.7 | Poor signal quality, noise |

### Optimization Passes

| Pass | Description | Optimization |
|------|-------------|--------------|
| `IRWindowOptimization` | Auto-window IR to remove late reflections | Shorter IR → faster processing |
| `FractionalDelayUpgrade` | Replace integer-sample delays → allpass fractional delays | Higher precision alignment |
| `GroupDelaySmoothing` | Apply smoothing to noisy group delay curves | Reduce measurement noise |

### Lowering Passes

| Pass | Description | Target |
|------|-------------|--------|
| `DeconvolutionToFFT` | Lower IR extraction → partitioned FFT convolution | MLIR linalg + FFT |
| `CrossCorrToVectorized` | Vectorize cross-correlation (SIMD) | MLIR vector dialect |
| `CUDALowering` | GPU kernels for large IR processing | CUDA/ROCm |

---

## Why This is AWESOME for Morphogen

### 1. Cross-Domain Operator Reuse

| Domain | Reused Operators | Use Case |
|--------|------------------|----------|
| **Audio** | FFT, IR extraction, cross-correlation | Time alignment, Auto-EQ |
| **Physics** | Cross-correlation, group delay | Modal analysis, vibration testing |
| **Graphics** | Phase alignment | Stereo 3D rendering |
| **Finance** | Cross-correlation | Asset correlation analysis |

**Same math, different domains.**

### 2. Natural Morphogen Workflow

Time alignment is a **textbook Morphogen pipeline**:

```
Measurement (sine_sweep)
    ↓
Recording (capture responses)
    ↓
Analysis (IR extraction, peak detection, group delay)
    ↓
Design (delay_designer, phase matching)
    ↓
Export (miniDSP, JSON)
    ↓
Validation (measure again, verify)
```

Every step is:
- ✅ Deterministic
- ✅ GPU-friendly
- ✅ Composable
- ✅ Reusable across domains

### 3. Clean Operator Boundaries

Each operator has:
- ✅ Single responsibility (peak detection ≠ delay design)
- ✅ Clear inputs/outputs
- ✅ No hidden state
- ✅ Composable primitives

### 4. MLIR-Ready

All operators map cleanly to MLIR:

| Operator | MLIR Dialect | Lowering |
|----------|--------------|----------|
| `sine_sweep` | `morphogen.signal` | Vectorized sin() |
| `fft` | `fft.fft_1d` | Vendor FFT (FFTW, cuFFT) |
| `impulse_response_extractor` | `linalg` + `fft` | FFT-based deconvolution |
| `cross_correlation` | `linalg.dot` | SIMD dot product |
| `group_delay` | `linalg` | Phase unwrap + derivative |
| `delay_designer` | `arith` + `scf.for` | Simple arithmetic |

### 5. Extends to Other Morphogen Use Cases

| Application | Time Alignment Operators |
|-------------|-------------------------|
| **Guitar modal modeling** | IR extraction, group delay |
| **Room correction** | Same measurement pipeline |
| **Beamforming** | Cross-correlation, delay computation |
| **Echo cancellation** | Cross-correlation, adaptive delays |
| **Speaker design** | Crossover phase alignment, group delay |

---

## Implementation Roadmap

### v0.8 (Immediate)
- ⬜ Add `measurement` subcategory to Layer 5
  - `sine_sweep`, `impulse_train`
- ⬜ Add `analysis` subcategory to Layer 5
  - `impulse_response_extractor`, `ir_peak_detect`, `cross_correlation`
- ⬜ Define `ImpulseResponse`, `DelayTime`, `DelayMap` types
- ⬜ Basic IR extraction workflow (sweep → recording → IR)

### v0.9 (Complete Time Alignment)
- ⬜ Add `alignment` subcategory to Layer 5
  - `delay_designer`, `crossover_phase_aligner`
- ⬜ Add `export` subcategory to Layer 5
  - `export_delays` (JSON, CSV)
- ⬜ Implement `group_delay` operator
- ⬜ Add `ImpulseResponseRef` with auto-anchors
- ⬜ Complete car audio example
- ⬜ Validation + optimization passes

### v1.0 (Advanced Features)
- ⬜ `export_delays` for miniDSP XML format
- ⬜ `export_report` (HTML/PDF with plots)
- ⬜ Fractional-sample delays (allpass filters)
- ⬜ GPU acceleration for large IR processing
- ⬜ Integration with Auto-EQ operators

---

## Comparison: Time Alignment vs Auto-EQ

Both workflows share operators but solve different problems:

| Feature | Time Alignment | Auto-EQ |
|---------|----------------|---------|
| **Goal** | Align arrival times + phase | Flatten frequency response |
| **Measurement** | Sine sweep | Sine sweep |
| **Analysis** | Peak detection, cross-correlation, group delay | FFT magnitude, smoothing |
| **Output** | Delays (ms) | EQ filters (gain vs frequency) |
| **Shared Ops** | `sine_sweep`, `impulse_response_extractor`, `fft` | `sine_sweep`, `impulse_response_extractor`, `fft` |
| **Unique Ops** | `ir_peak_detect`, `cross_correlation`, `group_delay`, `delay_designer` | `spectral_smoothing`, `target_curve`, `eq_designer` |

**Morphogen wins:** Same measurement infrastructure, different analysis → different outputs.

---

## References

- **../specifications/operator-registry.md** — Operator registry structure (7 layers)
- **AUDIO_SPECIFICATION.md** — Morphogen.Audio dialect specification
- **OPERATOR_REGISTRY_EXPANSION.md** — Seven domain expansion plan
- **ADR-002** — Cross-domain architectural patterns
- **../specifications/transform.md** — Transform operators (FFT, STFT)

---

## Summary

Time alignment is a **perfect Morphogen workflow** that demonstrates:

1. ✅ **Operator reuse** — Same FFT/IR ops used across audio, physics, graphics
2. ✅ **Clean composition** — Measurement → Analysis → Design → Export
3. ✅ **MLIR-friendly** — All ops map cleanly to vectorized/GPU code
4. ✅ **Domain extensibility** — Same operators apply to room correction, beamforming, modal analysis
5. ✅ **Real-world impact** — Solves critical problem in pro audio (car audio, studio monitors)

**Adding time alignment operators extends Morphogen's AudioDomain with minimal new infrastructure, maximum reuse, and natural composability.**

---

**End of Time Alignment Operators Specification**
