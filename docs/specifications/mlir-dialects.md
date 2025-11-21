# SPEC: Morphogen MLIR Dialects

**Version:** 1.0 Draft
**Status:** RFC
**Last Updated:** 2025-11-13

---

## Overview

Morphogen defines **four core MLIR dialects** that form the intermediate representation between the Graph IR and executable code:

1. **morphogen.stream** — Audio/control signal operations
2. **morphogen.field** — Spatial field operations
3. **morphogen.transform** — Domain transforms (FFT, STFT, etc.)
4. **morphogen.schedule** — Multirate scheduling and fencing

**Design Principle:** Keep dialects minimal, sharply defined, and avoid scope creep. Lower to standard MLIR dialects (linalg, affine, vector, arith, math) as soon as possible.

---

## Dialect 1: morphogen.stream

**Purpose:** Operations on time-varying signals (audio, control).

### Type: !morphogen.stream<T>

```mlir
!morphogen.stream<f32>              // Audio signal
!morphogen.stream<f64>              // High-precision signal
!morphogen.stream<complex<f32>>     // Complex signal (spectrum)
!morphogen.stream<vec<2xf32>>       // Stereo signal
```

**Attributes:**
- `domain` — time, frequency, cepstral
- `rate` — audio, control
- `sample_rate` — Samples per second

**Example:**
```mlir
%sig = ... : !morphogen.stream<f32, domain="time", rate="audio", sample_rate=48000>
```

---

### Operations

#### morphogen.stream.sample

**Description:** Sample a stream at current time.

**Syntax:**
```mlir
%value = morphogen.stream.sample %stream : !morphogen.stream<f32> -> f32
```

**Lowering:**
```mlir
// Lower to memory read
%value = memref.load %buffer[%sample_index] : memref<?xf32>
```

---

#### morphogen.stream.delay

**Description:** Delay a stream by N samples.

**Syntax:**
```mlir
%delayed = morphogen.stream.delay %stream, %samples : !morphogen.stream<f32>, i32 -> !morphogen.stream<f32>
```

**Attributes:**
- `max_delay` — Maximum delay capacity

**Lowering:**
```mlir
// Lower to circular buffer
%write_idx = arith.remui %current_sample, %max_delay : i32
memref.store %input, %delay_buffer[%write_idx] : memref<?xf32>

%read_idx = arith.subi %current_sample, %delay_samples : i32
%read_idx_wrapped = arith.remui %read_idx, %max_delay : i32
%output = memref.load %delay_buffer[%read_idx_wrapped] : memref<?xf32>
```

---

#### morphogen.stream.mix

**Description:** Mix multiple streams.

**Syntax:**
```mlir
%mixed = morphogen.stream.mix %stream1, %stream2, %stream3 : !morphogen.stream<f32> -> !morphogen.stream<f32>
```

**Lowering:**
```mlir
%sum = arith.addf %stream1, %stream2 : f32
%result = arith.addf %sum, %stream3 : f32
```

---

#### morphogen.stream.amplify

**Description:** Multiply stream by gain factor.

**Syntax:**
```mlir
%amplified = morphogen.stream.amplify %stream, %gain : !morphogen.stream<f32>, f32 -> !morphogen.stream<f32>
```

**Lowering:**
```mlir
%result = arith.mulf %stream, %gain : f32
```

---

#### morphogen.stream.filter

**Description:** IIR or FIR filter.

**Syntax:**
```mlir
%filtered = morphogen.stream.filter %stream, %coeffs, %state
    : !morphogen.stream<f32>, memref<?xf32>, memref<?xf32> -> !morphogen.stream<f32>
```

**Attributes:**
- `order` — Filter order
- `type` — "iir" or "fir"

**Lowering:**
```mlir
// IIR: y[n] = b0*x[n] + b1*x[n-1] + ... - a1*y[n-1] - ...
// Lower to affine loop with history buffer
```

---

## Dialect 2: morphogen.field

**Purpose:** Operations on spatial fields (2D, 3D grids).

### Type: !morphogen.field<T, Dim>

```mlir
!morphogen.field<f32, 2>              // 2D scalar field
!morphogen.field<vec<2xf32>, 2>       // 2D vector field (velocity)
!morphogen.field<f32, 3>              // 3D scalar field
```

**Attributes:**
- `grid` — Grid metadata (spacing, centering, bounds)
- `domain` — space, k-space

---

### Operations

#### morphogen.field.create

**Description:** Create a field with given shape and spacing.

**Syntax:**
```mlir
%field = morphogen.field.create %shape, %spacing, %init_value
    : index, f32, f32 -> !morphogen.field<f32, 2>
```

**Attributes:**
- `centering` — "node" or "cell"
- `boundary` — "periodic", "clamp", "reflect"

**Lowering:**
```mlir
%buffer = memref.alloc(%shape) : memref<?x?xf32>
linalg.fill ins(%init_value : f32) outs(%buffer : memref<?x?xf32>)
```

---

#### morphogen.field.stencil

**Description:** Apply stencil operation (Laplacian, gradient, etc.).

**Syntax:**
```mlir
%result = morphogen.field.stencil %field, %radius, %weights
    : !morphogen.field<f32, 2>, i32, memref<?xf32> -> !morphogen.field<f32, 2>
```

**Attributes:**
- `stencil_type` — "laplacian", "gradient", "custom"

**Lowering:**
```mlir
// Lower to affine loops with neighborhood access
affine.for %i = 0 to %height {
  affine.for %j = 0 to %width {
    %sum = arith.constant 0.0 : f32
    affine.for %di = -radius to radius {
      affine.for %dj = -radius to radius {
        %neighbor = memref.load %field[%i+%di, %j+%dj] : memref<?x?xf32>
        %weight = memref.load %weights[%di+radius, %dj+radius] : memref<?x?xf32>
        %product = arith.mulf %neighbor, %weight : f32
        %sum = arith.addf %sum, %product : f32
      }
    }
    memref.store %sum, %result[%i, %j] : memref<?x?xf32>
  }
}
```

---

#### morphogen.field.advect

**Description:** Advect field by velocity field.

**Syntax:**
```mlir
%advected = morphogen.field.advect %field, %velocity, %dt, %method
    : !morphogen.field<f32, 2>, !morphogen.field<vec<2xf32>, 2>, f32, string -> !morphogen.field<f32, 2>
```

**Attributes:**
- `method` — "semi_lagrangian", "bfecc", "maccormack"

**Lowering:**
```mlir
// Semi-Lagrangian:
// 1. Backtrace particle positions
// 2. Interpolate values at backtraced positions
affine.for %i = 0 to %height {
  affine.for %j = 0 to %width {
    %vel = memref.load %velocity[%i, %j] : memref<?x?xvec<2xf32>>
    %back_x = arith.subf %i, arith.mulf(%vel[0], %dt) : f32
    %back_y = arith.subf %j, arith.mulf(%vel[1], %dt) : f32
    %value = interpolate(%field, %back_x, %back_y) : f32
    memref.store %value, %result[%i, %j] : memref<?x?xf32>
  }
}
```

---

#### morphogen.field.reduce

**Description:** Reduce field to scalar (sum, max, min, mean).

**Syntax:**
```mlir
%scalar = morphogen.field.reduce %field, %op : !morphogen.field<f32, 2>, string -> f32
```

**Attributes:**
- `reduction_op` — "sum", "max", "min", "mean"

**Lowering:**
```mlir
// Lower to linalg.reduce
%result = linalg.reduce ins(%field : memref<?x?xf32>)
                        outs(%scalar : f32)
                        dimensions = [0, 1]
  (%arg0: f32, %arg1: f32) {
    %sum = arith.addf %arg0, %arg1 : f32
    linalg.yield %sum : f32
  }
```

---

## Dialect 3: morphogen.transform

**Purpose:** Domain transforms (FFT, STFT, DCT, wavelets).

### Operations

#### morphogen.transform.fft

**Description:** Fast Fourier Transform (time → frequency).

**Syntax:**
```mlir
%spectrum = morphogen.transform.fft %signal, %window, %norm
    : !morphogen.stream<f32> -> !morphogen.stream<complex<f32>>
```

**Attributes:**
- `nfft` — FFT size
- `window` — "hann", "hamming", "blackman", "rectangular"
- `norm` — "ortho", "forward", "backward"

**Lowering:**
```mlir
// 1. Apply window
%windowed = apply_window(%signal, %window) : tensor<?xf32>

// 2. FFT (lower to vendor library or reference)
%spectrum = fft.fft_1d %windowed {norm = "ortho"} : tensor<?xf32> -> tensor<?xcomplex<f32>>
```

---

#### morphogen.transform.ifft

**Description:** Inverse Fast Fourier Transform (frequency → time).

**Syntax:**
```mlir
%signal = morphogen.transform.ifft %spectrum, %norm
    : !morphogen.stream<complex<f32>> -> !morphogen.stream<f32>
```

**Lowering:**
```mlir
%signal = fft.ifft_1d %spectrum {norm = "ortho"} : tensor<?xcomplex<f32>> -> tensor<?xf32>
```

---

#### morphogen.transform.stft

**Description:** Short-Time Fourier Transform.

**Syntax:**
```mlir
%spectrogram = morphogen.transform.stft %signal, %n_fft, %hop_length, %window
    : !morphogen.stream<f32>, i32, i32, string -> tensor<?x?xcomplex<f32>>
```

**Attributes:**
- `n_fft` — FFT size per frame
- `hop_length` — Samples between frames
- `center` — Center windowing (bool)

**Lowering:**
```mlir
// 1. Frame signal (windowing)
%frames = frame_signal(%signal, %n_fft, %hop_length) : tensor<?x?xf32>

// 2. Apply window to each frame
%windowed = apply_window_2d(%frames, %window) : tensor<?x?xf32>

// 3. FFT each frame
%spectrogram = fft.fft_2d %windowed {norm = "ortho"} : tensor<?x?xf32> -> tensor<?x?xcomplex<f32>>
```

---

#### morphogen.transform.istft

**Description:** Inverse Short-Time Fourier Transform.

**Syntax:**
```mlir
%signal = morphogen.transform.istft %spectrogram, %hop_length, %window
    : tensor<?x?xcomplex<f32>>, i32, string -> !morphogen.stream<f32>
```

**Lowering:**
```mlir
// 1. IFFT each frame
%frames = fft.ifft_2d %spectrogram : tensor<?x?xcomplex<f32>> -> tensor<?x?xf32>

// 2. Overlap-add reconstruction
%signal = overlap_add(%frames, %hop_length, %window) : tensor<?xf32>
```

---

#### morphogen.transform.fft2d

**Description:** 2D FFT (space → k-space).

**Syntax:**
```mlir
%k_field = morphogen.transform.fft2d %field, %norm
    : !morphogen.field<f32, 2> -> !morphogen.field<complex<f32>, 2>
```

**Lowering:**
```mlir
%k_field = fft.fft_2d %field {norm = "ortho"} : tensor<?x?xf32> -> tensor<?x?xcomplex<f32>>
```

---

## Dialect 4: morphogen.schedule

**Purpose:** Multirate scheduling, event fencing, cross-rate resampling.

### Operations

#### morphogen.schedule.rate

**Description:** Declare execution rate for a block.

**Syntax:**
```mlir
morphogen.schedule.rate "audio" {
  // Operations run at audio rate
  %osc = morphogen.stream.sine %freq : f32 -> !morphogen.stream<f32>
  %lpf = morphogen.stream.filter %osc, ... : !morphogen.stream<f32> -> !morphogen.stream<f32>
}
```

**Attributes:**
- `rate_name` — "audio", "control", "visual", "sim"
- `sample_rate` — Samples per second

---

#### morphogen.schedule.fence

**Description:** Sample-accurate synchronization barrier.

**Syntax:**
```mlir
morphogen.schedule.fence %event_time : i64
```

**Semantics:**
- Execution pauses at `event_time`
- All pending operations complete
- Events fire
- Execution resumes

**Lowering:**
```mlir
// Lower to control flow
scf.if %current_sample == %event_time {
  // Fire events
  call @fire_events(%event_queue) : (!llvm.ptr) -> ()
}
```

---

#### morphogen.schedule.resample

**Description:** Resample stream from one rate to another.

**Syntax:**
```mlir
%resampled = morphogen.schedule.resample %stream, %from_rate, %to_rate, %mode
    : !morphogen.stream<f32>, i32, i32, string -> !morphogen.stream<f32>
```

**Attributes:**
- `mode` — "hold", "linear", "cubic", "sinc"

**Lowering:**
```mlir
// Zero-order hold
%ratio = arith.divsi %to_rate, %from_rate : i32
%output_size = arith.muli %input_size, %ratio : i32
%resampled = memref.alloc(%output_size) : memref<?xf32>

affine.for %i = 0 to %output_size {
  %src_idx = arith.divsi %i, %ratio : i32
  %value = memref.load %input[%src_idx] : memref<?xf32>
  memref.store %value, %resampled[%i] : memref<?xf32>
}
```

---

#### morphogen.schedule.hop

**Description:** Execute a hop (block of samples).

**Syntax:**
```mlir
morphogen.schedule.hop %start_sample, %hop_size {
  // Execute all rate groups for this hop
  ^bb0(%sample_idx: index):
    // Operations
}
```

**Lowering:**
```mlir
scf.for %i = %start_sample to %end_sample step %hop_size {
  // Execute hop body
}
```

---

## Lowering Strategy

Morphogen dialects lower to standard MLIR dialects in stages:

```
┌─────────────────────────────────────────┐
│  Morphogen Graph IR (JSON)                  │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│  Morphogen Dialects                         │
│  - morphogen.stream                         │
│  - morphogen.field                          │
│  - morphogen.transform                      │
│  - morphogen.schedule                       │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│  Standard MLIR Dialects                 │
│  - linalg (field ops)                   │
│  - affine (loops, stencils)             │
│  - vector (SIMD)                        │
│  - arith (arithmetic)                   │
│  - math (transcendentals)               │
│  - scf (control flow)                   │
│  - memref (memory)                      │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│  Backend Dialects                       │
│  - llvm (CPU)                           │
│  - gpu (CUDA/ROCm)                      │
│  - spirv (Vulkan)                       │
└─────────────────┬───────────────────────┘
                  ▼
┌─────────────────────────────────────────┐
│  Executable Code                        │
└─────────────────────────────────────────┘
```

---

## Example: Simple Synth

### Morphogen.Audio DSL

```morphogen
scene SimpleSynth {
  let osc = sine(440Hz)
  let env = adsr(attack=0.01s, decay=0.1s, sustain=0.7, release=0.3s)
  let modulated = osc * env
  out mono = modulated
}
```

---

### Morphogen Graph IR

```json
{
  "nodes": [
    {"id": "osc", "op": "sine", "params": {"freq": "440Hz"}, "rate": "audio"},
    {"id": "env", "op": "adsr", "params": {...}, "rate": "control"},
    {"id": "mul", "op": "multiply", "rate": "audio"}
  ],
  "edges": [
    {"from": "osc:out", "to": "mul:in1"},
    {"from": "env:out", "to": "mul:in2"}
  ]
}
```

---

### Morphogen Dialects (MLIR)

```mlir
module {
  func.func @simple_synth(%sample_rate: i32) -> !morphogen.stream<f32> {
    // Audio rate: Oscillator
    %osc = morphogen.schedule.rate "audio" {
      %freq = arith.constant 440.0 : f32
      %phase = arith.constant 0.0 : f32
      %sig = morphogen.stream.sine %freq, %phase : f32, f32 -> !morphogen.stream<f32>
      morphogen.schedule.yield %sig : !morphogen.stream<f32>
    }

    // Control rate: Envelope
    %env = morphogen.schedule.rate "control" {
      %attack = arith.constant 0.01 : f32
      %decay = arith.constant 0.1 : f32
      %sustain = arith.constant 0.7 : f32
      %release = arith.constant 0.3 : f32
      %envelope = morphogen.stream.adsr %attack, %decay, %sustain, %release
          : f32, f32, f32, f32 -> !morphogen.stream<f32>
      morphogen.schedule.yield %envelope : !morphogen.stream<f32>
    }

    // Resample control → audio
    %env_upsampled = morphogen.schedule.resample %env, 1000, 48000, "linear"
        : !morphogen.stream<f32>, i32, i32, string -> !morphogen.stream<f32>

    // Multiply
    %modulated = morphogen.stream.multiply %osc, %env_upsampled
        : !morphogen.stream<f32>, !morphogen.stream<f32> -> !morphogen.stream<f32>

    return %modulated : !morphogen.stream<f32>
  }
}
```

---

### Lowered to Standard Dialects

```mlir
module {
  func.func @simple_synth(%sample_rate: i32) -> memref<?xf32> {
    %c48000 = arith.constant 48000 : i32
    %c1000 = arith.constant 1000 : i32
    %duration = arith.constant 48000 : index  // 1 second
    %output = memref.alloc(%duration) : memref<?xf32>

    // Oscillator: y = sin(2π * freq * t / sample_rate)
    %freq = arith.constant 440.0 : f32
    %two_pi = arith.constant 6.28318530718 : f32
    %freq_norm = arith.divf %freq, %sample_rate : f32

    scf.for %i = 0 to %duration step 1 {
      %t = arith.index_cast %i : index to f32
      %phase = arith.mulf %freq_norm, %t : f32
      %phase_wrapped = arith.mulf %phase, %two_pi : f32
      %sample = math.sin %phase_wrapped : f32

      // Envelope (simplified: linear decay)
      %env = arith.constant 1.0 : f32  // TODO: Implement ADSR

      // Multiply
      %output_sample = arith.mulf %sample, %env : f32

      memref.store %output_sample, %output[%i] : memref<?xf32>
    }

    return %output : memref<?xf32>
  }
}
```

---

## Implementation Checklist

### Phase 1: Define Dialects (Weeks 9-10)
- [ ] Define types: `!morphogen.stream`, `!morphogen.field`
- [ ] Define operations for each dialect
- [ ] Write MLIR dialect .td files (TableGen)
- [ ] Generate C++ dialect code

### Phase 2: Lowering Passes (Week 11)
- [ ] Implement morphogen.stream → arith/math lowering
- [ ] Implement morphogen.field → linalg/affine lowering
- [ ] Implement morphogen.transform → fft dialect (or vendor calls)
- [ ] Implement morphogen.schedule → scf lowering

### Phase 3: Backend Integration (Week 12)
- [ ] Lower to LLVM dialect (CPU)
- [ ] Integrate FFT provider (FFTW stub)
- [ ] End-to-end test: compile simple graph

---

## Summary

The Morphogen MLIR Dialects provide:

✅ **Four minimal dialects** — stream, field, transform, schedule
✅ **Clean lowering path** — Morphogen → standard dialects → backends
✅ **Avoid scope creep** — Keep dialects small and focused
✅ **Extensible** — Add new ops without breaking existing code

This is the **compiler foundation** that turns Graph IR into executable code.

---

## References

- `graph-ir.md` — Graph IR is input to MLIR lowering
- `type-system.md` — Morphogen types map to MLIR types
- `scheduler.md` — morphogen.schedule implements scheduler semantics
- `transform.md` — morphogen.transform implements transform ops
- `operator-registry.md` — Operators lower to dialect ops
