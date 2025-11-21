# SPEC: Transform Dialect

**Version:** 1.0 Draft
**Status:** RFC
**Last Updated:** 2025-11-11

---

## Overview

The Transform Dialect makes **domain changes** a first-class operation in Morphogen. Transforms are not special-cased utilities; they are core grammatical elements with strict semantics, deterministic behavior, and profile-driven tuning.

**Design Principle:** FFT is not special — it's one instance of `transform.to(domain="frequency")`. All transforms follow the same pattern.

**Related Documentation:**
- **[DSL Framework Design](../architecture/dsl-framework-design.md)** - Vision for first-class translations with declarative syntax
- [Mathematical Transformation Metaphors](../reference/math-transformation-metaphors.md) - Pedagogical frameworks and intuitive understanding
- [Morphogen Categorical Structure](../architecture/morphogen-categorical-structure.md) - Functorial semantics of transforms

**Note:** This document describes the current procedural transform API (`transform.to/from`). For the future vision of declarative translation definitions with structure preservation and operator mappings, see the DSL Framework Design document.

---

## Core Operations

### `transform.to(x, domain, method, **attrs) -> Stream`

Convert stream `x` to a different domain representation.

**Parameters:**
- `x`: Source stream (any `Stream<T,D,R>`)
- `domain`: Target domain name (string)
- `method`: Transform method (domain-specific)
- `**attrs`: Transform-specific attributes (window, overlap, family, etc.)

**Returns:** `Stream<T',D',R'>` in target domain

**Example:**
```morphogen
let spec = transform.to(signal, domain="frequency", method="fft", window="hann")
```

---

### `transform.from(x, domain, method, **attrs) -> Stream`

Convert stream `x` back from a domain representation.

**Parameters:**
- `x`: Source stream in transformed domain
- `domain`: Source domain to convert from
- `method`: Inverse transform method
- `**attrs`: Transform-specific attributes

**Returns:** `Stream<T,D,R>` in original domain

**Example:**
```morphogen
let signal = transform.from(spectrum, domain="frequency", method="ifft")
```

---

### `transform.reparam(x, mapping) -> Stream`

Apply a coordinate transformation without changing domain.

**Parameters:**
- `x`: Source stream
- `mapping`: Coordinate mapping function or matrix

**Returns:** `Stream<T,D,R>` with transformed coordinates

**Example:**
```morphogen
# Mel-frequency warping
let mel_spec = transform.reparam(spectrum, mapping=mel_scale(n_mels=128))
```

---

## Domain Pairs & Methods

### 1. Time ↔ Frequency

**Forward Transforms:**

#### FFT (Fast Fourier Transform)
```morphogen
transform.to(sig, domain="frequency", method="fft",
             window="hann",      # Window function
             nfft=null,          # FFT size (default: signal length)
             center=true,        # Center window
             norm="ortho")       # Normalization mode
```

**Attributes:**
- `window`: `"hann"`, `"hamming"`, `"blackman"`, `"kaiser"`, `"rectangular"`
- `nfft`: FFT size (power of 2, >= signal length)
- `center`: Center windowing (bool)
- `norm`: `"ortho"` (√N), `"forward"` (1), `"backward"` (1/N)

**Returns:** `Stream<Complex<f32>, 1D, audio>` with frequency bins

---

#### STFT (Short-Time Fourier Transform)
```morphogen
transform.to(sig, domain="frequency", method="stft",
             window="hann",
             n_fft=2048,
             hop_length=512,
             center=true,
             norm="ortho")
```

**Attributes:**
- `window`: Window function
- `n_fft`: FFT size per frame
- `hop_length`: Samples between frames
- `center`: Pad signal for centered windows
- `norm`: Normalization mode

**Returns:** `Stream<Complex<f32>, 2D, audio>` (time × frequency)

---

**Inverse Transforms:**

```morphogen
# IFFT
transform.from(spec, domain="frequency", method="ifft",
               length=null,  # Output length (default: infer from spectrum)
               norm="ortho")

# ISTFT (overlap-add reconstruction)
transform.from(stft, domain="frequency", method="istft",
               hop_length=512,
               window="hann",
               center=true,
               length=null)
```

---

### 2. Time ↔ Cepstral

#### DCT (Discrete Cosine Transform)
```morphogen
transform.to(sig, domain="cepstral", method="dct",
             type=2,      # DCT type (1-4)
             norm="ortho")
```

**Use cases:** Compression, MFCC computation, cepstral analysis

**Inverse:**
```morphogen
transform.from(ceps, domain="cepstral", method="idct", type=2, norm="ortho")
```

---

### 3. Time ↔ Wavelet

#### Wavelet Transform
```morphogen
transform.to(sig, domain="wavelet", method="cwt",
             wavelet="morlet",   # Wavelet family
             scales=[1..128],    # Scale values
             sampling_period=1.0)
```

**Wavelet families:** `"morlet"`, `"mexican_hat"`, `"paul"`, `"dog"` (derivative of Gaussian)

**Returns:** `Stream<Complex<f32>, 2D, audio>` (scale × time)

**Inverse:**
```morphogen
transform.from(cwt, domain="wavelet", method="icwt", wavelet="morlet")
```

---

### 4. Space ↔ k-space (Spatial Frequency)

For 2D/3D fields (PDEs, images, volumes):

```morphogen
# 2D Fourier transform (spatial → k-space)
let k_field = transform.to(field, domain="k-space", method="fft2d",
                            norm="ortho")

# Apply filter in k-space (e.g., low-pass)
let filtered_k = k_field * gaussian_kernel(sigma=5.0)

# Transform back to spatial domain
let filtered = transform.from(filtered_k, domain="k-space", method="ifft2d")
```

**Use cases:** PDE spectral methods, image filtering, diffusion

---

### 5. Linear ↔ Perceptual

#### Mel Scale (Frequency Warping)
```morphogen
# Frequency → Mel frequency
let mel_spec = transform.reparam(spectrum, mapping=mel_scale(
    n_mels=128,
    fmin=0Hz,
    fmax=8000Hz
))

# Mel frequency → Frequency
let lin_spec = transform.reparam(mel_spec, mapping=inverse_mel_scale())
```

**Use cases:** Perceptual audio features, voice processing

---

### 6. Graph ↔ Spectral

For graph/network data:

```morphogen
# Graph Laplacian eigenbasis
let spectral = transform.to(graph, domain="spectral", method="laplacian",
                             k=50)  # Number of eigenvectors

# Graph signal filtering
let filtered = spectral * spectral_filter

# Back to graph domain
let smooth = transform.from(filtered, domain="spectral", method="inverse_laplacian")
```

**Use cases:** Graph signal processing, network analysis, smoothing

---

## Determinism & Profiles

### Strict Profile
- **Bit-exact** FFT (aligned to reference implementations)
- Fixed normalization
- Deterministic phase handling
- Golden test vectors included

### Repro Profile
- **Deterministic within floating-point precision**
- Vendor FFT libraries allowed (FFTW, MKL)
- Consistent windowing/normalization

### Live Profile
- **Lowest latency**
- Allows approximations (e.g., shorter FFT, lower overlap)
- Replayable but not bit-exact

---

## Normalization Modes

All transforms support explicit normalization control:

- `"ortho"`: Orthonormal (forward and inverse both √N)
- `"forward"`: Forward scaled by 1, inverse by 1/N
- `"backward"`: Forward scaled by 1/N, inverse by 1

**Default:** `"ortho"` (symmetric, energy-preserving)

---

## Window Functions

Standard window functions for all time-frequency transforms:

| Window | Formula | Use Case |
|--------|---------|----------|
| `hann` | Cosine-squared | General purpose, good sidelobe suppression |
| `hamming` | Raised cosine | Slightly better frequency resolution |
| `blackman` | Three-term cosine | Excellent sidelobe rejection |
| `kaiser` | Bessel-based, tunable β | Adjustable tradeoff (β parameter) |
| `rectangular` | No tapering | Maximum frequency resolution (high leakage) |

**Profile influence:**
- Strict: Exact window coefficients (reference implementation)
- Repro: Vendor-optimized windows (consistent results)
- Live: Fast approximations allowed

---

## Error Handling

### Type Errors
```morphogen
# ERROR: Cannot FFT a 2D field
let spec = transform.to(field2d, domain="frequency", method="fft")
# → Use method="fft2d" for 2D data
```

### Domain Mismatches
```morphogen
# ERROR: Inverse domain must match forward domain
let spec = transform.to(sig, domain="frequency", method="fft")
let back = transform.from(spec, domain="wavelet", method="icwt")
# → Domain mismatch: expected "frequency"
```

### Attribute Validation
```morphogen
# ERROR: hop_length must divide n_fft evenly for perfect reconstruction
transform.to(sig, domain="frequency", method="stft", n_fft=2048, hop_length=513)
```

---

## Implementation Notes

### Phase 1 (v0.4.0)
- ✅ FFT/IFFT (1D, time↔frequency)
- ✅ STFT/ISTFT (spectrogram)
- ✅ Window functions (hann, hamming, blackman)
- ✅ Profile-driven normalization

### Phase 2 (v0.5.0)
- DCT/IDCT (cepstral)
- Wavelet transforms (CWT/ICWT)
- Mel-scale warping

### Phase 3 (v0.6.0)
- FFT2D/IFFT2D (space↔k-space)
- Graph spectral transforms
- Vendor FFT provider integration (FFTW, MKL, cuFFT)

---

## Examples

### Example 1: Spectral Filtering
```morphogen
scene SpectralFilter {
  let sig = sine(440Hz) + sine(880Hz) + noise(seed=42) * 0.1

  # Transform to frequency domain
  let spec = transform.to(sig, domain="frequency", method="fft", window="hann")

  # Apply filter (zero out high frequencies)
  let filtered_spec = spec * lowpass_mask(cutoff=1000Hz)

  # Transform back
  let clean = transform.from(filtered_spec, domain="frequency", method="ifft")

  out mono = clean
}
```

### Example 2: STFT-based Processing
```morphogen
scene VocoderEffect {
  let voice = input_mono()

  # Compute STFT
  let stft = transform.to(voice, domain="frequency", method="stft",
                          n_fft=2048, hop_length=512)

  # Spectral manipulation (phase vocoder stretch)
  let stretched = time_stretch(stft, factor=1.5)

  # Reconstruct
  let output = transform.from(stretched, domain="frequency", method="istft",
                              hop_length=512)

  out mono = output
}
```

### Example 3: Mel-Frequency Features
```morphogen
scene MelFeatures {
  let audio = input_mono()

  # Compute spectrogram
  let spec = transform.to(audio, domain="frequency", method="stft", n_fft=2048)

  # Convert to Mel scale
  let mel = transform.reparam(spec, mapping=mel_scale(n_mels=128, fmax=8000Hz))

  # Log magnitude
  let log_mel = log(abs(mel) + 1e-8)

  # Visualize
  out visual = colorize(log_mel, palette="viridis")
}
```

---

## 7. Spatial Transformations & Coordinate Systems

**Inspired by TiaCAD's deterministic transform model.**

Spatial transformations are geometric operations on fields, meshes, or visual objects. Unlike time-frequency transforms that change representation, spatial transforms change position, orientation, or coordinate system while preserving the underlying data.

**Key principles (from TiaCAD):**
- **Explicit origins** — all rotations/scales specify a center point
- **Pure functions** — transforms create new objects, never mutate
- **Deterministic** — same transform, same result (strict profile)
- **Ordered composition** — transform chains are explicit sequences

---

### 7.1 Affine Transformations

Affine transforms preserve parallel lines and include translation, rotation, scale, and shear.

#### Translation

```morphogen
transform.translate(object, offset: Vec<Dim>)
```

**Parameters:**
- `object`: Field, mesh, or visual object
- `offset`: Translation vector (units match object's frame)

**Returns:** Transformed object

**Example:**
```morphogen
let field = field.zeros(shape=(100, 100))
let moved = transform.translate(field, offset=(10, 5))

# 3D translation
let mesh = mesh.sphere(radius=5mm)
let moved_mesh = transform.translate(mesh, offset=(10mm, 0, 5mm))
```

**Determinism:** Strict

---

#### Rotation

```morphogen
transform.rotate(object, angle, axis="z", origin=Anchor | Vec<Dim>)
```

**Parameters:**
- `object`: Object to rotate
- `angle`: Rotation angle (units: `rad` or `deg`)
- `axis`: Rotation axis (`"x"`, `"y"`, `"z"` for 3D; always Z for 2D)
- `origin`: **Explicit rotation center** (anchor or vector)

**Returns:** Rotated object

**Examples:**
```morphogen
# 2D rotation around center (EXPLICIT origin)
let box = geom.box(10mm, 10mm, 10mm)
let rotated = transform.rotate(
    box,
    angle = 45 deg,
    origin = box.anchor("center")  # Explicit!
)

# 3D rotation around custom point
let rotated_3d = transform.rotate(
    mesh,
    angle = 90 deg,
    axis = "y",
    origin = mesh.anchor("edge_bottom_left")
)

# ❌ INVALID: Implicit origin not allowed
let bad = transform.rotate(mesh, 45 deg)  # Compiler error: origin required
```

**Determinism:** Strict

**Key insight from TiaCAD:** Explicit origins prevent rotation-around-origin bugs. Users must think about where rotation happens.

---

#### Scale

```morphogen
transform.scale(object, factor: f64 | Vec<Dim>, origin=Anchor | Vec<Dim>)
```

**Parameters:**
- `object`: Object to scale
- `factor`: Uniform scale (scalar) or non-uniform (vector)
- `origin`: **Explicit scale center**

**Returns:** Scaled object

**Examples:**
```morphogen
# Uniform scale from center
let scaled = transform.scale(
    mesh,
    factor = 2.0,
    origin = mesh.anchor("center")
)

# Non-uniform scale (stretch)
let stretched = transform.scale(
    field,
    factor = (2.0, 1.0, 0.5),  # 2x in X, 1x in Y, 0.5x in Z
    origin = (0, 0, 0)
)
```

**Determinism:** Strict

---

#### Shear

```morphogen
transform.shear(object, axis, angle, origin=Anchor | Vec<Dim>)
```

**Parameters:**
- `object`: Object to shear
- `axis`: Shear axis (`"x"`, `"y"`, `"z"`)
- `angle`: Shear angle (or factor)
- `origin`: Shear reference point

**Example:**
```morphogen
# Shear in X direction
let sheared = transform.shear(field, axis="x", angle=15 deg, origin=(0,0))
```

**Determinism:** Strict

---

#### Mirror

```morphogen
transform.mirror(object, plane: String | Vec<Dim>)
```

**Parameters:**
- `object`: Object to mirror
- `plane`: Mirror plane (`"xy"`, `"xz"`, `"yz"` or normal vector)

**Example:**
```morphogen
# Mirror across XY plane
let mirrored = transform.mirror(mesh, plane="xy")

# Mirror across custom plane
let mirrored_custom = transform.mirror(mesh, plane=(1, 1, 0))  # Diagonal
```

**Determinism:** Strict

---

#### Generic Affine Transform

```morphogen
transform.affine(object, matrix: Mat<Dim+1, Dim+1>)
```

Apply arbitrary affine transformation via matrix.

**Parameters:**
- `object`: Object to transform
- `matrix`: (Dim+1)×(Dim+1) homogeneous transformation matrix

**Example:**
```morphogen
# Compose rotation + translation manually
let mat = affine_matrix(
    rotation = rotation_matrix(angle=45 deg, axis="z"),
    translation = (10, 5, 0)
)
let transformed = transform.affine(mesh, matrix=mat)
```

**Determinism:** Strict

---

### 7.2 Coordinate System Conversions

Convert between different coordinate representations (Cartesian, polar, spherical, cylindrical).

**See also:** coordinate-frames.md for full frame specification.

---

#### Cartesian ↔ Polar (2D)

```morphogen
# Cartesian → Polar
let polar_field = transform.to_coord(
    cartesian_field,
    coord_type = "polar",
    origin = (0, 0)  # Polar origin
)

# Polar → Cartesian
let cartesian = transform.to_coord(
    polar_field,
    coord_type = "cartesian"
)
```

**Coordinate mapping:**
- Cartesian `(x, y)` → Polar `(r, θ)`
  - `r = √(x² + y²)`
  - `θ = atan2(y, x)`
- Polar `(r, θ)` → Cartesian `(x, y)`
  - `x = r cos(θ)`
  - `y = r sin(θ)`

**Returns:** Field with new coordinate system (requires resampling)

**Determinism:** Repro (interpolation involved)

---

#### Cartesian ↔ Spherical (3D)

```morphogen
# Cartesian → Spherical
let spherical = transform.to_coord(
    cartesian_field_3d,
    coord_type = "spherical",
    origin = (0, 0, 0)
)

# Spherical → Cartesian
let cartesian = transform.to_coord(
    spherical,
    coord_type = "cartesian"
)
```

**Coordinate mapping:**
- Cartesian `(x, y, z)` → Spherical `(r, θ, φ)`
  - `r = √(x² + y² + z²)` (radial distance)
  - `θ = atan2(y, x)` (azimuthal angle)
  - `φ = acos(z / r)` (polar angle)
- Spherical `(r, θ, φ)` → Cartesian `(x, y, z)`
  - `x = r sin(φ) cos(θ)`
  - `y = r sin(φ) sin(θ)`
  - `z = r cos(φ)`

**Determinism:** Repro

---

#### Cartesian ↔ Cylindrical (3D)

```morphogen
# Cartesian → Cylindrical
let cylindrical = transform.to_coord(
    cartesian_field,
    coord_type = "cylindrical",
    axis = "z"  # Cylinder axis
)
```

**Coordinate mapping:**
- Cartesian `(x, y, z)` → Cylindrical `(ρ, θ, z)`
  - `ρ = √(x² + y²)` (radial distance from axis)
  - `θ = atan2(y, x)` (azimuthal angle)
  - `z = z` (height along axis)

**Determinism:** Repro

---

### 7.3 Projective Transformations

Projective transforms include perspective projection and homography.

#### Perspective Projection

```morphogen
transform.projection(
    object_3d,
    method = "perspective",
    fov = 60 deg,          # Field of view
    aspect = 16/9,         # Aspect ratio
    near = 0.1,            # Near clipping plane
    far = 100.0            # Far clipping plane
)
```

**Use cases:** 3D → 2D rendering, camera views

**Returns:** 2D projected object

**Determinism:** Strict

---

#### Homography (2D Perspective Warp)

```morphogen
transform.homography(image, matrix: Mat<3,3>)
```

Apply 2D perspective transformation.

**Parameters:**
- `image`: 2D field or image
- `matrix`: 3×3 homography matrix

**Use cases:** Image rectification, perspective correction

**Determinism:** Repro (interpolation)

---

### 7.4 Frame-Aware Transformations

Transformations integrate with the coordinate frame system (see coordinate-frames.md).

**Frame transformations:**

```morphogen
# Create frame
let frame_A = Frame<3, Cartesian, m>(
    origin = (0, 0, 0),
    basis = I  # Identity (no rotation)
)

# Transform frame (rotation + translation)
let frame_B = frame.transform(
    frame_A,
    operations = [
        (translate, offset=(10, 5, 0)),
        (rotate, angle=45 deg, axis="z")
    ]
)

# Convert point between frames
let point_in_A = (1, 2, 3)
let point_in_B = frame.to_frame(point_in_A, from=frame_A, to=frame_B)
```

**Key feature:** Transforms compose deterministically (order matters).

---

### 7.5 Transform Composition

Transforms can be composed explicitly:

```morphogen
# Sequential composition (order matters!)
let transformed = object
    |> transform.translate(offset=(10, 0, 0))
    |> transform.rotate(angle=45 deg, axis="z", origin=.center)
    |> transform.scale(factor=2.0, origin=.center)

# Verify order matters
let A = obj |> translate(...) |> rotate(...)
let B = obj |> rotate(...) |> translate(...)
assert_ne!(A, B)  # Different results!
```

**TiaCAD principle:** Sequential transforms are transparent. No hidden state.

---

### 7.6 Warping & Reparameterization

`transform.reparam` applies arbitrary coordinate mappings (warps).

**Extended usage for spatial fields:**

```morphogen
# Define warp function
let warp = λ (x, y): (x + 0.1 * sin(y), y + 0.1 * cos(x))

# Apply warp to field
let warped = transform.reparam(field, mapping=warp)

# Radial warp (fisheye effect)
let fisheye = transform.reparam(
    image,
    mapping = radial_warp(strength=0.5, center=(50, 50))
)

# Polar to Cartesian warp
let unwrapped = transform.reparam(
    polar_field,
    mapping = polar_to_cartesian(origin=(0, 0))
)
```

**Determinism:** Repro (requires interpolation)

---

### 7.7 Determinism Profiles for Spatial Transforms

| Transform | Strict | Repro | Live |
|-----------|--------|-------|------|
| Translation | ✅ Bit-exact | ✅ Bit-exact | ✅ Bit-exact |
| Rotation | ✅ Reference sin/cos | ✅ Vendor math lib | ⚠️ Fast approx |
| Scale | ✅ Bit-exact | ✅ Bit-exact | ✅ Bit-exact |
| Mirror | ✅ Bit-exact | ✅ Bit-exact | ✅ Bit-exact |
| Coord conversion | ⚠️ N/A (uses Repro) | ✅ Deterministic interp | ⚠️ Lower quality |
| Perspective | ✅ Reference math | ✅ Vendor math lib | ⚠️ Fast approx |
| Warp (reparam) | ⚠️ N/A (uses Repro) | ✅ Deterministic interp | ⚠️ Adaptive quality |

**Key constraint:** Coordinate conversions require interpolation, so they use **Repro** profile minimum.

---

### 7.8 Examples

#### Example 1: Rotate Field Around Custom Origin

```morphogen
scene RotatedField {
    # Create 2D field
    let field = field.from_function(
        shape = (100, 100),
        fn = λ (x, y): sin(x * 0.1) * cos(y * 0.1)
    )

    # Rotate 45° around center
    let rotated = transform.rotate(
        field,
        angle = 45 deg,
        origin = field.anchor("center")  # Explicit!
    )

    out visual = colorize(rotated)
}
```

---

#### Example 2: Cartesian to Polar Conversion

```morphogen
scene PolarView {
    # Create Cartesian field (grid)
    let cartesian = field.from_function(
        shape = (200, 200),
        bounds = ((-10, 10), (-10, 10)),
        fn = λ (x, y): exp(-(x² + y²) / 10)  # Gaussian
    )

    # Convert to polar coordinates
    let polar = transform.to_coord(
        cartesian,
        coord_type = "polar",
        origin = (0, 0)
    )

    # Polar field now has coordinates (r, θ)
    # Sample at r=5, θ=π/4
    let value = field.sample(polar, at=(5.0, π/4))

    out visual = [
        colorize(cartesian, title="Cartesian"),
        colorize(polar, title="Polar")
    ]
}
```

---

#### Example 3: Transform Chain (Geometry)

```morphogen
part TransformedBracket {
    # Start with simple box
    let base = geom.box(20mm, 10mm, 5mm)

    # Chain transforms (order matters!)
    let transformed = base
        # 1. Translate to origin
        |> transform.translate(offset=(-10mm, -5mm, 0))

        # 2. Rotate 30° around Z (now around origin because we translated)
        |> transform.rotate(angle=30 deg, axis="z", origin=(0, 0, 0))

        # 3. Scale 2x (from origin)
        |> transform.scale(factor=2.0, origin=(0, 0, 0))

        # 4. Translate back
        |> transform.translate(offset=(10mm, 5mm, 0))

    transformed
}
```

---

#### Example 4: Fisheye Warp

```morphogen
scene FisheyeEffect {
    let image = field.load("input.png")

    # Define radial warp (fisheye)
    let warp = λ (x, y): {
        let center = (image.width / 2, image.height / 2)
        let dx = x - center.x
        let dy = y - center.y
        let r = sqrt(dx² + dy²)
        let factor = 1.0 + 0.0005 * r²  # Quadratic distortion

        (center.x + dx / factor, center.y + dy / factor)
    }

    # Apply warp
    let fisheye = transform.reparam(image, mapping=warp)

    out visual = fisheye
}
```

---

#### Example 5: Frame Conversion (3D CAD)

```morphogen
assembly RobotArm {
    # Define frames for each link
    let base_frame = Frame<3, Cartesian, mm>(origin=(0,0,0))

    let link1_frame = frame.transform(
        base_frame,
        operations = [
            (translate, offset=(0, 0, 50mm)),
            (rotate, angle=30 deg, axis="z")
        ]
    )

    let link2_frame = frame.transform(
        link1_frame,
        operations = [
            (translate, offset=(0, 0, 100mm)),
            (rotate, angle=45 deg, axis="y")
        ]
    )

    # Place parts in their respective frames
    let link1 = geom.cylinder(radius=5mm, height=100mm)
        |> object.set_frame(link1_frame)

    let link2 = geom.cylinder(radius=3mm, height=80mm)
        |> object.set_frame(link2_frame)

    [link1, link2]
}
```

---

## 8. Implementation Roadmap

### Phase 1 (v0.4.0) — ✅ Completed
- FFT/IFFT, STFT/ISTFT
- Window functions
- Time-frequency transforms

### Phase 2 (v0.5.0) — Planned
- DCT/IDCT
- Wavelet transforms
- Mel-scale warping

### Phase 3 (v0.6.0) — Planned
- FFT2D/IFFT2D
- Graph spectral transforms

### Phase 4 (v0.9.0) — Geometry Integration
- **Affine transforms** (translate, rotate, scale, mirror)
- **Coordinate conversions** (Cartesian ↔ polar ↔ spherical ↔ cylindrical)
- **Frame-aware transforms** (integration with coordinate-frames.md)
- **Projective transforms** (perspective, homography)
- **Warping** (extend `reparam` for spatial fields)

---

## Summary

The Transform Dialect provides:

✅ **Uniform grammar** for all domain changes
✅ **First-class transforms** (not library calls)
✅ **Profile-driven determinism** (strict/repro/live)
✅ **Composable operations** (chain transforms, mix domains)
✅ **Extensible** (new domains/methods via registry)
✅ **Spatial transforms** (affine, coordinate conversions, projective)
✅ **Explicit origins** (TiaCAD principle: no implicit rotation/scale centers)
✅ **Frame integration** (works with coordinate-frames.md)

Transforms are the bridge between Morphogen's multi-domain vision and practical computation.

---

## References

### Morphogen Documentation
- [Mathematical Transformation Metaphors](../reference/math-transformation-metaphors.md) — Pedagogical frameworks and intuitive understanding
- [Advanced Visualizations](../reference/advanced-visualizations.md) — Visualization techniques for transforms
- **coordinate-frames.md** — Frame/anchor system
- **geometry.md** — Geometric operators
- **operator-registry.md** — Layer 2 (transforms)

### External References
- **TiaCAD v3.x** — Deterministic transform model, explicit origins
