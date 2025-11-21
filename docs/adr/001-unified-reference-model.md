# Architecture Decision Record — Unified Reference & Frame Model in Morphogen

**Date:** 2025-11
**Status:** 🚀 APPROVED – Begin Implementation
**Decision:** Morphogen will adopt a unified domain-agnostic reference system, auto-generated anchors, and first-class frames, replacing any prior ad-hoc reference semantics.

---

## 1. Context

During early Morphogen design (kernel, Graph IR, operator registry), domains such as:

- audio
- fields
- visuals
- transforms
- agents
- physics
- geometry (future)
- simulations

were allowed to define their own reference semantics implicitly — e.g., beat positions, sample indices, field coordinates, keyframes, spatial markers, etc.

This creates **fragmentation**:

- different DSLs compute "anchors" differently
- cross-domain transforms don't line up
- operators require awkward domain-specific metadata
- scheduling and determinism become harder
- backends cannot reliably optimize repositioning or offsets
- debugging becomes domain-specific and irregular

**TiaCAD v3.0 revealed a much cleaner approach:**
One unified reference object (`SpatialRef`) with orientation-aware frames, auto-anchors, and pure-function resolution.

This model is **domain-agnostic** and solves several problems Morphogen has not yet solved.

---

## 2. Decision

Morphogen adopts a **unified, typed, pure, deterministic reference system** across all domains.

The core components:

### 1. `Ref` — a domain-agnostic reference object

```rust
Ref {
    domain: "audio" | "field" | "visual" | "agent" | "geom" | ...
    type: "point" | "frame" | "interval" | "axis" | "event" | ...
    payload:   # domain-specific semantic data
    frame:     # optional, but recommended for most domains
}
```

### 2. `Frame` — first-class orientation & basis object

```rust
Frame {
    origin : vec3 or vecN   (domain-dependent)
    x_axis : vec3 (optional)
    y_axis : vec3 (optional)
    z_axis : vec3 (optional)
    metadata: deterministic profile information
}
```

`Frame` generalizes TiaCAD's orientation system and extends it into Morphogen's multi-dimensional domains.

### 3. Auto-generated anchors for every domain

Every `Stream<T,D,R>` / `Field<T>` / `Evt<A>` / visual clip / agent / geometry primitive **automatically provides canonical references**, such as:

**Audio**
- `.onsets`
- `.peaks`
- `.zero_crossings`
- `.rms_window(N)`
- `.fft_bin(i).center`

**Fields & PDEs**
- `.center`
- `.boundary.xmin`, `.boundary.ymax`, …
- `.grad_axis_x`
- `.cell(i,j).frame`

**Visuals**
- `.focal_center`
- `.bounding_box`
- `.timeline_start`

**Agents / Physics**
- `.com`
- `.forward_axis`
- `.sensor_frame('left_eye')`

**Geometry (future Morphogen extension)**
- `.face_top`
- `.edge(i)`
- `.axis_z`

### 4. All offsets are interpreted in local frames, not global coordinates

This mirrors TiaCAD's "offsets follow the face orientation" rule and ensures **cross-domain physical correctness**.

### 5. All resolvers are pure functions

Given:
- Graph IR
- Operator Registry metadata
- Domain inputs

They produce **deterministic** `Ref` and `Frame` values with **no side effects**.

---

## 3. Consequences

### ✔️ Positive

#### 1. Cross-domain consistency

Transformations between time domains, spaces, coordinate systems, frequency representations, or simulation grids become **predictable and systematic**.

#### 2. Stronger operator metadata

Operators can now say:

```yaml
requires: Frame
produces: Frame
aligns_to: Ref
```

which unifies:
- convolution windows
- FFT bin alignment
- k-space ↔ real-space
- geometric frame sampling
- agent orientation transforms
- animation timeline anchoring

#### 3. Graph IR becomes much richer and simpler

No more special casing:
- sample positions
- axis-aligned offsets
- domain-dependent coordinate hacks

Everything flows through a **small, elegant `Ref` + `Frame` vocabulary**.

#### 4. Powerful DSL ergonomics

Morphogen.Audio and RiffStack become dramatically clearer:

```morphogen
# align filter cutoff to spectral centroid
cutoff = transform(spec.centroid).to("Hz")

# field sampling on local gradient frame
sample = field.sample_along(field.grad_frame, offset=[1,0,0])
```

#### 5. Determinism gets stronger

`Frames` + pure resolution → **reproducible pipelines**.

#### 6. Backend optimization becomes easier

Low-level backends (GPU, CPU, FFT, PDE) can optimize further because:
- frames expose coordinate semantics
- references expose alignment and boundaries
- offsets become physically meaningful

---

### ❌ Negative

#### 1. Breaking change

Earlier experimental code assumed domain-specific reference behaviors.

#### 2. Requires kernel changes

Graph IR schema must include:

```yaml
ref: {...}
frame: {...}
```

#### 3. Frontends must update

- Morphogen.Audio must understand `.onsets`, `.beats`, `.frame`, etc.
- RiffStack must map controls to canonical references.

#### 4. Operator registry requires rework

Certain categories need updated types:
- `Ctl` may become `Ref<Ctl>` or `Frame<Ctl>`
- transforms now emit `Frame` metadata

---

## 4. Technical Changes

### 4.1 New Core Types

```rust
Ref { domain, type, payload, frame }
Frame { origin, axes..., metadata }
```

**Integration into Morphogen Kernel**

- integrated into Types & Units
- stored in Graph IR nodes
- validated during kernel introspection
- used by the multi-rate scheduler to align events
- used by transform dialect to encode coordinate conversions

---

## 5. Implementation Plan

### Phase 1: Core API (2 weeks)

- Implement `Ref` and `Frame` kernel types
- Update Graph IR schema
- Add pure-function ref resolution
- Add determinism metadata

### Phase 2: Auto-anchors (3 weeks)

- Per-domain canonical anchors (audio, fields, visual, events)
- Operator registry updates
- DSL sugar for anchor access

### Phase 3: Frame-aware transforms (2–4 weeks)

- space→k-space with frame alignment
- convolution kernel orientation
- sample-accurate event alignment
- field/mesh sampling with local frames
- visual transform alignment

### Phase 4: Frontend updates (3 weeks)

- Morphogen.Audio: anchors and frame-aware transforms
- RiffStack: control anchoring, event alignment
- Diagnostics for anchors & frames

---

## 6. Success Criteria

- ✅ Deterministic frame semantics across domains
- ✅ Auto-anchors produce stable results across all profiles
- ✅ Graph IR and Operator Registry updated cleanly
- ✅ No ad-hoc reference logic left in any domain
- ✅ DSLs become more expressive with fewer operators
- ✅ Backends validate frame alignment and optimize correctly
- ✅ Golden tests include frame-based introspection

---

## 7. Final Assessment

This ADR is a **foundational improvement**.
It gives Morphogen:

- cleaner math
- more expressive semantics
- simpler IR
- better determinism
- stronger cross-domain coupling
- future-proof multi-modal capabilities

**This is not domain creep —**
**this is sharpening the core kernel's semantics so every domain becomes easier to build.**

---

## References

- **TiaCAD v3.x** — Unified `SpatialRef` system with orientation-aware frames
- **../specifications/coordinate-frames.md** — Morphogen frame/anchor specification
- **../specifications/transform.md** — Frame-aware transformations
- **../specifications/geometry.md** — Geometry domain with auto-generated anchors
- **../architecture/domain-architecture.md** — Cross-domain reference model
