# SPEC: Coordinate Frames and Anchors

**Version:** 1.0 Draft
**Status:** RFC
**Last Updated:** 2025-11-15
**Dependencies:** type-system.md, transform.md, operator-registry.md

---

## Overview

This specification defines **coordinate frames** and **anchors** as first-class concepts in Morphogen, providing a unified system for spatial, temporal, and structural reference across all domains.

**Key insight from TiaCAD:** Reference-based composition (via anchors) is more compositional, declarative, and robust than hierarchical nesting. This pattern applies far beyond geometry.

---

## Motivation

Across domains, Morphogen programs need to:

1. **Reference locations** — in space, time, signal chains, or abstract graphs
2. **Compose transformations** — rotations, translations, warps, with clear origins
3. **Express intent declaratively** — "align these two points" rather than "translate by (compute offset)"
4. **Maintain determinism** — explicit frames prevent hidden state and frame-of-reference bugs

**Problem:** Current Morphogen has implicit coordinate systems. Fields assume global grids. Transforms lack explicit origins. Composition is manual.

**Solution:** Introduce frames and anchors as typed, first-class objects that work uniformly across geometry, audio, physics, agents, fields, and visuals.

---

## Core Concepts

### 1. Coordinate Frame

A **coordinate frame** is a local coordinate system defined by:

- **Origin** — reference point (position in parent frame)
- **Basis** — orientation (rotation/axes)
- **Scale** — metric (units/spacing)
- **Type** — coordinate system kind (Cartesian, polar, spherical, etc.)

**Properties:**
- Frames can be hierarchical (child frame defined relative to parent)
- Frames are immutable (transforming a frame creates a new frame)
- Frames carry metadata (bounds, centering, units)

**Type signature:**
```morphogen
Frame<Dim, CoordType, Units>
  Dim: 1 | 2 | 3 | Time | Frequency | ...
  CoordType: Cartesian | Polar | Spherical | Cylindrical | ...
  Units: m | s | Hz | rad | ...
```

**Examples:**
```morphogen
# 2D Cartesian frame with meter units
let world_frame = Frame<2, Cartesian, m>(
    origin = (0.0, 0.0),
    basis = ((1, 0), (0, 1)),
    scale = 1.0
)

# Polar frame centered at (10, 5) in world frame
let polar_frame = Frame<2, Polar, (m, rad)>(
    parent = world_frame,
    origin = (10.0, 5.0)
)

# Temporal frame (audio/signal)
let beat_frame = Frame<Time, Beat, bpm>(
    tempo = 120 bpm,
    offset = 0.0 s
)
```

---

### 2. Anchor

An **anchor** is a named reference point within an object or field, defined in a local coordinate frame.

**Anchor types:**

| Category | Examples |
|----------|----------|
| **Geometry** | `.center`, `.face_top`, `.edge_left`, `.corner_nw`, `.axis_z` |
| **Audio** | `.onset`, `.beat`, `.downbeat`, `.peak`, `.zero_crossing` |
| **Physics** | `.center_of_mass`, `.joint`, `.contact_point`, `.axis_rotation` |
| **Agents** | `.sensor_front`, `.waypoint`, `.goal`, `.spawn_point` |
| **Fields** | `.sample_point`, `.gradient_max`, `.boundary_north` |
| **Visuals** | `.pivot`, `.camera`, `.light_source`, `.layer_origin` |

**Type signature:**
```morphogen
Anchor<Frame, T>
  Frame: Coordinate frame the anchor is defined in
  T: Type of anchored object (Mesh, Signal, Body, Agent, Field, ...)
```

**Properties:**
- Anchors provide **position** (point in frame)
- Anchors may provide **orientation** (local axes for 2D/3D objects)
- Anchors are **auto-generated** from object geometry/structure
- Anchors are **named** and **queryable**

**Examples:**
```morphogen
# Geometric anchors (auto-generated for meshes)
let box = mesh.box(width=10, height=5, depth=3)
let top_face = box.anchor("face_top")      # Returns Anchor<3, Mesh>
let center = box.anchor("center")          # Center of bounding box

# Audio anchors (auto-generated from analysis)
let kick = audio.load("kick.wav")
let onset = kick.anchor("onset")           # First significant onset
let peak = kick.anchor("peak")             # Maximum amplitude point

# Physics anchors (defined by simulation)
let body = physics.rigid_body(shape=box)
let com = body.anchor("center_of_mass")
let joint_top = body.anchor("joint", location="top")

# Agent anchors (user-defined + auto-generated)
let robot = agent.create(model="wheeled")
robot.define_anchor("lidar", position=(0, 0.5, 1.0))
let sensor = robot.anchor("lidar")
```

---

### 3. Reference-Based Composition

**Traditional hierarchical composition** (problematic):
```morphogen
# Implicit parent-child nesting
scene.add_child(part_A)
part_A.add_child(part_B)  # B's position depends on A's transform
part_A.rotate(45)          # Implicitly affects B (hidden coupling)
```

**Problems:**
- Hidden state (child position depends on parent's mutable state)
- Non-declarative (must specify operations, not intent)
- Hard to refactor (changing hierarchy breaks everything)

**Reference-based composition** (TiaCAD model):
```morphogen
# Flat object registry + explicit references
let part_A = mesh.box(...)
let part_B = mesh.cylinder(...)

# Declarative placement via anchor mapping
let assembly = mesh.place(
    part_B,
    anchor = part_B.anchor("base"),     # Bottom of cylinder
    at = part_A.anchor("face_top")      # Top face of box
)
```

**Benefits:**
- Explicit dependencies (no hidden state)
- Declarative intent ("align these anchors")
- Easy refactoring (change part_A's definition without breaking placement)
- Compositional (placement is a pure function)

---

### 4. Deterministic Transformations

All transformations must specify:

1. **Explicit origin** — rotation/scale center (no implicit frame)
2. **Pure functions** — input frame → output frame (no mutation)
3. **Ordered composition** — transform chains are explicit sequences

**Transformation types:**

| Transform | Parameters | Determinism |
|-----------|-----------|-------------|
| Translation | `offset: Vec<Dim>` | Strict |
| Rotation | `angle: Angle, axis: Vec3, origin: Anchor` | Strict |
| Scale | `factor: f64, origin: Anchor` | Strict |
| Affine | `matrix: Mat<Dim+1>` | Strict |
| Warp | `mapping: Field<Dim, Dim>` | Repro (if field is repro) |

**Examples:**
```morphogen
# Rotation with explicit origin
let rotated = transform.affine(
    mesh,
    rotation = 45 deg,
    axis = "z",
    origin = mesh.anchor("center")  # Explicit: rotate around center
)

# Chained transforms (order matters!)
let transformed = mesh
    |> transform.translate(offset=(10, 0, 0))
    |> transform.rotate(angle=90 deg, axis="y", origin=.center)
    |> transform.scale(factor=2.0, origin=.center)

# Create new frame from transform
let new_frame = frame.transform(
    old_frame,
    operations = [
        (translate, offset=(5, 5)),
        (rotate, angle=30 deg)
    ]
)
```

**Contrast with implicit transforms:**
```morphogen
# ❌ Implicit origin (ambiguous!)
let rotated = transform.rotate(mesh, 45 deg)  # Rotate around what?

# ✅ Explicit origin (clear!)
let rotated = transform.rotate(
    mesh,
    angle = 45 deg,
    origin = mesh.anchor("center")
)
```

---

## Type System Integration

### Frame Types

```morphogen
# Generic frame
Frame<Dim, CoordType, Units>

# Concrete examples
Frame<2, Cartesian, m>            # 2D Cartesian in meters
Frame<3, Spherical, (m, rad, rad)> # 3D spherical
Frame<Time, Beat, bpm>             # Temporal frame (music)
Frame<Frequency, Log, Hz>          # Log-frequency frame (audio)
```

### Anchor Types

```morphogen
# Generic anchor
Anchor<Frame, T>

# Concrete examples
Anchor<Frame<3, Cartesian, m>, Mesh>     # 3D geometric anchor
Anchor<Frame<Time, Sample, s>, Signal>   # Temporal signal anchor
Anchor<Frame<2, Cartesian, m>, Agent>    # 2D agent position anchor
```

### Transform Types

```morphogen
# Transform between frames
Transform<Frame_A, Frame_B>

# Affine transform (preserves dimension)
Affine<Dim> <: Transform<Frame<Dim, _, _>, Frame<Dim, _, _>>

# Coordinate system conversion
CoordConversion<CoordType_A, CoordType_B>
  <: Transform<Frame<Dim, CoordType_A, _>, Frame<Dim, CoordType_B, _>>
```

---

## Operator Registry Extension

### Layer 2: Transform Operators (Extended)

#### Frame Management

```morphogen
frame.create(
    dim: Int,
    coord_type: CoordType,
    origin: Vec<Dim> = (0, ...),
    basis: Mat<Dim, Dim> = I,
    scale: f64 = 1.0,
    units: Units = dimensionless
) -> Frame<Dim, CoordType, Units>
```

```morphogen
frame.transform(
    frame: Frame<Dim, CT, U>,
    operations: [(TransformOp, Params)]
) -> Frame<Dim, CT, U>
```

```morphogen
frame.to_parent(
    child_frame: Frame<Dim, CT, U>,
    point: Vec<Dim>
) -> Vec<Dim>
# Convert point from child frame to parent frame coordinates
```

#### Anchor Operations

```morphogen
anchor.create(
    name: String,
    object: T,
    position: Vec<Dim>,
    orientation: Mat<Dim, Dim> = I  # For 2D/3D anchors
) -> Anchor<Frame<Dim, _, _>, T>
```

```morphogen
anchor.resolve(
    object: T,
    name: String | Pattern
) -> Anchor<Frame, T>
# Query anchor by name (e.g., "face_top", ">Z" for highest Z face)
```

```morphogen
anchor.position(anchor: Anchor<F, T>) -> Vec<Dim>
anchor.orientation(anchor: Anchor<F, T>) -> Mat<Dim, Dim>
anchor.frame(anchor: Anchor<F, T>) -> F
```

#### Placement & Alignment

```morphogen
object.place(
    object: T,
    anchor: Anchor<F, T>,
    at: Anchor<F, U> | Vec<Dim>,
    align_orientation: bool = true
) -> T
# Place object such that its anchor coincides with target
```

```morphogen
object.align(
    objects: [T],
    anchors: [Anchor<F, T>],
    axis: Vec<Dim>
) -> [T]
# Align multiple objects along an axis using specified anchors
```

#### Coordinate Conversions

```morphogen
transform.to_coord(
    field: Field<T, Frame<Dim, CT_A, U>>,
    coord_type: CT_B
) -> Field<T, Frame<Dim, CT_B, U>>
# Convert field to different coordinate system
# Example: Cartesian -> Polar
```

**Examples:**
```morphogen
# Cartesian to polar
let polar_field = transform.to_coord(
    cartesian_field,
    coord_type = Polar
)

# Polar to spherical (3D)
let spherical = transform.to_coord(
    polar_field_3d,
    coord_type = Spherical
)
```

---

## Cross-Domain Applications

### 1. Geometry (Meshes, CAD)

**Use case:** Assemble parts declaratively

```morphogen
let base = mesh.box(width=10, height=2, depth=10)
let column = mesh.cylinder(radius=1, height=8)
let top = mesh.cone(radius=2, height=3)

# Stack using anchors
let tower = [
    base,
    column |> place(anchor=.bottom, at=base.anchor("face_top")),
    top    |> place(anchor=.bottom, at=column.anchor("face_top"))
]
```

**Anchors generated:**
- `.center`, `.face_{top,bottom,left,right,front,back}`
- `.corner_{nwt,net,swt,set,nwb,neb,swb,seb}` (8 corners for box)
- `.edge_{...}` (12 edges for box)
- `.axis_{x,y,z}` (local axes)

---

### 2. Audio (Signals, Events)

**Use case:** Align beats, onsets, or markers

```morphogen
let kick = audio.load("kick.wav")
let snare = audio.load("snare.wav")

# Align snare onset to kick's second beat
let aligned = signal.place(
    snare,
    anchor = snare.anchor("onset"),
    at = kick.anchor("beat", index=1)  # Second beat
)
```

**Anchors generated:**
- `.onset` — first significant transient
- `.beat` — beat grid (if tempo known)
- `.peak` — maximum amplitude
- `.zero_crossing` — specific zero crossings
- `.marker_{name}` — user-defined markers

---

### 3. Physics (Rigid Bodies, Constraints)

**Use case:** Define joints between bodies

```morphogen
let body_A = physics.rigid_body(shape=box_A)
let body_B = physics.rigid_body(shape=box_B)

# Hinge joint at specific anchor
let joint = physics.hinge_joint(
    body_A = body_A,
    body_B = body_B,
    anchor_A = body_A.anchor("edge_top_left"),
    anchor_B = body_B.anchor("edge_bottom_left"),
    axis = (0, 1, 0)  # Rotate around Y
)
```

**Anchors generated:**
- `.center_of_mass` — COM (may differ from geometric center)
- `.joint_{name}` — predefined joint locations
- `.contact_{...}` — collision contact points (dynamic)

---

### 4. Agents (Robotics, Simulation)

**Use case:** Sensor placement, waypoints

```morphogen
let robot = agent.create(model="quadrotor")

# Define sensor anchors
robot.define_anchor("camera", position=(0.1, 0, -0.05), orientation=...)
robot.define_anchor("lidar", position=(0, 0, 0.1))

# Query sensor frame for rendering
let camera_frame = robot.anchor("camera").frame()
```

**Anchors generated/defined:**
- `.spawn_point` — initial position
- `.goal` — target position
- `.waypoint_{i}` — path waypoints
- `.sensor_{name}` — user-defined sensors
- `.actuator_{name}` — actuator attachment points

---

### 5. Fields (Grids, PDEs)

**Use case:** Sample points, boundary conditions

```morphogen
let temperature_field = field.zeros(shape=(100, 100))

# Define boundary anchors
let north_boundary = temperature_field.anchor("boundary_north")
let south_boundary = temperature_field.anchor("boundary_south")

# Apply boundary conditions
temperature_field = field.set(
    temperature_field,
    at = north_boundary,
    value = 100.0  # Hot boundary
)
```

**Anchors generated:**
- `.boundary_{north,south,east,west,top,bottom}` — grid boundaries
- `.center` — field center
- `.sample_{i,j,k}` — specific grid points
- `.gradient_max` — location of maximum gradient

---

### 6. Visuals (Rendering, Layers)

**Use case:** Pivot points, camera frames

```morphogen
let sprite = visual.load("character.png")

# Rotate around custom pivot (not top-left corner)
let rotated = transform.rotate(
    sprite,
    angle = 45 deg,
    origin = sprite.anchor("pivot")  # Custom-defined pivot
)

# Render from camera frame
let rendered = visual.render(
    scene,
    camera_frame = camera.anchor("eye").frame()
)
```

**Anchors generated/defined:**
- `.pivot` — rotation/scale origin (user-defined)
- `.layer_origin` — layer coordinate system origin
- `.camera` — camera position + look direction
- `.light_{name}` — light source positions

---

## Implementation Considerations

### 1. Anchor Auto-Generation

**Geometry:**
- Use bounding box for `.center`, `.corner_{...}`
- Use mesh topology for `.face_{...}`, `.edge_{...}`
- Use principal axes for `.axis_{...}`

**Audio:**
- Use onset detection for `.onset`, `.beat`
- Use peak finding for `.peak`
- Use tempo analysis for `.beat` grid

**Physics:**
- Use mass distribution for `.center_of_mass`
- Use shape analysis for `.joint` suggestions

**Fields:**
- Use grid metadata for `.boundary_{...}`
- Use gradient analysis for `.gradient_max`

### 2. Anchor Queries

Support flexible anchor queries:

```morphogen
# Direct name lookup
box.anchor("face_top")

# Pattern-based query (highest Z face)
box.anchor(">Z")

# Indexed anchors
kick.anchor("beat", index=2)  # Third beat (0-indexed)

# Filtered anchors
mesh.anchor("face", filter=λ f: f.normal.z > 0.9)
```

### 3. Frame Hierarchies

Frames form a directed acyclic graph (DAG):

```
world_frame
  ├─ object_A_frame
  │   ├─ anchor_1_frame
  │   └─ anchor_2_frame
  └─ object_B_frame
      └─ anchor_3_frame
```

**Conversions:**
- `frame.to_parent(child_frame, point)` — convert point to parent coords
- `frame.to_world(frame, point)` — convert point to world coords
- `frame.to_frame(point, from_frame, to_frame)` — arbitrary conversion

### 4. Determinism Profiles

All frame/anchor operations respect determinism profiles:

| Operation | Profile |
|-----------|---------|
| `frame.create` | Strict |
| `frame.transform` | Strict (if transforms are strict) |
| `anchor.resolve` | Strict |
| `object.place` | Strict |
| `transform.to_coord` | Repro (involves interpolation) |

### 5. MLIR Lowering

**Morphogen IR (frontend):**
```morphogen
let placed = mesh.place(
    part,
    anchor = part.anchor("base"),
    at = target.anchor("top")
)
```

**Graph IR (kernel boundary):**
```
%anchor_a = anchor.resolve %part, "base"
%anchor_b = anchor.resolve %target, "top"
%transform = anchor.compute_transform %anchor_a, %anchor_b
%placed = mesh.apply_transform %part, %transform
```

**MLIR (backend):**
```mlir
%pos_a = anchor.get_position %anchor_a : !anchor<f64, 3>
%pos_b = anchor.get_position %anchor_b : !anchor<f64, 3>
%offset = arith.subf %pos_b, %pos_a : vector<3xf64>
%transform = transform.translation %offset : !transform.affine<3>
%placed = mesh.apply_transform %part, %transform : !mesh.solid -> !mesh.solid
```

---

## Integration with Existing Morphogen Dialects

### Transform Dialect

**Existing:** `transform.reparam(field, mapping)`

**Extended:**
- `transform.reparam` becomes frame-aware
- Coordinate conversions use frames
- Explicit origin for all affine transforms

**Example:**
```morphogen
# Old (implicit)
let rotated = transform.rotate(mesh, 45 deg)  # Rotate around origin?

# New (explicit)
let rotated = transform.rotate(
    mesh,
    angle = 45 deg,
    origin = mesh.anchor("center")  # Explicit
)
```

### Field Dialect

**Existing:** `Field<T, Grid>`

**Extended:** `Field<T, Frame>`

Fields now carry explicit coordinate frames:

```morphogen
# Field with Cartesian frame
let cartesian_field: Field<f64, Frame<2, Cartesian, m>> = field.zeros(
    shape = (100, 100),
    frame = Frame<2, Cartesian, m>(bounds=((0, 10), (0, 10)))
)

# Convert to polar coordinates
let polar_field: Field<f64, Frame<2, Polar, (m, rad)>> = transform.to_coord(
    cartesian_field,
    coord_type = Polar
)
```

### Type System

Add frame/anchor types:

```morphogen
# New primitive types
Frame<Dim, CoordType, Units>
Anchor<Frame, T>
Transform<Frame_A, Frame_B>

# Units integration
Frame<2, Cartesian, m>       # Length units
Frame<Time, Sample, s>       # Time units
Frame<Frequency, Log, Hz>    # Frequency units
```

---

## Examples

### Example 1: CAD Assembly (Geometry)

```morphogen
# Define parts
let base_plate = mesh.box(width=20, depth=20, height=2)
let pillar = mesh.cylinder(radius=1.5, height=10)
let top_sphere = mesh.sphere(radius=3)

# Assemble using anchors
let assembly = [
    base_plate,

    # Place pillar on top of base, centered
    pillar |> mesh.place(
        anchor = pillar.anchor("bottom"),
        at = base_plate.anchor("face_top"),
        align_orientation = true
    ),

    # Place sphere on top of pillar
    top_sphere |> mesh.place(
        anchor = top_sphere.anchor("bottom"),
        at = pillar.anchor("face_top")
    )
]

# Rotate entire assembly around base center
let rotated_assembly = assembly |> map(λ part:
    transform.rotate(
        part,
        angle = 45 deg,
        axis = "z",
        origin = base_plate.anchor("center")
    )
)
```

### Example 2: Beat-Aligned Audio (Audio)

```morphogen
# Load samples
let kick = audio.load("kick.wav")
let snare = audio.load("snare.wav")
let hihat = audio.load("hihat.wav")

# Create temporal frame (120 BPM)
let beat_frame = Frame<Time, Beat, bpm>(tempo=120, offset=0.0)

# Align samples to beat grid
let pattern = [
    kick  |> signal.place(anchor=.onset, at=beat_frame.anchor("beat", index=0)),
    snare |> signal.place(anchor=.onset, at=beat_frame.anchor("beat", index=4)),
    hihat |> signal.place(anchor=.onset, at=beat_frame.anchor("beat", index=2)),
    hihat |> signal.place(anchor=.onset, at=beat_frame.anchor("beat", index=6))
]

# Mix to single signal
let mixed = signal.mix(pattern)
```

### Example 3: Physics Joint (Physics)

```morphogen
# Create rigid bodies
let body_A = physics.rigid_body(
    shape = mesh.box(width=5, height=2, depth=3),
    mass = 10.0
)

let body_B = physics.rigid_body(
    shape = mesh.box(width=3, height=3, depth=3),
    mass = 5.0
)

# Define hinge joint using anchors
let hinge = physics.hinge_joint(
    body_A = body_A,
    body_B = body_B,
    anchor_A = body_A.anchor("edge_top_right"),
    anchor_B = body_B.anchor("edge_bottom_left"),
    axis = (0, 1, 0),  # Rotate around Y axis
    limits = (-90 deg, 90 deg)
)

# Simulate
let sim = physics.simulate(
    bodies = [body_A, body_B],
    constraints = [hinge],
    duration = 10.0 s,
    dt = 0.01 s
)
```

### Example 4: Agent Sensors (Agents)

```morphogen
# Create agent
let drone = agent.create(model="quadrotor")

# Define sensor anchors
drone.define_anchor("camera_down",
    position = (0, 0, -0.1),
    orientation = look_down()
)

drone.define_anchor("lidar",
    position = (0, 0, 0.05)
)

# Place agent at waypoint
let positioned_drone = agent.place(
    drone,
    anchor = drone.anchor("center"),
    at = waypoint_1
)

# Render from camera perspective
let camera_view = visual.render(
    scene,
    camera_frame = positioned_drone.anchor("camera_down").frame()
)
```

### Example 5: Field Boundary Conditions (Fields)

```morphogen
# Create temperature field
let temp = field.zeros(shape=(100, 100), frame=cartesian_2d)

# Apply boundary conditions using anchors
temp = temp
    |> field.set(at=.anchor("boundary_north"), value=100.0)  # Hot top
    |> field.set(at=.anchor("boundary_south"), value=0.0)    # Cold bottom
    |> field.set(at=.anchor("boundary_east"),  value=50.0)   # Warm right
    |> field.set(at=.anchor("boundary_west"),  value=50.0)   # Warm left

# Solve Laplace equation
let steady_state = field.solve_laplace(temp)

# Sample at center
let center_temp = field.sample(steady_state, at=.anchor("center"))
```

---

## Testing Strategy

### 1. Core Frame/Anchor Tests

- Frame creation with different coordinate types
- Frame transformations (determinism)
- Frame hierarchy conversions (to_parent, to_world)
- Anchor resolution (by name, by pattern)
- Anchor position/orientation queries

### 2. Cross-Domain Tests

- Geometry: mesh placement, alignment
- Audio: beat alignment, onset detection
- Physics: joint creation, COM calculation
- Agents: sensor frame queries
- Fields: boundary anchor resolution

### 3. Determinism Tests

All operations must pass golden tests:

```morphogen
# Frame creation is deterministic
assert_eq!(
    frame.create(dim=2, coord_type=Cartesian),
    frame.create(dim=2, coord_type=Cartesian)
)

# Anchor resolution is deterministic
let box = mesh.box(width=10, height=5, depth=3)
assert_eq!(
    box.anchor("face_top"),
    box.anchor("face_top")
)

# Placement is deterministic
let placed_1 = mesh.place(part, anchor=.bottom, at=target.top)
let placed_2 = mesh.place(part, anchor=.bottom, at=target.top)
assert_mesh_eq!(placed_1, placed_2, tolerance=1e-12)
```

### 4. Transform Chain Tests

Verify composition is explicit and ordered:

```morphogen
# Order matters
let A = mesh |> translate(...) |> rotate(...)
let B = mesh |> rotate(...) |> translate(...)
assert_ne!(A, B)  # Different results (rotation origin differs)

# Explicit origin consistency
let rotated = transform.rotate(mesh, 45 deg, origin=.center)
assert_position_eq!(
    rotated.anchor("center").position(),
    mesh.anchor("center").position()
)  # Center didn't move
```

---

## Future Extensions

### 1. Anchor Arithmetic

```morphogen
# Midpoint between two anchors
let mid = anchor.midpoint(anchor_A, anchor_B)

# Offset anchor
let offset = anchor.offset(anchor, delta=(1, 0, 0))
```

### 2. Constraint-Based Placement

```morphogen
# Solve for positions satisfying constraints
let positioned = constraint.solve([
    distance(part_A.anchor("center"), part_B.anchor("center")) == 10.0,
    aligned(part_A.anchor("axis_z"), part_B.anchor("axis_z")),
    part_C.anchor("base").z == 0.0
])
```

### 3. Dynamic Anchors

```morphogen
# Anchor that tracks moving object
let trajectory = physics.simulate(body)
let moving_anchor = anchor.dynamic(
    trajectory,
    name = "center_of_mass",
    time_dependent = true
)
```

### 4. Anchor Visualization (Debug)

```morphogen
# Render anchors for debugging
let debug_view = visual.render(
    mesh,
    show_anchors = true,
    anchor_scale = 0.1
)
```

---

## References

- **TiaCAD v3.x** — Reference/anchor model for parametric CAD
- transform.md — Transform dialect (`reparam`, coordinate conversions)
- type-system.md — Units and type system
- operator-registry.md — Operator organization
- ../architecture/domain-architecture.md — Cross-domain vision

---

## Summary

**Coordinate frames and anchors** provide:

1. **Unified reference system** — works across geometry, audio, physics, agents, fields, visuals
2. **Declarative composition** — express intent, not operations
3. **Deterministic transforms** — explicit origins, pure functions, no hidden state
4. **Type safety** — frames/anchors are typed, preventing frame-of-reference bugs
5. **Better UX** — users think in terms of "align these points", not "compute offset and apply"

This system is inspired by TiaCAD's reference-based composition model, generalized to all Morphogen domains.

**Key insight:** Anchors unify positional (geometry), temporal (audio), structural (graphs), and abstract (latent spaces) references into a single coherent model.

---

**Status:** RFC — ready for review and implementation planning
**Next steps:** Prototype in geometry domain, extend to audio/physics, finalize MLIR lowering
