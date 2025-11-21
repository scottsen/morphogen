# 📐 Morphogen.Geometry Specification v1.0

**A declarative geometry and mesh processing dialect built on the Morphogen kernel.**

**Inspired by TiaCAD's reference-based composition model.**

---

## 0. Overview

Morphogen.Geometry is a typed, declarative geometry computation dialect layered on the Morphogen kernel.
It provides deterministic semantics, parametric modeling primitives, and composable spatial constructs.
It is an intermediate layer that sits between:

- **User applications** — CAD tools, 3D printing, computational geometry, simulation preprocessing
- **Morphogen Core** — the deterministic MLIR-based execution kernel
- **Backend engines** — CadQuery, CGAL, OpenCASCADE, or custom GPU mesh kernels

---

## 1. Language Philosophy

| Principle | Meaning |
|-----------|---------|
| **Reference-based composition** | Objects are placed via anchors, not nested hierarchies. |
| **Deterministic transforms** | Explicit origins, pure functions, no hidden state. |
| **Typed shapes** | Every geometric object has a type (Solid, Shell, Face, Edge, Vertex). |
| **Declarative modeling** | Describe what to build, not how to build it step-by-step. |
| **Backend-neutral** | Operations defined semantically; lowering varies by backend. |
| **Cross-domain integration** | Geometry integrates with Fields (CFD), Physics (collision), Visuals (rendering). |

**Key insight from TiaCAD:** Anchors replace hierarchical assemblies, making composition more robust and declarative.

---

## 2. Core Types

All geometry types are defined in the kernel's type system with explicit dimensionality and topology.

| Type | Description | Dimensionality | Examples |
|------|-------------|----------------|----------|
| `Solid` | Closed 3D volume | 3D | Box, sphere, extruded polygon |
| `Shell` | Open 3D surface | 2.5D | Mesh, NURBS surface |
| `Face` | Bounded 2D surface in 3D | 2D in 3D | Rectangle, circle, polygon |
| `Wire` | Connected 1D curve in 3D | 1D in 3D | Polyline, spline, arc |
| `Edge` | Single 1D curve segment | 1D | Line segment, arc, Bezier |
| `Vertex` | 0D point in 3D | 0D | (x, y, z) |
| `Sketch` | 2D planar construction | 2D | Rectangle, circle, polygon (pre-extrusion) |
| `Mesh<T>` | Discrete mesh with vertex data | 3D or 2D | Triangle/quad mesh, point cloud |
| `Frame<3>` | 3D coordinate frame | Meta | See coordinate-frames.md |
| `Anchor<Frame, T>` | Named reference point | Meta | `.center`, `.face_top`, `.edge_left` |

**Units:** `m`, `mm`, `cm`, `in`, `ft` (length); `rad`, `deg` (angle); `m³` (volume); `m²` (area).

Type safety prevents mixing incompatible operations (e.g., can't extrude a `Solid`, only a `Sketch` or `Face`).

---

## 3. Structural Constructs

### 3.1 part

Defines a self-contained geometric component.

```morphogen
part Bracket {
    let base = sketch.rectangle(width=50mm, height=30mm)
        |> extrude(height=5mm)

    let hole = sketch.circle(radius=5mm)
        |> extrude(height=5mm)
        |> place(anchor=.center, at=base.anchor("face_top").offset(10mm, 10mm, 0))

    base - hole  # Boolean difference
}
```

- Parts are pure functions (parameters → geometry)
- Parts can be instantiated multiple times with different parameters

### 3.2 assembly

Composes multiple parts using reference-based placement.

```morphogen
assembly Tower {
    let base = part.call(Bracket, width=100mm)
    let pillar = geom.cylinder(radius=10mm, height=200mm)
    let cap = geom.sphere(radius=20mm)

    [
        base,
        pillar |> place(anchor=.bottom, at=base.anchor("face_top")),
        cap    |> place(anchor=.bottom, at=pillar.anchor("face_top"))
    ]
}
```

- **No parent-child hierarchy** — flat list of parts with explicit placement
- **Declarative** — placement describes intent ("bottom of pillar → top of base")
- **Refactor-safe** — changing `base` doesn't break `pillar` placement

### 3.3 import / export

Interoperability with standard formats.

```morphogen
import step("bracket.step")       # Import STEP CAD file
import stl("mesh.stl")            # Import STL mesh
export step("output.step", assembly=Tower)
export stl("output.stl", mesh=result, tolerance=0.01mm)
```

---

## 4. Coordinate System & Frames

See **coordinate-frames.md** for full details.

**Core concepts:**

- **Frame<3>** — Local coordinate system (origin, basis, scale)
- **Anchor** — Named reference point (`.center`, `.face_top`, `.edge_left`)
- **Placement** — Map object anchor to target anchor (declarative positioning)

**Default frame:** Right-handed Cartesian (XYZ), origin at (0,0,0), units in `mm`.

**Frame conversions:**
```morphogen
# Cartesian to cylindrical
let cylindrical = transform.to_coord(solid, coord_type=Cylindrical)

# Spherical coordinates
let spherical = transform.to_coord(field, coord_type=Spherical)
```

---

## 5. Geometric Operators

Each operator is a pure function from one or more geometric objects to one or more geometric objects.

### 5.1 Primitives (3D Solids)

```morphogen
geom.box(width, height, depth, centered=true)
geom.sphere(radius)
geom.cylinder(radius, height, centered=true)
geom.cone(radius_bottom, radius_top, height)
geom.torus(major_radius, minor_radius)
geom.wedge(dx, dy, dz, xmin, zmin, xmax, zmax)  # Tapered box
```

**Examples:**
```morphogen
let cube = geom.box(10mm, 10mm, 10mm)
let ball = geom.sphere(5mm)
let pipe = geom.cylinder(radius=3mm, height=20mm)
```

**Properties:**
- All primitives centered at origin by default (unless `centered=false`)
- Auto-generate anchors: `.center`, `.face_{top,bottom,...}`, `.edge_{...}`, `.corner_{...}`

---

### 5.2 Sketch Operations (2D → 2D)

Sketches are 2D planar constructions (on XY plane by default).

```morphogen
sketch.rectangle(width, height, centered=true)
sketch.circle(radius)
sketch.ellipse(major, minor)
sketch.polygon(points: [(f64, f64)])
sketch.regular_polygon(n_sides, radius)
sketch.arc(radius, start_angle, end_angle)
sketch.spline(points, tangents=auto)
sketch.text(string, font, size)
```

**Sketch boolean ops:**
```morphogen
sketch.union(s1, s2, ...)
sketch.difference(s1, s2)
sketch.intersection(s1, s2)
sketch.offset(sketch, distance)  # Parallel offset
```

**Examples:**
```morphogen
# Rectangle with circular hole
let plate = sketch.rectangle(50mm, 30mm)
let hole = sketch.circle(5mm)
let plate_with_hole = sketch.difference(plate, hole)

# Rounded rectangle (offset + fillet)
let rounded = sketch.rectangle(40mm, 20mm)
    |> sketch.offset(-2mm)  # Inset
    |> sketch.fillet(radius=2mm)
```

---

### 5.3 Extrusion & Revolution (2D → 3D)

```morphogen
extrude(sketch, height)
extrude_along(sketch, path: Wire)
revolve(sketch, axis="z", angle=360deg)
loft(sketches: [Sketch], ruled=false)
sweep(profile: Sketch, path: Wire, twist=0deg)
```

**Examples:**
```morphogen
# Simple extrusion
let block = sketch.rectangle(20mm, 10mm)
    |> extrude(height=5mm)

# Revolution (create vase)
let profile = sketch.polygon([(0,0), (10,0), (8,20), (5,25)])
let vase = revolve(profile, axis="y", angle=360deg)

# Loft between circles (cone-like shape)
let bottom = sketch.circle(10mm)
let top = sketch.circle(5mm) |> transform.translate((0, 0, 20mm))
let tapered = loft([bottom, top])

# Sweep along path
let circle_profile = sketch.circle(2mm)
let helix_path = geom.helix(radius=10mm, pitch=5mm, turns=3)
let spring = sweep(circle_profile, path=helix_path)
```

---

### 5.4 Boolean Operations (3D)

```morphogen
geom.union(s1, s2, ...)         # Combine solids
geom.difference(s1, s2)         # Subtract s2 from s1
geom.intersection(s1, s2)       # Common volume
geom.symmetric_difference(s1, s2)  # XOR
```

**Operator overloading:**
```morphogen
let combined = solid_A + solid_B       # Union
let subtracted = solid_A - solid_B     # Difference
let intersect = solid_A & solid_B      # Intersection
```

**Examples:**
```morphogen
# Bracket with hole
let base = geom.box(50mm, 30mm, 5mm)
let hole = geom.cylinder(radius=3mm, height=5mm)
    |> place(anchor=.center, at=base.anchor("center"))
let bracket = base - hole

# Fillet union (rounded join)
let rounded_union = geom.union(part_A, part_B, fillet=2mm)
```

---

### 5.5 Finishing Operations

```morphogen
geom.fillet(solid, edges: [Edge], radius)    # Round edges
geom.chamfer(solid, edges: [Edge], distance)  # Bevel edges
geom.shell(solid, faces: [Face], thickness)   # Hollow out
geom.draft(solid, faces: [Face], angle, neutral_plane)  # Taper for molding
```

**Edge/face selection:**
```morphogen
# Select edges by filter
let top_edges = solid.edges(filter = λ e: e.center().z > 10mm)

# Select faces by normal direction
let vertical_faces = solid.faces(filter = λ f: abs(f.normal().z) < 0.1)

# Predefined selectors
solid.edges(">Z")   # Edges with highest Z
solid.faces("|Z")   # Faces parallel to Z axis
```

**Examples:**
```morphogen
# Fillet all top edges
let rounded = geom.box(20mm, 20mm, 10mm)
    |> geom.fillet(edges=.edges(">Z"), radius=2mm)

# Chamfer specific edges
let chamfered = solid
    |> geom.chamfer(edges=[edge_1, edge_2], distance=1mm)

# Shell (hollow out)
let hollow_box = geom.box(30mm, 30mm, 30mm)
    |> geom.shell(faces=[.face("top")], thickness=2mm)
```

---

### 5.6 Pattern Operations

```morphogen
pattern.linear(object, direction, count, spacing)
pattern.circular(object, axis, count, angle=360deg)
pattern.grid(object, rows, cols, row_spacing, col_spacing)
pattern.along_path(object, path: Wire, count, align=true)
```

**Examples:**
```morphogen
# Linear array
let hole = geom.cylinder(radius=2mm, height=5mm)
let holes_linear = pattern.linear(
    hole,
    direction=(10mm, 0, 0),
    count=5,
    spacing=10mm
)

# Circular pattern (bolt holes)
let bolt = geom.cylinder(radius=3mm, height=10mm)
let bolts = pattern.circular(
    bolt,
    axis="z",
    count=6,
    angle=360deg
) |> place(anchor=.center, at=(0, 30mm, 0))  # Offset from center

# Grid pattern
let pin = geom.cylinder(radius=0.5mm, height=3mm)
let pin_grid = pattern.grid(
    pin,
    rows=10,
    cols=10,
    row_spacing=2.54mm,
    col_spacing=2.54mm
)
```

---

### 5.7 Transformations

See **transform.md** and **coordinate-frames.md**.

```morphogen
transform.translate(object, offset: Vec3)
transform.rotate(object, angle, axis, origin=.center)
transform.scale(object, factor, origin=.center)
transform.mirror(object, plane)
transform.affine(object, matrix: Mat4)
```

**Examples:**
```morphogen
# Translate
let moved = geom.box(10mm, 10mm, 10mm)
    |> transform.translate((20mm, 0, 0))

# Rotate around custom origin
let rotated = cylinder
    |> transform.rotate(
        angle=45deg,
        axis="z",
        origin=cylinder.anchor("edge_bottom_left")
    )

# Mirror across XY plane
let mirrored = solid |> transform.mirror(plane="xy")

# Scale non-uniformly
let stretched = box
    |> transform.scale(factor=(2.0, 1.0, 0.5), origin=.center)
```

---

### 5.8 Measurement & Query

```morphogen
geom.measure.volume(solid: Solid) -> f64[m³]
geom.measure.area(face: Face) -> f64[m²]
geom.measure.length(edge: Edge) -> f64[m]
geom.measure.bounds(object) -> BoundingBox
geom.measure.center_of_mass(solid) -> Vec3
geom.measure.normal(face, at: Vec2) -> Vec3
geom.measure.distance(obj_a, obj_b) -> f64
```

**Examples:**
```morphogen
let box = geom.box(10mm, 20mm, 30mm)
let vol = geom.measure.volume(box)  # Returns 6000 mm³

let bbox = geom.measure.bounds(box)
assert_eq!(bbox.width, 10mm)
assert_eq!(bbox.height, 20mm)
assert_eq!(bbox.depth, 30mm)

let com = geom.measure.center_of_mass(complex_solid)
let distance = geom.measure.distance(solid_a, solid_b)
```

---

### 5.9 Mesh Operations

Discrete mesh processing (for STL, OBJ, analysis).

```morphogen
mesh.from_solid(solid, tolerance=0.01mm) -> Mesh<Vertex>
mesh.subdivide(mesh, method="catmull-clark|loop", iterations=1)
mesh.smooth(mesh, iterations=1)
mesh.decimate(mesh, target_faces=1000)
mesh.laplacian(mesh) -> SparseMatrix  # Mesh Laplacian
mesh.sample(mesh, field: Field<T>) -> Mesh<T>  # Sample field at vertices
mesh.normals(mesh) -> Mesh<Vec3>
```

**Examples:**
```morphogen
# Convert solid to mesh
let solid = geom.sphere(10mm)
let mesh = mesh.from_solid(solid, tolerance=0.1mm)

# Subdivide for smoothness
let smooth = mesh.subdivide(mesh, method="catmull-clark", iterations=2)

# Compute Laplacian for PDE solving
let L = mesh.laplacian(mesh)

# Sample field at mesh vertices
let temperature_field = field.solve_heat(...)
let temp_mesh = mesh.sample(mesh, temperature_field)
```

---

### 5.10 Advanced Operations

```morphogen
geom.offset(solid, distance)  # Offset surface (positive = expand)
geom.thicken(face, thickness)  # Convert face to solid
geom.project(wire, face) -> Wire  # Project curve onto surface
geom.split(solid, face) -> [Solid]  # Split solid by face
geom.convex_hull(points: [Vec3]) -> Solid
geom.voronoi(points: [Vec3], bounds: BoundingBox) -> [Solid]
```

---

## 6. Anchor System

See **coordinate-frames.md** for full specification.

### Auto-Generated Anchors

Every geometric object automatically provides anchors:

| Object | Anchors |
|--------|---------|
| **Box** | `.center`, `.face_{top,bottom,left,right,front,back}`, `.corner_{...}`, `.edge_{...}` |
| **Cylinder** | `.center`, `.face_{top,bottom}`, `.axis`, `.edge_{top,bottom}` |
| **Sphere** | `.center`, `.pole_{north,south}`, `.equator` |
| **Generic Solid** | `.center` (bounding box center), `.face_{...}`, `.edge_{...}`, `.vertex_{...}` |

### Anchor Queries

```morphogen
# Direct name lookup
box.anchor("face_top")

# Pattern-based (highest Z face)
box.anchor(">Z")

# Filter-based
solid.anchor("face", filter = λ f: f.area() > 100mm²)
```

### Placement via Anchors

```morphogen
let part_B = mesh.place(
    part_B,
    anchor = part_B.anchor("bottom"),
    at = part_A.anchor("face_top"),
    align_orientation = true  # Match orientations
)
```

---

## 7. Determinism & Profiles

All geometry operations respect Morphogen's determinism profiles:

| Operation | Profile | Notes |
|-----------|---------|-------|
| Primitives | Strict | Exact mathematical definitions |
| Boolean ops | Strict | Deterministic within floating precision |
| Extrude/revolve | Strict | Pure geometric transforms |
| Fillet/chamfer | Repro | Iterative solver (backend-dependent) |
| Mesh generation | Repro | Tessellation parameters affect output |
| Loft/sweep | Repro | Spline fitting |

**Profile annotations:**
```morphogen
@determinism(strict)
let box = geom.box(10mm, 10mm, 10mm)

@determinism(repro)
let filleted = geom.fillet(box, edges=..., radius=2mm)
```

---

## 8. Backend Integration

Morphogen.Geometry is backend-neutral. Operations are semantically defined; lowering varies by backend.

| Backend | Status | Capabilities |
|---------|--------|--------------|
| **CadQuery** | Planned | Full 3D CAD (OCCT-based) |
| **CGAL** | Future | Robust boolean ops, mesh processing |
| **OpenCASCADE** | Future | Industrial CAD kernel |
| **GPU Mesh Kernel** | Research | Parallel mesh operations |
| **Implicit Surface (SDFs)** | Research | GPU-friendly representations |

**Backend capabilities:**
```yaml
operator:
  name: geom.boolean.union
  input_types: [Solid, Solid]
  output_types: [Solid]
  determinism: strict
  backend_caps:
    cadquery: supported
    cgal: supported
    gpu_sdf: supported (implicit conversion)
    wasm: partial (no NURBS)
```

**Backend selection:**
```morphogen
@backend(cadquery)
let solid = geom.box(...) + geom.sphere(...)

@backend(gpu_sdf)
let field = field.from_solid(solid)  # Converts to SDF
```

---

## 9. Integration with Other Domains

### 9.1 Fields (CFD, Heat Transfer)

```morphogen
# Convert geometry to signed distance field
let solid = geom.sphere(10mm)
let sdf = field.from_solid(solid, bounds=..., resolution=(100,100,100))

# Sample field at surface
let temperature = field.solve_heat(...)
let surface_temp = mesh.sample(mesh.from_solid(solid), temperature)
```

### 9.2 Physics (Collision, Dynamics)

```morphogen
# Create rigid body from geometry
let solid = geom.box(10mm, 10mm, 10mm)
let body = physics.rigid_body(
    shape = solid,
    mass = 1.0 kg,
    frame = solid.anchor("center").frame()
)
```

### 9.3 Visuals (Rendering)

```morphogen
# Render geometry
let rendered = visual.render(
    solid,
    camera_frame = camera.frame(),
    lighting = "pbr",
    material = material.metal(roughness=0.2)
)
```

---

## 10. Examples

### Example 1: Parametric Bracket

```morphogen
part Bracket(width=50mm, height=30mm, thickness=5mm, hole_radius=3mm) {
    # Base plate
    let base = sketch.rectangle(width, height)
        |> extrude(thickness)

    # Mounting holes (4 corners)
    let hole = sketch.circle(hole_radius)
        |> extrude(thickness)

    let holes = [
        hole |> place(anchor=.center, at=base.anchor("corner_nw").offset(5mm, -5mm, 0)),
        hole |> place(anchor=.center, at=base.anchor("corner_ne").offset(-5mm, -5mm, 0)),
        hole |> place(anchor=.center, at=base.anchor("corner_sw").offset(5mm, 5mm, 0)),
        hole |> place(anchor=.center, at=base.anchor("corner_se").offset(-5mm, 5mm, 0))
    ]

    # Subtract holes
    base - holes.fold(geom.union)
}

# Instantiate with different sizes
let small = Bracket(width=40mm, height=25mm)
let large = Bracket(width=80mm, height=60mm, hole_radius=5mm)
```

### Example 2: Assembly with Reference-Based Placement

```morphogen
assembly RobotArm {
    let base = geom.cylinder(radius=30mm, height=20mm)
    let link_1 = geom.box(10mm, 10mm, 100mm)
    let joint = geom.sphere(radius=8mm)
    let link_2 = geom.box(8mm, 8mm, 80mm)

    [
        base,

        # Link 1 stands on base
        link_1 |> place(
            anchor = link_1.anchor("face_bottom"),
            at = base.anchor("face_top")
        ),

        # Joint at top of link 1
        joint |> place(
            anchor = .center,
            at = link_1.anchor("face_top")
        ),

        # Link 2 starts at joint (rotated 45 degrees)
        link_2
            |> transform.rotate(angle=45deg, axis="y", origin=.center)
            |> place(
                anchor = link_2.anchor("face_bottom"),
                at = joint.anchor("pole_north")
            )
    ]
}
```

### Example 3: Circular Pattern (Bolt Holes)

```morphogen
part FlangePlate(radius=50mm, thickness=10mm, bolt_count=8, bolt_radius=4mm) {
    # Main disc
    let disc = sketch.circle(radius)
        |> extrude(thickness)

    # Center hole
    let center_hole = sketch.circle(radius * 0.3)
        |> extrude(thickness)

    # Bolt hole pattern
    let bolt_hole = sketch.circle(bolt_radius)
        |> extrude(thickness)
        |> transform.translate((radius * 0.75, 0, 0))  # Offset from center

    let bolt_pattern = pattern.circular(
        bolt_hole,
        axis = "z",
        count = bolt_count
    )

    # Subtract holes
    disc - center_hole - bolt_pattern.fold(geom.union)
}
```

### Example 4: Lofted Shape (Vase)

```morphogen
part Vase(base_radius=30mm, top_radius=20mm, height=100mm, n_sections=5) {
    # Create profile sections
    let sections = (0..n_sections).map(λ i: {
        let t = i / (n_sections - 1)  # 0..1
        let z = t * height
        let r = lerp(base_radius, top_radius, t)

        sketch.circle(r)
            |> transform.translate((0, 0, z))
    })

    # Loft between sections
    loft(sections, ruled=false)
}
```

### Example 5: Mesh Processing & Analysis

```morphogen
# Load STL, smooth, and analyze
let raw_mesh = import("scan.stl")

let smoothed = raw_mesh
    |> mesh.subdivide(method="loop", iterations=1)
    |> mesh.smooth(iterations=3)

# Compute mesh Laplacian
let L = mesh.laplacian(smoothed)

# Solve heat equation on mesh
let heat_field = field.solve_heat(
    laplacian = L,
    boundary_conditions = ...,
    dt = 0.01 s,
    steps = 100
)

# Visualize
let colored_mesh = mesh.sample(smoothed, heat_field)
visual.render(colored_mesh, colormap="viridis")
```

---

## 11. Testing Strategy

### 11.1 Determinism Tests

All geometric operations must pass golden tests:

```morphogen
# Primitives are deterministic
assert_eq!(
    geom.box(10mm, 10mm, 10mm),
    geom.box(10mm, 10mm, 10mm)
)

# Boolean ops are deterministic
let result_1 = solid_A + solid_B
let result_2 = solid_A + solid_B
assert_solid_eq!(result_1, result_2, tolerance=1e-12)

# Anchor resolution is deterministic
assert_eq!(
    box.anchor("face_top"),
    box.anchor("face_top")
)
```

### 11.2 Measurement Tests

Verify geometric properties:

```morphogen
let cube = geom.box(10mm, 10mm, 10mm)
assert_approx_eq!(geom.measure.volume(cube), 1000.0 mm³, tol=1e-9)

let sphere = geom.sphere(radius=10mm)
assert_approx_eq!(
    geom.measure.volume(sphere),
    (4.0/3.0) * π * (10mm)³,
    tol=1e-6
)
```

### 11.3 Transformation Tests

Verify explicit origins:

```morphogen
# Rotation around center preserves center position
let box = geom.box(10mm, 10mm, 10mm)
let rotated = transform.rotate(box, 45deg, axis="z", origin=.center)
assert_vec_eq!(
    rotated.anchor("center").position(),
    box.anchor("center").position(),
    tol=1e-12
)
```

### 11.4 Backend Equivalence Tests

Verify different backends produce equivalent results:

```morphogen
@backend(cadquery)
let result_cq = geom.box(10mm, 10mm, 10mm) + geom.sphere(5mm)

@backend(cgal)
let result_cgal = geom.box(10mm, 10mm, 10mm) + geom.sphere(5mm)

assert_solid_equivalent!(result_cq, result_cgal, tolerance=1e-6)
```

---

## 12. Future Extensions

### 12.1 NURBS & Splines

```morphogen
nurbs.surface(control_points, u_degree, v_degree, u_knots, v_knots)
nurbs.curve(control_points, degree, knots)
bezier.curve(control_points)
```

### 12.2 Topology Optimization

```morphogen
optimize.topology(
    domain = bounding_box,
    loads = [...],
    constraints = [...],
    objective = "minimize_compliance",
    volume_fraction = 0.3
)
```

### 12.3 Generative Design

```morphogen
generate.lattice(
    unit_cell = "gyroid",
    bounds = bounding_box,
    cell_size = 5mm
)
```

### 12.4 Sheet Metal Operations

```morphogen
sheet.bend(face, angle, radius)
sheet.unfold(solid) -> Sketch
```

---

## 13. Operator Registry Summary

| Category | Operators | Layer |
|----------|-----------|-------|
| **Primitives** | box, sphere, cylinder, cone, torus, wedge | Layer 6b |
| **Sketches** | rectangle, circle, polygon, arc, spline | Layer 6b |
| **Extrusion** | extrude, revolve, loft, sweep | Layer 6b |
| **Booleans** | union, difference, intersection | Layer 6b |
| **Patterns** | linear, circular, grid, along_path | Layer 6b |
| **Finishing** | fillet, chamfer, shell, draft | Layer 6b |
| **Transforms** | translate, rotate, scale, mirror, affine | Layer 2 |
| **Measurement** | volume, area, length, bounds, COM, distance | Layer 6b |
| **Mesh** | from_solid, subdivide, laplacian, sample | Layer 4b |
| **Anchors** | anchor.resolve, anchor.position, place | Layer 2 |

See **operator-registry.md** for full registry integration.

---

## 14. References

- **TiaCAD v3.x** — Reference-based composition model, anchor system, declarative CAD
- **coordinate-frames.md** — Frame/anchor system specification
- **transform.md** — Spatial transformations
- **operator-registry.md** — 7-layer operator architecture
- **../architecture/domain-architecture.md** — Cross-domain vision
- **CadQuery** — Python-based CAD scripting (backend target)
- **OpenCASCADE** — Industrial CAD kernel (backend target)

---

## Summary

Morphogen.Geometry provides:

1. **Declarative CAD** — Describe what to build, not how
2. **Reference-based composition** — Anchors replace hierarchies (TiaCAD model)
3. **Deterministic transforms** — Explicit origins, pure functions
4. **Backend-neutral** — Semantic operations, multiple lowering targets
5. **Cross-domain integration** — Works with Fields, Physics, Visuals
6. **Type safety** — Strong typing prevents invalid operations
7. **Parametric modeling** — Parts/assemblies are pure functions

**Key innovation from TiaCAD:** Anchors unify spatial references across domains, making composition robust and declarative.

---

**Status:** v1.0 Draft — ready for review and implementation planning
**Next steps:** Backend integration (CadQuery), MLIR lowering, cross-domain examples
