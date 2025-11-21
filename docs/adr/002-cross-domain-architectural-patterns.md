# ADR 002: Cross-Domain Architectural Patterns from TiaCAD, RiffStack, and Strudel

**Status:** APPROVED
**Date:** 2025-11-15
**Authors:** Morphogen Architecture Team
**Supersedes:** N/A
**Related:** ADR-001 (Unified Reference Model)

---

## Context

Morphogen's evolution from a single-domain system to a multi-domain platform requires proven architectural patterns. We have analyzed three successful domain-specific systems:

1. **TiaCAD** - Parametric CAD with spatial references and geometry operators
2. **RiffStack** - Audio DSP framework with extensible operator registry
3. **Strudel/TidalCycles** - Live coding pattern systems for temporal composition

Each system independently developed solutions to common architectural challenges:
- Domain-specific type systems
- Operator extensibility and discovery
- Reference systems for composition
- Multi-layer complexity models (beginner → expert)
- Optimization passes and lowering strategies

This ADR extracts and generalizes these patterns for Morphogen's multi-domain architecture.

---

## Decision

### 1. Domain Isolation with Unified Interface Contracts

**Pattern:** Each domain is self-contained with standardized interfaces.

**From TiaCAD & RiffStack:**
- Clear separation between domain logic and execution runtime
- Unified operator registry per domain
- Domain-specific reference types (SpatialRef, AudioNodeRef, etc.)
- Self-contained builder pipelines

**Morphogen Implementation:**
```
Every Morphogen domain MUST provide:
├── Types (domain-specific primitives + reference types)
├── Operators (layered: atomic → composite → constructs → presets)
├── Passes (optimization + lowering strategies)
├── Constraints (validation + domain invariants)
└── Debug Tools (introspection + visualization)
```

**Examples:**
- **PhysicsDomain:** BodyRef, ParticleRef, ForceRef + integrators + spatial partitioning passes
- **AudioDomain:** NodeRef, PortRef + generators/filters/FX + graph optimization passes
- **GeometryDomain:** SpatialRef, FaceRef + primitives/transforms/booleans + mesh simplification passes
- **FinanceDomain:** CurveRef, MonteCarloRef + stochastic processes + variance reduction passes

---

### 2. Unified Reference Systems (One Reference Type Per Domain)

**Pattern:** Each domain gets a single, unified reference type with auto-generated anchors.

**From TiaCAD's SpatialRef breakthrough:**
- One reference type (`SpatialRef`) unified all spatial composition
- Auto-generated anchors (`face_top`, `axis_y`, `center`, `origin`)
- Zero fragmentation: no separate FaceRef/EdgeRef/VertexRef in user code
- Local frames for domain-relative transformations

**Morphogen Generalization:**
```
Every domain MUST have:
1. One primary reference type (e.g., PhysicsRef, AudioRef)
2. Auto-generated anchors/interfaces
3. Frame + orientation + metadata
4. Composability via reference chaining
```

**Domain-Specific Reference Models:**

| Domain | Primary Ref | Auto-Generated Anchors | Use Case |
|--------|-------------|------------------------|----------|
| **Physics** | `BodyRef` | `center_of_mass`, `collision_normal`, `local_axes` | Force application, collision detection |
| **Audio** | `NodeRef` | `input_port[0..n]`, `output_port[0..n]`, `param[name]` | Audio graph patching |
| **Graphics** | `ObjectRef` | `bounding_box.{min,max,center}`, `local_transform` | Scene composition |
| **Finance** | `CurveRef` | `curve_point[t]`, `expiry`, `strike_grid` | Derivatives pricing |
| **Geometry** | `SpatialRef` | `face_{top,bottom,left,right}`, `axis_{x,y,z}`, `center` | CAD composition |
| **Neural** | `LayerRef` | `weights`, `biases`, `activations`, `gradients` | Network assembly |
| **Pattern** | `EventRef` | `cycle[n]`, `beat[n]`, `subdivision[n]` | Temporal sequencing |

**Benefits:**
- Dramatic UX improvement: fewer raw coordinates, cleaner patterns
- Enables reference-based composition without internal implementation leakage
- Automatic validation and type safety

---

### 3. Auto-Registration and Plugin Discovery

**Pattern:** Zero-friction extensibility through automatic operator discovery.

**From TiaCAD & RiffStack:**
- Decorators for operator metadata (`@operator`, `@generator`, `@filter`)
- Registry scanning at startup
- User extension folders (`~/.morphogen/domains/{name}/ops/`)
- No manual editing of central operator tables

**Morphogen Implementation:**
```python
# User-defined operator in ~/.morphogen/domains/physics/ops/custom_force.py

@morphogen.operator(
    domain="physics",
    category="forces",
    layer=2,  # composite operator
    deterministic=True,
    tags=["gravity", "n-body"]
)
def barnes_hut_gravity(
    bodies: List[BodyRef],
    theta: float = 0.5,  # accuracy parameter
    G: float = 6.674e-11
) -> ForceField:
    """Barnes-Hut approximation for O(N log N) gravity."""
    # Implementation
    ...
```

**Auto-Discovery Process:**
1. Scan `morphogen/stdlib/{domain}/` (built-in operators)
2. Scan `~/.morphogen/domains/{domain}/ops/` (user operators)
3. Build operator registry with metadata
4. Validate signatures and determinism profiles
5. Enable IDE autocomplete and type checking

**Result:** Morphogen becomes a **platform**, not a library.

---

### 4. Multi-Layer Complexity Model

**Pattern:** Four-layer operator hierarchy from atomic to high-level.

**From TiaCAD & RiffStack:**
Both systems independently converged on layered complexity:

```
Layer 1: Atomic Ops       (add, multiply, sine, sphere)
Layer 2: Composite Ops    (biquad filter, loft, monte_carlo_step)
Layer 3: Constructs/DSL   (reverb, sketch, heston_model)
Layer 4: Presets          (studio_reverb, bolt_pattern, black_scholes)
```

**Morphogen Adoption:**
Every domain implements this hierarchy to serve:
- **Beginners:** Use Layer 4 presets
- **Intermediate:** Compose Layer 3 constructs
- **Advanced:** Build from Layer 2 composites
- **Experts:** Access Layer 1 atomic ops + raw MLIR

**Example: AudioDomain Layers**
```
Layer 1: sine, add, multiply, delay_sample
Layer 2: biquad_filter, envelope_adsr, wavetable_osc
Layer 3: reverb_schroeder, chorus, compressor
Layer 4: vocal_chain_preset, mastering_chain, studio_fx_rack
```

**Example: PhysicsDomain Layers**
```
Layer 1: gravity_force_pair, euler_step, octree_insert
Layer 2: barnes_hut_force, rk4_integrator, broadphase_grid
Layer 3: n_body_system, collision_detector, rigid_body_dynamics
Layer 4: solar_system_preset, molecular_dynamics, granular_flow
```

---

### 5. Pass-Based Optimization is Universal

**Pattern:** Domain-specific optimization passes, not just MLIR compilation.

**From TiaCAD & RiffStack:**
- **TiaCAD:** Geometry passes (mesh simplification, constraint solving, auto-anchoring)
- **RiffStack:** DSP passes (filter merging, node pruning, graph flattening)
- **Insight:** Passes are **domain primitives**, not compiler-specific

**Morphogen Generalization:**
```
Every domain needs its own pass system:
├── Validation Passes   (type checking, constraint solving)
├── Optimization Passes (domain-specific simplifications)
├── Lowering Passes     (domain IR → MLIR dialects)
└── Backend Passes      (CPU/GPU code generation)
```

**Domain Pass Examples:**

| Domain | Pass Type | Example Pass | Transformation |
|--------|-----------|--------------|----------------|
| **Physics** | Optimization | `SymplecticEnforcement` | Euler → Verlet/Yoshida |
| **Physics** | Lowering | `NBodyToBarnesHut` | O(N²) → O(N log N) |
| **Audio** | Optimization | `FilterMerging` | Cascade filters → single biquad |
| **Audio** | Lowering | `GraphFlattening` | Node graph → vectorized DSP |
| **Finance** | Optimization | `VarianceReduction` | Crude MC → antithetic/Sobol |
| **Finance** | Lowering | `MonteCarloToGPU` | Serial → CUDA kernels |
| **Geometry** | Optimization | `MeshSimplification` | High-poly → LOD meshes |
| **Geometry** | Lowering | `BooleanToMesh` | CSG tree → triangle mesh |
| **Graphics** | Optimization | `SceneFlattening` | Deep hierarchy → flat arrays |
| **Graphics** | Lowering | `ShaderOptimization` | AST → SPIR-V |
| **Neural** | Optimization | `KernelFusion` | Separate ops → fused kernel |
| **Neural** | Lowering | `QuantizationPass` | FP32 → INT8 |

**Implementation:**
```python
# morphogen/mlir/passes/physics/symplectic_enforcement.py

class SymplecticEnforcementPass(DomainPass):
    """Convert non-symplectic integrators to symplectic forms."""

    domain = "physics"

    def visit_integrator_op(self, op):
        if op.method == "euler" and op.is_hamiltonian:
            # Replace with Verlet for energy conservation
            return self.replace_with_verlet(op)
        elif op.method == "rk4" and op.requires_symplectic:
            # Replace with Yoshida 4th-order
            return self.replace_with_yoshida(op)
        return op
```

**Key Insight:** Passes enable **expert-level optimization** without forcing users to understand MLIR.

---

### 6. Auto-Generated Anchors for Usability

**Pattern:** Automatic generation of named access points (anchors/interfaces).

**From TiaCAD's Breakthrough:**
Auto-generated anchors (`face_top`, `center`, `axis_y`) eliminated the need for raw coordinate manipulation:

```python
# Before (raw coordinates):
cylinder = Cylinder(base=(0, 0, 0), height=10, radius=5)
box = Box(corner=(5, 0, 10), ...)  # Must compute manually

# After (auto-anchors):
cylinder = Cylinder(height=10, radius=5)
box = Box().place_on(cylinder.face_top)  # Automatic!
```

**Morphogen Generalization:**
Auto-anchors apply to **every domain**:

**Physics Domain:**
```python
body = RigidBody(mass=10, shape=sphere)
# Auto-anchors:
force.apply_at(body.center_of_mass)
sensor.attach_to(body.collision_normal)
frame.align_with(body.local_axes.x)
```

**Audio Domain:**
```python
filter = BiquadFilter(freq=440, Q=2.0)
# Auto-ports:
osc >> filter.input[0]
filter.output[0] >> reverb.input["left"]
filter.param["freq"].modulate(lfo)
```

**Finance Domain:**
```python
curve = YieldCurve(tenors=[1, 2, 5, 10], rates=[...])
# Auto-anchors:
forward_rate = curve.point[2.5]  # Interpolated
price = option.value_at(curve.expiry)
scenario = monte_carlo.on_grid(curve.strike_grid)
```

**Graphics Domain:**
```python
mesh = GLTFModel("character.glb")
# Auto-anchors:
light.look_at(mesh.bounding_box.center)
camera.position = mesh.bounding_box.max + Vector3(0, 2, 0)
particle_emitter.spawn_at(mesh.local_transform.origin)
```

**Implementation Strategy:**
1. **Introspection:** Domain objects expose anchor metadata
2. **Code Generation:** Anchors generated at graph build time
3. **Type Safety:** Anchors typed per domain (SpatialAnchor, AudioPort, etc.)
4. **Documentation:** Auto-generate anchor lists in docs

**Result:** Massive UX win - cleaner patterns, fewer raw values, better composition.

---

### 7. Domain Separation + Cross-Domain Flows

**Pattern:** Strict domain isolation with well-defined inter-domain interfaces.

**Key Insight:**
Morphogen's power comes from:
1. **Isolated domains** (no internal leakage)
2. **Composable flows** between domains

**Cross-Domain Flow Examples:**

| Source → Target | Flow Type | Example Use Case |
|-----------------|-----------|------------------|
| **Physics → Audio** | Sonification | Collision forces → percussion synthesis |
| **Audio → Graphics** | Visualization | FFT spectrum → particle colors |
| **Geometry → Physics** | Mesh Import | CAD mesh → collision geometry |
| **Finance → ML** | Data Feed | Monte Carlo paths → training data |
| **ML → Geometry** | Generation | GAN → procedural 3D shapes |
| **Pattern → Audio** | Sequencing | TidalCycles patterns → audio events |
| **Pattern → Graphics** | Animation | Euclidean rhythms → keyframe timing |
| **Physics → Graphics** | Rendering | Particle positions → instanced rendering |

**Implementation:**
```python
# Example: Physics → Audio sonification

# Physics domain
bodies = PhysicsDomain.n_body_simulation(
    num_bodies=100,
    integrator="verlet",
    forces=["gravity", "collision"]
)

# Cross-domain flow (physics → audio)
collisions = bodies.events.on_collision()
audio_triggers = AudioDomain.from_physics_events(
    collisions,
    mapping={
        "impulse": "amplitude",  # Force → volume
        "body_id": "pitch",      # Body → frequency
        "position.x": "pan"      # Position → stereo
    }
)

# Audio domain
drums = AudioDomain.percussion_synth(
    triggers=audio_triggers,
    envelope=ADSR(attack=0.001, decay=0.1)
)
```

**Cross-Domain Interface Contract:**
```python
# morphogen/interfaces/cross_domain.py

class DomainInterface:
    """Base class for inter-domain data flows."""

    source_domain: str
    target_domain: str

    def transform(self, source_data: Any) -> Any:
        """Convert source domain data to target domain format."""
        raise NotImplementedError

    def validate(self) -> bool:
        """Ensure data types are compatible across domains."""
        raise NotImplementedError
```

**Key Requirement:**
Each domain MUST expose:
1. **Input Interface:** What data it can accept from other domains
2. **Output Interface:** What data it can provide to other domains
3. **Transform Functions:** Conversion between domain data models

---

## Consequences

### Positive

1. **Unified Architecture**
   - Every domain follows the same structural pattern
   - Reduced cognitive load for developers
   - Consistent user experience across domains

2. **Extensibility**
   - Plugin operators without core code modification
   - Domain-specific optimizations without MLIR expertise
   - Community can contribute domain libraries

3. **Composability**
   - Reference-based composition is safe and typed
   - Cross-domain flows are first-class features
   - Auto-anchors eliminate boilerplate

4. **Performance**
   - Domain-specific passes optimize before MLIR lowering
   - Multi-tier complexity allows expert control when needed
   - GPU/CPU backends benefit from domain knowledge

5. **Developer Experience**
   - Beginners use high-level presets
   - Experts access low-level controls
   - Progressive disclosure of complexity

6. **Platform Viability**
   - Morphogen becomes extensible platform, not fixed library
   - Third-party domains can integrate cleanly
   - Ecosystem growth through plugin architecture

### Negative

1. **Implementation Complexity**
   - Each domain requires significant upfront design
   - Reference systems need careful type design
   - Pass infrastructure per domain is non-trivial

2. **Documentation Burden**
   - Every domain needs comprehensive docs
   - Cross-domain flows need clear examples
   - Auto-anchor lists must be maintained

3. **Performance Validation**
   - Domain passes must be benchmarked
   - Cross-domain flows may introduce overhead
   - GPU lowering varies by domain complexity

### Mitigations

1. **Domain Templates**
   - Create scaffolding tools: `kairo new-domain <name>`
   - Provide reference implementations (GeometryDomain, AudioDomain)
   - Document patterns in implementation guide

2. **Progressive Rollout**
   - Start with 3 core domains (Geometry, Audio, Physics)
   - Add next-wave domains incrementally
   - Validate architecture before expanding

3. **Testing Infrastructure**
   - Domain conformance tests (all domains must pass)
   - Cross-domain integration tests
   - Performance regression suites

---

## References

- **ADR-001:** Unified Reference Model
- **../architecture/domain-architecture.md:** Comprehensive domain vision
- **../specifications/operator-registry.md:** Operator metadata schema
- **../specifications/coordinate-frames.md:** Frames and anchor system
- **../specifications/geometry.md:** Geometry domain (TiaCAD patterns)

---

## Appendix: Domain Comparison Table

| Pattern | TiaCAD (Geometry) | RiffStack (Audio) | Strudel (Pattern) | Morphogen Generalization |
|---------|-------------------|-------------------|-------------------|----------------------|
| **Reference System** | `SpatialRef` | `NodeRef`, `PortRef` | `EventRef` | One primary ref per domain |
| **Auto-Anchors** | `face_top`, `center` | `input[0]`, `output[0]` | `cycle[n]`, `beat[n]` | Domain-specific anchors |
| **Operator Layers** | 4 layers | 4 layers | 3 layers | Standardized 4-layer model |
| **Extensibility** | `~/.tiacad/ops` | `~/.riffstack/ops` | Pattern combinators | `~/.morphogen/domains/{name}/ops` |
| **Passes** | Mesh simplification | Graph flattening | Pattern expansion | Domain-specific pass system |
| **Type Safety** | Length units | Sample rate types | Temporal types | Unit system per domain |

---

**Conclusion:**

These patterns are not theoretical - they are battle-tested in production systems. Adopting them for Morphogen ensures:
- **Proven architectures** from TiaCAD, RiffStack, and Strudel
- **Unified design** across all domains
- **Extensible platform** for community growth

Morphogen is not a library. **Morphogen is a platform.**
