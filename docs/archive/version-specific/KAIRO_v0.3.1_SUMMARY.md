# Morphogen v0.3.1 - Summary & Decision Document

**Date**: 2025-11-06
**Session**: vebabe-1106
**Decision**: Evolve Creative Computation DSL → Morphogen v0.3.1

---

## Executive Summary

After analyzing Creative Computation DSL v0.2.2 and the Morphogen v0.3.0 spec from ChatGPT, we've created **Morphogen v0.3.1** - a refined specification that:

1. ✅ **Adopts v0.3.0's best ideas** (flow blocks, @state, explicit RNG, profiles)
2. ✅ **Preserves v0.2.2's strengths** (explicit domain types, comprehensive stdlib, clear module system)
3. ✅ **Adds missing completeness** (functions, lambdas, if/else, struct syntax)
4. ✅ **Defers complexity** (Space abstraction, streaming I/O, advanced generics)

**Result**: A complete, implementable language specification ready for v0.3.1 MVP.

---

## Why This Is The Right Move

### 1. Perfect Timing

**Current Status of Creative Computation DSL v0.2.2:**
- ✅ Frontend complete (lexer, parser, type system, AST)
- ❌ Runtime **not started**
- ❌ Field operations **not started**
- ❌ Visualization **not started**

**Implication**: Since the runtime doesn't exist yet, there's **no breaking change**! We can evolve the language semantics before implementation begins.

### 2. Better Semantics

| Aspect | v0.2.2 | Morphogen v0.3.1 | Winner |
|--------|--------|--------------|--------|
| **Time model** | `step` blocks | `flow(dt)` blocks | v0.3.1 - more explicit |
| **State** | `step.state()` | `@state` declarations | v0.3.1 - declarative |
| **RNG** | Implicit seed | Explicit `rng` objects | v0.3.1 - transparent |
| **Type clarity** | Good | Same (kept explicit domains) | Tie |
| **Module system** | Good | Same (preserved) | Tie |
| **Stdlib** | Comprehensive | Same (preserved) | Tie |
| **Examples** | 17 complete | 4 complete (expandable) | v0.2.2 (but transferable) |

**Key Insight**: Morphogen v0.3.1 = v0.3.0 semantics + v0.2.2 completeness

### 3. Better Branding

- **"Creative Computation DSL"** - Generic, long, descriptive
- **"Morphogen"** - Unique, memorable, brandable, elegant

**Etymology**: Cairo (graphics) + Kairos (Greek: opportune moment, creative time)

### 4. Clear Positioning

| Project | Domain | Status | Relationship |
|---------|--------|--------|--------------|
| **RiffStack** | Audio-only (stack-based, YAML) | MVP Complete ✅ | Audio specialist |
| **Morphogen** | Multi-domain (typed DSL) | Foundation complete, runtime in progress | Platform vision |
| **TiaCAD** | Parametric CAD | Established | Architectural inspiration |

**Message**:
- RiffStack = "Audio tool you can use today"
- Morphogen = "Unified creative computation platform"

---

## Key Changes: v0.2.2 → v0.3.1

### Syntax Changes

#### Temporal Model
```diff
# v0.2.2
- step {
-     temp = diffuse(temp, rate=0.1, dt=0.1)
- }

# v0.3.1
+ flow(dt=0.1, steps=100) {
+     temp = diffuse(temp, rate=0.1, dt)
+ }
```

**Why better**: Explicit dt parameter, clearer temporal scope

#### State Management
```diff
# v0.2.2
- step {
-     temp = step.state(field.random((128, 128), seed=42))
- }

# v0.3.1
+ @state temp : Field2D<f32> = random_normal(seed=42, shape=(128, 128))
+
+ flow(dt=0.1) {
+     temp = diffuse(temp, rate=0.2, dt)
+ }
```

**Why better**: Declarative, analyzable, hot-reload friendly

#### RNG
```diff
# v0.2.2
- agents = agent.alloc(count=100, fn=spawn_random)
- # seed is implicit/global

# v0.3.1
+ agents = alloc(count=100, init=spawn_random)
+
+ fn spawn_random(id: u32, rng: RNG) -> Particle {
+     return Particle {
+         pos: rng.uniform_vec2(min=(0, 0), max=(100, 100))
+     }
+ }
```

**Why better**: Explicit randomness, transparent, composable

### Type System: Keep Explicit Domains

**Decision**: Preserve v0.2.2's explicit domain types

```morphogen
# We use this (clear)
temp : Field2D<f32 [K]>
agents : Agents<Particle>
audio : Signal<f32>

# NOT this (over-abstract)
temp : Flow<Field2D<f32 [K]>>     # Too abstract!
```

**Rationale**: Flow<T> is an implementation detail, not user-facing syntax

### New Additions (Not in v0.2.2 or v0.3.0)

#### 1. Function Syntax (Explicit)
```morphogen
fn clamp(x: f32, min: f32, max: f32) -> f32 {
    return max(min, min(x, max))
}
```

#### 2. Lambda Syntax
```morphogen
field.map(|x| x * 2.0)
agents.map(|a| { vel: a.vel * 0.99, pos: a.pos + a.vel * dt })
```

#### 3. If/Else Expressions
```morphogen
color = if temp > 100.0 { "red" } else { "blue" }
```

#### 4. Struct Keyword
```morphogen
struct Particle {
    pos: Vec2<f32 [m]>
    vel: Vec2<f32 [m/s]>
    mass: f32 [kg]
}
```

---

## What We're NOT Including (Deferred to v0.4+)

| Feature | Rationale for Deferring |
|---------|------------------------|
| **Space abstraction** | Adds complexity; Field2D/Field3D shape is sufficient for MVP |
| **Streaming I/O** | Focus on batch mode first; streams add determinism complexity |
| **Generic types** | Can implement concrete types first; generics are optimization |
| **Error handling (Result/try)** | Runtime exceptions sufficient for v0.3; proper errors in v0.4 |
| **Match expressions** | If/else covers most use cases; match is sugar |

**Philosophy**: Ship a complete, working MVP before adding advanced features

---

## Comparison Matrix

### Language Versions

| Feature | v0.2.2 | v0.3.0 (ChatGPT) | v0.3.1 (Final) |
|---------|--------|------------------|----------------|
| **Temporal** | `step` | `flow(dt)` | `flow(dt)` ✅ |
| **State** | `step.state()` | `@state` | `@state` ✅ |
| **RNG** | Implicit | Explicit | Explicit ✅ |
| **Types** | Explicit domains | Abstract Flow<T> | **Explicit domains** ✅ |
| **Functions** | Implicit | Unclear | **Explicit fn/lambda** ✅ |
| **Modules** | Clear | Unclear compose | **Clear** ✅ |
| **Profiles** | Per-op config | Profile blocks | Profile blocks ✅ |
| **Stdlib** | Comprehensive | Overview | **Comprehensive** ✅ |
| **Examples** | 17 complete | Few | **4 complete** (expandable) ✅ |
| **If/Else** | Implicit | Missing | **Explicit** ✅ |
| **Structs** | `=` syntax | `=` syntax | **struct keyword** ✅ |

### Vs. Other Languages

| Feature | Morphogen v0.3.1 | Python+NumPy | GLSL | Faust |
|---------|--------------|--------------|------|-------|
| **Deterministic** | ✅ Bitwise | ⚠️ Partial | ❌ No | ✅ Yes |
| **Multi-domain** | ✅ All | ⚠️ Libraries | ❌ Graphics only | ❌ Audio only |
| **Type safety** | ✅ Static | ❌ Dynamic | ✅ Static | ✅ Static |
| **Units** | ✅ Built-in | ❌ No | ❌ No | ❌ No |
| **Hot-reload** | ✅ Yes | ⚠️ Partial | ❌ No | ❌ No |
| **MLIR** | ✅ Yes | ❌ No | ❌ No | ❌ No |

---

## Migration Strategy

### Phase 1: Rename (1 day)

**Repository**:
```bash
gh repo rename kairo
git remote set-url origin https://github.com/scottsen/morphogen.git
mv ~/src/projects/creative-computation-dsl ~/src/projects/kairo
```

**Python Package**:
```bash
mv creative_computation kairo
# Update setup.py, imports
```

**TIA Integration**:
```bash
mv ~/.tia/projects/creative-computation-dsl.yaml ~/.tia/projects/morphogen.yaml
# Update config fields
```

### Phase 2: Documentation (2 days)

**Preserve History**:
```bash
mkdir docs/legacy
cp SPECIFICATION.md docs/legacy/CCDSL_v0.2.2_SPECIFICATION.md
cp MVP.md docs/legacy/CCDSL_v0.2.2_MVP.md
```

**New Docs**:
- Copy `KAIRO_v0.3.1_SPECIFICATION.md` to repo
- Update `README.md`
- Create `MIGRATION.md` (v0.2.2 → v0.3.1 guide)

### Phase 3: Frontend Updates (1 week)

**Parser Updates**:
- Add `flow` keyword (remove `step`)
- Add `@state` decorator
- Add `fn` keyword
- Add lambda syntax `|args| expr`
- Add `struct` keyword
- Update `use` statement handling

**Type System**:
- Keep existing Field2D, Field3D, Agents, Signal
- Remove Flow<T> wrapper (internal implementation detail)

### Phase 4: Runtime Implementation (6 weeks)

**Week 1-2: Core Runtime**
- `flow()` scheduler
- `@state` management (double-buffering)
- Explicit RNG (Philox)

**Week 3-4: Field Dialect**
- Field data structure
- Basic ops (map, combine)
- PDE ops (diffuse, advect, project)

**Week 5: Visual Dialect**
- Colorization
- Display window
- Frame output

**Week 6: Polish**
- Error messages
- Examples
- Testing

---

## Implementation Roadmap

### MVP Timeline: 8 Weeks

**Phase 1: Frontend (2 weeks)** ✅ Mostly done! Just need updates
- Update parser for v0.3.1 syntax
- Add function/lambda support
- Add if/else expressions

**Phase 2: Core Runtime (2 weeks)**
- Flow scheduler
- State management (double-buffer)
- RNG (Philox)

**Phase 3: Field Dialect (2 weeks)**
- Field2D data structure
- Element-wise ops (map, combine)
- PDE operations (diffuse, advect, project)

**Phase 4: Visual + Polish (2 weeks)**
- Colorization
- Display window (pygame/similar)
- 3-4 working examples
- Basic documentation

### Post-MVP: v0.3.x → v0.4.0

**v0.3.2** (Month 3):
- Agent dialect
- Profile system
- More examples

**v0.3.3** (Month 4):
- Signal dialect
- Audio output
- Performance optimization

**v0.4.0** (Month 5-6):
- Space abstraction
- Streaming I/O
- Generic types
- Error handling

---

## Success Criteria

### MVP Complete When:

1. ✅ **3+ Examples Work**:
   - Diffusion (heat equation)
   - Smoke simulation (Navier-Stokes)
   - Reaction-diffusion (Gray-Scott)

2. ✅ **Determinism Verified**:
   - Same code + same seed = bitwise identical results
   - Across multiple runs
   - Test suite validates

3. ✅ **Documentation Complete**:
   - README with quickstart
   - Full specification (done!)
   - API reference
   - Example gallery

4. ✅ **Performance Acceptable**:
   - 256×256 diffusion: >30 FPS
   - 256×256 smoke: >15 FPS
   - Interactive visualization

5. ✅ **Clean Architecture**:
   - MLIR lowering working
   - Profile system functional
   - Hot-reload working (nice-to-have)

---

## Risk Assessment

### Low Risk ✅

- **Frontend changes** - Parser updates are straightforward
- **Rename** - No runtime to break
- **Branding** - Morphogen is a better name
- **Specification** - Complete and detailed

### Medium Risk ⚠️

- **Timeline** - 8 weeks is ambitious but achievable
- **MLIR complexity** - Lowering may take longer than expected
- **Performance** - May need optimization iteration

### Mitigation

- **Start with Python runtime** - Prove semantics before MLIR
- **Incremental implementation** - Get one example working first
- **Profile early** - Identify bottlenecks quickly

---

## Cross-References

### Related Projects

**RiffStack** (`~/src/projects/riffstack`):
- Update README to reference `kairo` (not creative-computation-dsl)
- Keep as separate project (audio-focused)

**TiaCAD** (`~/src/projects/tiacad`):
- Architectural inspiration
- Share registry patterns

### TIA Integration

**Beth Topics**:
```yaml
beth_topics:
  - kairo
  - dsl
  - creative-computation
  - simulation
  - determinism
  - mlir
```

**Project Config** (`~/.tia/projects/morphogen.yaml`):
```yaml
project_id: kairo
name: Morphogen
type: dsl
status: foundation-complete
github: https://github.com/scottsen/kairo
description: A language of creative determinism for simulation, sound, and visual form
```

---

## Next Actions

### Immediate (This Session)
1. ✅ Draft Morphogen v0.3.1 specification (DONE)
2. ✅ Create summary document (DONE)
3. ⏸️ Get approval from user

### Next Session (If Approved)
1. Rename repository and package
2. Copy specification to repo
3. Update README and docs
4. Update cross-references (RiffStack, TIA config)
5. Begin parser updates

---

## Conclusion

**Morphogen v0.3.1 represents the best evolution path**:

- ✅ **Better semantics** than v0.2.2
- ✅ **More complete** than ChatGPT v0.3.0
- ✅ **Perfect timing** (no runtime to break)
- ✅ **Better branding** (unique, memorable name)
- ✅ **Clear positioning** (complements RiffStack)
- ✅ **Implementable** (8-week MVP timeline)
- ✅ **Well-specified** (complete documentation)

**Recommendation**: Proceed with rename and implementation.

---

**Document**: `/home/scottsen/src/tia/sessions/vebabe-1106/KAIRO_v0.3.1_SUMMARY.md`
**Specification**: `/home/scottsen/src/tia/sessions/vebabe-1106/KAIRO_v0.3.1_SPECIFICATION.md`
**Generated**: 2025-11-06
