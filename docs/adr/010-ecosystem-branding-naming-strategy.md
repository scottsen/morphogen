# ADR-010: Ecosystem Branding & Naming Strategy

**Status:** Accepted (Domain Layer Naming) / Superseded (Project Naming - see ADR-011)
**Date:** 2025-11-17
**Authors:** Architecture Review
**Decision:** Establish consistent naming conventions across platform's layered architecture
**Related:** [ADR-011: Project Renaming - Morphogen & Philbrick](011-project-renaming-morphogen-philbrick.md)

---

## Note: Project Naming Decision

**This ADR originally proposed naming conventions for "Morphogen" but the project has been renamed.**

**See [ADR-011](011-project-renaming-morphogen-philbrick.md) for:**
- Digital platform: **Morphogen → Morphogen** (honors Turing's morphogenesis)
- Analog platform: **Philbrick** (honors George A. Philbrick)
- Modules: **Philbricks** (composable function blocks)

**This ADR (010) remains valid for:**
- Domain library naming rules (single-word lowercase)
- Layer 3 kernel namespace (`morphogen.internal`)
- Domain renames (rigidbody→rigid, sparse_linalg→linalg, etc.)

**Update all references below:**
- "Morphogen" → "Morphogen" (for digital platform)
- Add "Philbrick" (for analog platform)

---

---

## Context

Morphogen (formerly Morphogen) has evolved from a single-domain DSL (Creative Computation DSL) into a multi-domain platform with:
- 23+ domain libraries (field, agent, audio, visual, rigid body, graph, signal, etc.)
- Multiple abstraction layers (kernel, domains, frontends)
- Two user-facing surfaces (Morphogen.Audio, RiffStack)
- Sister project: Philbrick (analog modular computation platform)
- Growing ecosystem of cross-domain capabilities

As the platform scales, inconsistent naming creates friction:
- Mix of styles: `field` vs `rigidbody` vs `sparse_linalg` vs `statemachine`
- Unclear layer membership: Is this kernel? Domain library? User surface?
- No clear pattern for new domains: What should "chemistry" be named?
- Python namespace conflicts: Where do kernel primitives live vs domains?

This ADR establishes a coherent naming strategy that:
1. Makes the architecture visible through naming
2. Scales to 50+ domains without confusion
3. Guides contributors adding new domains
4. Creates professional, memorable branding

**Note:** Project-level naming (Morphogen, Philbrick) is covered in ADR-011.

---

## Investigation

### Current State Analysis

#### What Exists (November 2025)

**User-Facing Surfaces:**
- `Morphogen.Audio` - Compositional audio DSL ✅ (formerly Morphogen.Audio)
- `RiffStack` - Live performance environment ✅ (separate brand, working)

**Domain Libraries (morphogen/stdlib/):**
```
field.py            ✅ Good: single word, lowercase
agent.py            ✅ Good: single word, lowercase
audio.py            ✅ Good: single word, lowercase
visual.py           ✅ Good: single word, lowercase
rigidbody.py        ⚠️ Inconsistent: compound word, should be 'rigid'
sparse_linalg.py    ⚠️ Inconsistent: underscore, should be 'linalg'
statemachine.py     ⚠️ Inconsistent: compound word, should be 'state'
io_storage.py       ⚠️ Inconsistent: underscore, should be 'io'
integrators.py      ⚠️ Inconsistent: plural, should be 'integrator'
optimization.py     ⚠️ Inconsistent: noun form, should be 'optimize'
acoustics.py        ⚠️ Inconsistent: plural, should be 'acoustic'
cellular.py         ✅ Good: single word, lowercase
noise.py            ✅ Good: single word, lowercase
color.py            ✅ Good: single word, lowercase
image.py            ✅ Good: single word, lowercase
palette.py          ✅ Good: single word, lowercase
terrain.py          ✅ Good: single word, lowercase
graph.py            ✅ Good: single word, lowercase
signal.py           ✅ Good: single word, lowercase
vision.py           ✅ Good: single word, lowercase
neural.py           ✅ Good: single word, lowercase
genetic.py          ✅ Good: single word, lowercase
```

**Score: 14/23 (61%) follow ideal pattern**

**Kernel Components (currently scattered):**
- Type system: `Stream`, `Field`, `Evt`, `Agents` in various modules
- Scheduler: Mixed between runtime and compiler
- Transform dialect: Partially in domains, partially kernel
- Operator registry: In specifications, not implemented
- No clear namespace boundary

#### Architecture Review

From ARCHITECTURE.md, ECOSYSTEM_MAP.md, and related docs:

**Three Clear Layers:**
1. **Kernel** - Types, scheduler, transforms, registry, MLIR (foundation)
2. **Domain Libraries** - field, agent, audio, physics, etc. (optional capabilities)
3. **User Surfaces** - Morphogen.Audio, RiffStack, future DSLs (human-friendly)

**Key Insight:** Naming should make layer membership obvious.

#### Naming Patterns in Successful Projects

**Unix Philosophy (single-word commands):**
- `grep`, `awk`, `sed`, `vim`, `git`
- Principle: Short, memorable, lowercase

**Python Ecosystem (namespaces):**
- `numpy.linalg`, `scipy.optimize`, `torch.nn`
- Principle: Hierarchical namespaces for organization

**Domain-Specific Languages:**
- `React.Component`, `Vue.defineComponent`
- Principle: Dot-notation for framework exports

**Design System Best Practice:**
- Clear layer visibility
- Consistent patterns (all domains follow same rules)
- Scalable (easy to add new domains)

---

## Decision

### Naming System: Three-Layer Strategy

#### Layer 1: Platform & User Surfaces

**Pattern:** `Morphogen.X` for user-facing DSLs (updated from Morphogen per ADR-011)

```
Morphogen                # Platform brand (formerly Morphogen)
├── Morphogen.Audio      # Compositional audio DSL (implemented)
├── Morphogen.Physics    # Future: Physics simulation DSL
├── Morphogen.Visual     # Future: Visual composition DSL
└── RiffStack            # Performance environment (separate brand)
```

**Rules:**
- Platform name is always capitalized: `Morphogen`
- User surfaces use dot-notation: `Morphogen.X`
- RiffStack remains separate brand (dual-surface strategy)

**Rationale:** Clear signal that this is a "language" or "environment", not a library.

**Note:** See ADR-011 for why "Morphogen" (Turing's morphogenesis) was chosen.

---

#### Layer 2: Domain Libraries

**Pattern:** Single-word, lowercase, noun form

**Proposed Taxonomy (6 tiers):**

**Tier 1: Core Computational Primitives**
```
field        # Dense grid operations
agent        # Sparse particle systems
stream       # Time-series operations
event        # Event sequences
```

**Tier 2: Physical Simulation**
```
fluid        # Fluid dynamics
thermal      # Heat transfer
rigid        # Rigid body physics (renamed from rigidbody)
acoustic     # Acoustics (renamed from acoustics)
circuit      # Circuit simulation
optics       # Optics (future)
```

**Tier 3: Signal & Media Processing**
```
audio        # Audio synthesis & DSP
visual       # Rendering & composition
signal       # Signal processing
image        # Image processing
video        # Video processing (future)
color        # Color operations
```

**Tier 4: Mathematical & Computational**
```
linalg       # Linear algebra (renamed from sparse_linalg)
optimize     # Optimization (renamed from optimization)
graph        # Graph/network analysis
neural       # Neural networks
symbolic     # Symbolic math (future)
```

**Tier 5: Generative & Procedural**
```
noise        # Noise generation
terrain      # Terrain generation
fractal      # Fractal generation
palette      # Color palettes
cellular     # Cellular automata
```

**Tier 6: System & Infrastructure**
```
io           # I/O and storage (renamed from io_storage)
state        # State machines (renamed from statemachine)
integrator   # Time integrators (renamed from integrators)
```

**Naming Rules:**
1. **Single word** - No compounds (`rigid` not `rigidbody`)
2. **Lowercase** - Unix convention (`linalg` not `LinAlg`)
3. **No underscores** - Avoid `sparse_linalg`, use `linalg`
4. **Noun form** - Domain objects, not actions
5. **Singular** - `integrator` not `integrators`
6. **Domain-first** - What domain, not implementation (`linalg` not `sparse_linalg`)

**Rationale:**
- Consistent, predictable pattern
- Easy to type in `use field, agent, audio`
- Scales to 100+ domains
- Professional, Unix-like aesthetic
- Discoverable (users can guess names)

---

#### Layer 3: Kernel (Internal)

**Pattern:** `morphogen.internal` namespace for kernel primitives (updated from kairo per ADR-011)

**Proposed Structure:**
```python
morphogen/
├── __init__.py              # Top-level exports
├── internal/                # Kernel implementation
│   ├── types.py            # Stream, Field, Evt, Agents
│   ├── scheduler.py        # Wiener Scheduler (multirate scheduling)
│   ├── transform.py        # Transform dialect
│   ├── registry.py         # Operator registry
│   ├── profiles.py         # Determinism profiles
│   └── snapshot.py         # Snapshot ABI
├── stdlib/                  # Domain libraries
│   ├── field.py
│   ├── agent.py
│   └── ...
├── compiler/                # MLIR backend
│   ├── dialects/
│   └── passes/
└── frontends/               # DSL surfaces
    ├── audio/              # Morphogen.Audio
    └── riffstack/          # RiffStack
```

**Import Patterns:**

**Beginner (Domain-level only):**
```python
from morphogen.stdlib import field, agent, audio
# Never sees "internal"
```

**Intermediate (Types + Domains):**
```python
from morphogen import Stream, Field       # Exported at top level
from morphogen.stdlib import field, audio
```

**Advanced (Kernel access):**
```python
from morphogen.internal import scheduler, transform, registry
from morphogen.stdlib import field, audio
```

**Rationale:**
- `internal` clearly signals "kernel/advanced"
- Matches Python convention (like `_internal`)
- Selective top-level exports for common types
- Clear boundary: `stdlib` = user domains, `internal` = kernel

---

### Specific Renames Required

| Current | Proposed | Rationale | Breaking Change? |
|---------|----------|-----------|------------------|
| `rigidbody` | `rigid` | Shorter, noun form, follows pattern | Yes (minor) |
| `sparse_linalg` | `linalg` | Remove underscore, domain-first naming | Yes (minor) |
| `statemachine` | `state` | Shorter, follows pattern | Yes (minor) |
| `integrators` | `integrator` | Singular noun form | Yes (minor) |
| `io_storage` | `io` | Remove underscore, shorter | Yes (minor) |
| `optimization` | `optimize` | Verb form, shorter | Yes (minor) |
| `acoustics` | `acoustic` | Singular form | Yes (minor) |

**Migration Strategy:** Aliases during transition (see Implementation Plan)

---

### Core Type Names

**Keep as-is** - These are well-designed:
```python
Stream<T, Domain, Rate>  # Time-varying signals ✅
Field<T, Space>          # Spatial data ✅
Evt<A>                   # Events ✅
Agents<T>                # Particle collections ✅
```

**Rationale:** PascalCase for types is standard, names are clear and distinctive.

---

## Implementation Plan

### Phase 1: Documentation (No Breaking Changes) - Week 1

**Goal:** Establish conventions in documentation without code changes.

**Tasks:**
1. Create `docs/reference/NAMING_CONVENTIONS.md`
2. Update README.md with clear layer hierarchy
3. Add "Domain Reference" showing all domains in tiers
4. Document import patterns for each user level
5. Update contribution guide with naming rules

**Deliverables:**
- [ ] NAMING_CONVENTIONS.md
- [ ] Updated README.md
- [ ] Updated CONTRIBUTING.md
- [ ] This ADR (010)

---

### Phase 2: Kernel Namespace (v0.11) - Weeks 2-3

**Goal:** Create `morphogen.internal` namespace and reorganize kernel code.

**Tasks:**
1. Create `morphogen/internal/` directory
2. Move kernel components to `internal/`:
   - `types.py` - Core type definitions
   - `scheduler.py` - Multirate scheduler
   - `transform.py` - Transform dialect
   - `registry.py` - Operator registry implementation
   - `profiles.py` - Determinism profiles
   - `snapshot.py` - Snapshot ABI
3. Update `morphogen/__init__.py` to export common types
4. Update all imports across codebase
5. Add tests for new import paths

**Deliverables:**
- [ ] `morphogen/internal/` structure
- [ ] Top-level exports in `morphogen/__init__.py`
- [ ] Updated imports (all files)
- [ ] Test coverage for new structure

---

### Phase 3: Domain Aliases (v0.11) - Week 4

**Goal:** Add aliases for renamed domains without breaking existing code.

**Implementation:**
```python
# morphogen/stdlib/__init__.py
from . import rigidbody
from . import rigidbody as rigid  # New alias

from . import sparse_linalg
from . import sparse_linalg as linalg  # New alias

# ... etc for all renames
```

**Documentation:**
```python
# In each renamed module:
"""
This module is being renamed:
  Old: rigidbody
  New: rigid

Both imports work in v0.11:
  from morphogen.stdlib import rigidbody  # Deprecated
  from morphogen.stdlib import rigid      # Preferred

The old name will be removed in v1.0.
"""
```

**Tasks:**
1. Add aliases for all 7 renames
2. Update examples to use new names
3. Add deprecation warnings (soft, not errors)
4. Update documentation to prefer new names

**Deliverables:**
- [ ] Aliases in `stdlib/__init__.py`
- [ ] Deprecation notices in docstrings
- [ ] Examples updated to new names
- [ ] Documentation uses new names exclusively

---

### Phase 4: Hard Renames (v1.0) - Future

**Goal:** Complete migration to new names, remove old aliases.

**Tasks:**
1. Rename actual files:
   - `rigidbody.py` → `rigid.py`
   - `sparse_linalg.py` → `linalg.py`
   - etc.
2. Remove aliases from `__init__.py`
3. Update all imports (should be minimal after Phase 3)
4. Final documentation sweep

**Migration Guide:**
```markdown
# Migrating from v0.10 to v1.0

## Domain Renames

Update your imports:

```python
# Old (v0.10)
from morphogen.stdlib import rigidbody, sparse_linalg, statemachine

# New (v1.0)
from morphogen.stdlib import rigid, linalg, state
```

All functionality is identical, only names changed.
```

**Deliverables:**
- [ ] Renamed files
- [ ] Migration guide
- [ ] Updated changelog
- [ ] v1.0 release notes

---

## Consequences

### Positive

✅ **Clear Architecture** - Layer membership obvious from naming
✅ **Scalable** - Easy to add 50+ domains following pattern
✅ **Professional** - Consistent, Unix-like aesthetic
✅ **Discoverable** - Users can guess domain names
✅ **Memorable** - Short, distinctive names (`rigid`, `linalg`, `optimize`)
✅ **Distinctive** - "Morphogen.Audio" + domain composition is unique branding
✅ **Pythonic** - Follows Python ecosystem conventions

### Negative

⚠️ **Breaking Changes** - 7 domain renames (mitigated by aliases)
⚠️ **Migration Effort** - Users must update imports (but aliases ease transition)
⚠️ **Documentation Debt** - Must update all docs/examples (necessary anyway)

### Neutral

◼️ **Learning Curve** - New users learn one consistent pattern vs many inconsistent ones
◼️ **Community Impact** - Early-stage project, small user base, low impact

---

## Alternatives Considered

### Alternative 1: Keep Current Naming (Status Quo)

**Pros:**
- No breaking changes
- No migration effort

**Cons:**
- Inconsistency increases as domains grow
- No clear guidance for new domains
- Professional polish suffers
- Technical debt accumulates

**Verdict:** Rejected - Problem gets worse over time, better to fix early.

---

### Alternative 2: Aggressive Namespacing (morphogen.domain.X)

```python
from morphogen.domain.field import *
from morphogen.domain.agent import *
from morphogen.kernel.types import Stream
```

**Pros:**
- Maximum clarity
- Easy to find things

**Cons:**
- Verbose imports
- Not ergonomic for DSL use case (`use field` > `use morphogen.domain.field`)
- Overthinking the problem

**Verdict:** Rejected - Too heavy for benefit.

---

### Alternative 3: No Kernel Namespace (Keep Scattered)

Just improve domain naming, leave kernel code where it is.

**Pros:**
- Less refactoring
- Simpler initial structure

**Cons:**
- Kernel code remains scattered
- No clear boundary for advanced users
- Harder to document "what's kernel vs domain"

**Verdict:** Rejected - `morphogen.internal` provides clarity worth the effort.

---

### Alternative 4: Different Domain Taxonomy

Group domains differently (e.g., `morphogen.physics.*`, `morphogen.media.*`).

**Pros:**
- Logical groupings
- Namespace organization

**Cons:**
- Arbitrary boundaries (is `audio` media or signal?)
- More verbose
- Harder to remember categories

**Verdict:** Rejected - Flat namespace with tier docs is simpler.

---

## Cross-Domain Composition Branding

This naming strategy supports Morphogen's **killer differentiator** - cross-domain composition.

**Marketing Examples:**

```morphogen
# Three domains, one program, zero glue code
use fluid, acoustic, audio

let sound = fluid.simulate(engine_pulse)
           |> acoustic.propagate(waveguide)
           |> audio.synthesize(mic_position)
```

```morphogen
# Circuit → Audio synthesis (impossible elsewhere)
use circuit, audio

let pedal_response = circuit.analyze(tube_screamer)
let tone = audio.apply_circuit(guitar_signal, pedal_response)
```

**Taglines:**
- "Circuit → Audio. Fluid → Acoustic. Geometry → Fields."
- "Three domains. One program. Zero glue code."
- "Morphogen: Where domains compose."

---

## Documentation Impact

### New Documents Required

1. **docs/reference/NAMING_CONVENTIONS.md** - This strategy documented for contributors
2. **docs/reference/DOMAIN_CATALOG.md** - All domains organized by tier
3. **docs/guides/IMPORT_PATTERNS.md** - How to import for different user levels

### Updates Required

1. **README.md** - Add clear layer hierarchy visualization
2. **ARCHITECTURE.md** - Reference `morphogen.internal` namespace
3. **ECOSYSTEM_MAP.md** - Use new domain names throughout
4. **All example files** - Update to new import patterns
5. **CONTRIBUTING.md** - Add domain naming rules

---

## Success Metrics

**After Phase 2 (Kernel Namespace):**
- [ ] All kernel code in `morphogen.internal/`
- [ ] Clear import patterns documented
- [ ] Zero confusion about "what's kernel vs domain"

**After Phase 3 (Domain Aliases):**
- [ ] All 23 domains follow naming pattern (via aliases)
- [ ] All examples use new names
- [ ] Zero breaking changes for users

**After Phase 4 (v1.0):**
- [ ] Clean codebase with consistent naming
- [ ] New contributors follow pattern naturally
- [ ] Professional polish evident in naming

---

## References

**Architecture Documents:**
- [ARCHITECTURE.md](../../ARCHITECTURE.md) - Morphogen Stack architecture
- [ECOSYSTEM_MAP.md](../../ECOSYSTEM_MAP.md) - Complete domain map
- [ROADMAP.md](../../ROADMAP.md) - Development roadmap

**Related ADRs:**
- [ADR-001: Unified Reference Model](001-unified-reference-model.md)
- [ADR-002: Cross-Domain Architectural Patterns](002-cross-domain-architectural-patterns.md)

**Implementation Resources:**
- [Domain Implementation Guide](../guides/domain-implementation.md)
- [Domain Architecture](../architecture/domain-architecture.md)

---

## Appendix A: Complete Domain Catalog (Proposed)

### Implemented Domains (v0.10)

| Domain | Status | Tier | Lines of Code | Rename? |
|--------|--------|------|---------------|---------|
| `field` | ✅ Production | 1 | 450 | No |
| `agent` | ✅ Production | 1 | 380 | No |
| `audio` | ✅ Production | 3 | 890 | No |
| `visual` | ✅ Production | 3 | 320 | No |
| `rigid` | ✅ Production | 2 | 410 | Yes (from rigidbody) |
| `cellular` | ✅ Production | 5 | 180 | No |
| `graph` | ✅ Production | 4 | 520 | No |
| `signal` | ✅ Production | 3 | 470 | No |
| `terrain` | ✅ Production | 5 | 390 | No |
| `vision` | ✅ Production | 3 | 580 | No |
| `state` | ✅ Production | 6 | 280 | Yes (from statemachine) |
| `noise` | ✅ Production | 5 | 195 | No |
| `color` | ✅ Production | 3 | 140 | No |
| `image` | ✅ Production | 3 | 230 | No |
| `palette` | ✅ Production | 5 | 125 | No |
| `linalg` | ✅ Production | 4 | 340 | Yes (from sparse_linalg) |
| `optimize` | ✅ Production | 4 | 450 | Yes (from optimization) |
| `neural` | ✅ Production | 4 | 380 | No |
| `genetic` | ✅ Production | 4 | 290 | No |
| `integrator` | ✅ Production | 6 | 210 | Yes (from integrators) |
| `io` | ✅ Production | 6 | 180 | Yes (from io_storage) |
| `acoustic` | ✅ Production | 2 | 160 | Yes (from acoustics) |
| `flappy` | 🚧 Example | - | 95 | No (example only) |

**Total: 23 domains, 7 renames needed**

### Planned Domains (Future)

| Domain | Tier | Specification Status | Priority |
|--------|------|---------------------|----------|
| `fluid` | 2 | Complete (1079 lines) | High |
| `thermal` | 2 | Complete (1079 lines) | High |
| `circuit` | 2 | Complete (1136 lines) | High |
| `chem` | 2 | Complete (2200+ lines) | Medium |
| `fractal` | 5 | Partial | Medium |
| `optics` | 2 | Not started | Low |
| `symbolic` | 4 | Not started | Low |
| `video` | 3 | Not started | Medium |

---

## Appendix B: Import Pattern Examples

### Beginner Level
```python
# Simple domain usage
from morphogen.stdlib import field, agent, audio

# Use domains directly
sim = field.diffuse(temperature, rate=0.1, dt=0.01)
particles = agent.integrate(particles, dt=0.01)
sound = audio.sine(freq=440)
```

### Intermediate Level
```python
# Types + domains
from kairo import Stream, Field, Evt
from morphogen.stdlib import field, agent, audio

# Create typed variables
temp: Field[float] = field.zeros(shape=(256, 256))
sig: Stream[float] = audio.sine(freq=440)
```

### Advanced Level
```python
# Kernel access
from morphogen.internal import scheduler, transform, registry
from morphogen.stdlib import field, audio

# Customize scheduler
scheduler.set_rates(audio=48000, control=1000)

# Use transform dialect
spectrum = transform.fft(audio_signal)
```

### Framework Developer Level
```python
# Full access
from morphogen.internal import *
from morphogen.compiler import dialects, passes
from morphogen.stdlib import field, audio

# Extend compiler
@passes.register_pass("custom_fusion")
def custom_fusion_pass(module):
    # Custom optimization logic
    pass
```

---

**Status:** Proposed
**Next Steps:** Review with maintainers, begin Phase 1 (documentation)
**Target:** v0.11 for kernel namespace, v1.0 for complete migration
