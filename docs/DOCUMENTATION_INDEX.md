# Morphogen Documentation Index

**Generated**: 2025-11-21
**Updated**: 2025-11-21
**Purpose**: Comprehensive index of all documentation with structure and navigation

> 💡 **Tip**: Use `./scripts/reveal.sh` to explore any document incrementally:
> - Level 0: Metadata (size, lines, type)
> - Level 1: Structure (headings, functions, classes) — **Most useful for initial exploration**
> - Level 2: Preview (sample content) — **Good for getting a feel for the content**
> - Level 3: Full content — **Use when you need complete details**
>
> **Token savings**: Level 1 = ~5-10% of tokens | Level 2 = ~20-30% | Level 3 = 100%

## Root-Level Documentation

### README.md (1,331 lines, 45KB)
**Purpose**: Project overview, getting started, domain catalog
**Key Sections**:
- Why Morphogen Exists
- Two Surfaces, One Kernel (Morphogen.Audio + RiffStack)
- Cross-Domain in Action
- Quick Start & Installation
- Language Overview (temporal model, state management, deterministic RNG)
- 15 Domain Catalogs (Field, RigidBody, Agent, Audio, Graph, Signal, etc.)
- Examples (Fluid Simulation, Reaction-Diffusion)
- Ecosystem Vision & Professional Applications
- Sister Project: Philbrick

**Navigate**: `./scripts/reveal.sh 1 README.md`

---

### SPECIFICATION.md (2,282 lines, 54KB)
**Purpose**: Complete Morphogen language specification
**Key Sections**:
- Lexical structure (keywords, operators, literals)
- Type system (primitives, composites, physical units)
- Temporal programming model (`flow` blocks, `@state` declarations)
- Domain system (`use` statements, cross-domain composition)
- Operators across all domains
- Execution model & determinism

**Navigate**: `./scripts/reveal.sh 1 SPECIFICATION.md`

---

### STATUS.md (1,096 lines, 45KB)
**Purpose**: Current implementation status by domain
**Key Sections**:
- Version history (v0.1 - v0.11.0)
- Domain implementation status (40+ domains)
- Test coverage statistics (900+ tests)
- Known issues & future work

**Navigate**: `./scripts/reveal.sh 1 STATUS.md`

---

### ECOSYSTEM_MAP.md (703 lines, 23KB)
**Purpose**: Map of all domains and roadmap
**Key Sections**:
- Complete domain catalog (organized by category)
- Implementation phases
- Cross-domain integration patterns
- Future expansion plans

**Navigate**: `./scripts/reveal.sh 1 ECOSYSTEM_MAP.md`

---

### CHANGELOG.md (2,875 lines, 129KB)
**Purpose**: Detailed version history and release notes
**Key Sections**:
- Version-by-version changes (v0.1.0 through v0.11.0)
- Feature additions, bug fixes, breaking changes
- Migration guides between versions

**Navigate**: `./scripts/reveal.sh 2 CHANGELOG.md` (large file - use preview)

---

## Documentation Directories

### docs/README.md (207 lines, 14KB)
**Purpose**: Documentation navigation guide
**Structure**:
- Quick Start links
- Documentation by category (Philosophy, Architecture, Specifications, etc.)
- Finding what you need (task-based navigation)
- Recent changes

**Navigate**: `./scripts/reveal.sh 1 docs/README.md`

---

## Architecture Documentation (7 documents)

### docs/architecture/overview.md
**Purpose**: System architecture overview
**Topics**: Kernel, frontends, Graph IR, MLIR compilation, execution model

### docs/architecture/domain-architecture.md (2,266 lines, 110KB)
**Purpose**: Deep technical vision for all domains
**Coverage**: 20+ domain specifications with operators, types, integration patterns

### docs/architecture/dsl-framework-design.md
**Purpose**: Vision for domain reasoning language
**Topics**: First-class domains, translations, composition patterns

### docs/architecture/continuous-discrete-semantics.md
**Purpose**: Dual computational models (continuous vs discrete)

### docs/architecture/gpu-mlir-principles.md
**Purpose**: GPU execution and MLIR integration strategy

### docs/architecture/interactive-visualization.md
**Purpose**: Visualization approach and capabilities

**Explore**: `find docs/architecture -name "*.md" -exec scripts/reveal.sh 1 {} \;`

---

## Specifications (27 documents)

### Domain Specifications

| Domain | File | Lines | Focus |
|--------|------|-------|-------|
| Chemistry | chemistry.md | 1,002 | Molecular dynamics, kinetics, thermodynamics |
| Circuit | circuit.md | 1,136 | SPICE-like circuit simulation |
| Geometry | geometry.md | 3,000+ | CAD/parametric geometry |
| Physics | physics-domains.md | - | Fluid, thermal, combustion |
| Procedural Gen | procedural-generation.md | - | Noise, terrain, emergence |
| Audio | ambient-music.md | - | Compositional audio synthesis |
| Video/Audio | video-audio-encoding.md | - | Encoding and synchronization |
| Timbre | timbre-extraction.md | - | Instrument modeling |

### Infrastructure Specifications

| Specification | File | Purpose |
|--------------|------|---------|
| Graph IR | graph-ir.md | Frontend-kernel boundary |
| MLIR Dialects | mlir-dialects.md | Custom MLIR dialects (6 dialects) |
| Type System | type-system.md | Complete type system spec |
| Level 3 Types | level-3-type-system.md | Cross-domain type safety |
| Operator Registry | operator-registry.md | 500+ operator catalog |
| Scheduler | scheduler.md | Multirate deterministic scheduling |
| Transform | transform.md | Transform operators |
| Transform Comp | transform-composition.md | Composable transforms |
| Profiles | profiles.md | Execution profiles (strict/repro/live) |
| Units | units.md | Physical unit system |
| Snapshot ABI | snapshot-abi.md | State serialization |

**Explore**: `scripts/reveal.sh 1 docs/specifications/chemistry.md`

---

## Guides (5 documents)

### docs/guides/domain-implementation.md
**Purpose**: How to add new domains to Morphogen
**Topics**: Operator definition, registry integration, testing, documentation

### docs/guides/optimization-implementation.md
**Purpose**: Implementing optimization algorithms

### docs/guides/output-generation.md
**Purpose**: Generating visualizations and outputs

### docs/guides/DOMAIN_FINISHING_GUIDE.md
**Purpose**: Checklist for completing domain implementations

**Explore**: `scripts/reveal.sh 1 docs/guides/domain-implementation.md`

---

## Examples (6 documents)

### docs/examples/
- **emergence-cross-domain.md**: Emergent behavior from cross-domain coupling
- **j-tube-firepit-multiphysics.md**: Multiphysics combustion simulation
- **kerbal-space-program.md**: Orbital mechanics example
- **racing-ai-pipeline.md**: AI agent racing simulation
- **ambient-music-pipelines.md**: Generative music examples

**Explore**: `scripts/reveal.sh 1 docs/examples/emergence-cross-domain.md`

---

## Use Cases (2 documents)

### docs/use-cases/2-stroke-muffler-modeling.md
Complete example: Fluid dynamics → Acoustics → Audio synthesis

### docs/use-cases/chemistry-unified-framework.md
Chemistry simulation framework across multiple domains

---

## Reference Documentation (18 documents)

### Operator Catalogs
- **operator-registry-expansion.md**: Complete operator registry
- **procedural-operators.md**: Procedural generation operators
- **emergence-operators.md**: Emergence domain operators
- **genetic-algorithm-operators.md**: GA operators
- **optimization-algorithms.md**: Optimization operators

### Visualization & Sonification
- **visualization-ideas-by-domain.md** (56KB): Comprehensive visualization catalog
- **advanced-visualizations.md**: Advanced visualization techniques
- **audio-visualization-ideas.md**: Audio-reactive visuals
- **visual-domain-quickref.md**: Visual domain quick reference

### Theoretical Frameworks
- **universal-domain-frameworks.md** (28KB): Mathematical foundations
- **math-transformation-metaphors.md** (25KB): Intuitive transform explanations
- **mathematical-music-frameworks.md**: Music theory foundations

### Patterns & Best Practices
- **multiphysics-success-patterns.md** (30KB): 12 battle-tested patterns
- **time-alignment-operators.md**: Temporal synchronization

**Explore**: `scripts/reveal.sh 1 docs/reference/multiphysics-success-patterns.md`

---

## Architecture Decision Records (12 ADRs)

Located in `docs/adr/`:

| ADR | Title | Key Decision |
|-----|-------|--------------|
| 001 | Unified Reference Model | Cross-domain anchor system |
| 002 | Cross-Domain Patterns | Integration patterns |
| 003 | Circuit Modeling | SPICE-like domain design |
| 004 | Instrument Modeling | Physical modeling synthesis |
| 005 | Emergence Domain | Emergent behavior framework |
| 006 | Chemistry Domain | Molecular simulation approach |
| 007 | GPU-First Domains | GPU acceleration strategy |
| 008 | Procedural Generation | Noise/terrain domain design |
| 009 | Ambient Music & Generative | Generative music framework |
| 010 | Ecosystem Branding | Morphogen/Philbrick naming |
| 011 | Project Renaming | Kairo → Morphogen transition |
| 012 | Universal Domain Translation | Cross-domain translation patterns |

**Explore**: `scripts/reveal.sh 1 docs/adr/001-unified-reference-model.md`

---

## Philosophy Documentation (5 documents)

### docs/philosophy/
- **formalization-and-knowledge.md**: How formalization transforms knowledge
- **universal-dsl-principles.md**: Design philosophy (8 principles)
- **operator-foundations.md**: Mathematical operator theory
- **categorical-structure.md**: Category-theoretic formalization

**Explore**: `scripts/reveal.sh 1 docs/philosophy/formalization-and-knowledge.md`

---

## Roadmap & Planning

### docs/roadmap/
- **language-features.md**: Path to language 1.0 (finalized vs planned features)
- **mvp.md**: MVP roadmap
- **v0.1.md**: Version 0.1 roadmap
- **implementation-progress.md**: Current implementation status
- **testing-strategy.md**: Testing approach

### docs/planning/
- **ROADMAP_2025_Q4.md**: Q4 2025 execution plan
- **SHOWCASE_OUTPUT_STRATEGY.md**: Demo strategy
- **KAIRO_2.0_STRATEGIC_ANALYSIS.md**: Strategic analysis

---

## Analysis & Reports

### docs/analysis/
- **DOMAIN_VALIDATION_REPORT.md**: Implementation vs documentation alignment
- **KAIRO_RENAME_ANALYSIS.md**: Rename impact analysis (5,344 occurrences across 356 files)
- **AGENTS_DOMAIN_ANALYSIS.md**: Agent domain technical analysis
- **CROSS_DOMAIN_IMPLEMENTATION_SUMMARY.md**: Cross-domain integration summary

---

## Claude Context

### claude.md (258 lines, 9KB)
**Purpose**: Context file for Claude AI assistant
**Contents**:
- Project overview and capabilities
- Key file locations
- reveal tool usage guide
- Development workflow
- Common tasks and examples

**Navigate**: `scripts/reveal.sh 1 claude.md`

---

## Scripts & Tools

### scripts/reveal.sh
**Purpose**: Progressive file explorer (wrapper)
**Usage**: `./scripts/reveal.sh <level> <file>`
**Levels**: 0=metadata, 1=structure, 2=preview, 3=full

### scripts/reveal.py
**Purpose**: Local Python implementation of reveal tool
**Features**: Metadata, structure extraction, preview, full content with line numbers

### scripts/gh.py
**Purpose**: GitHub issue/PR management without gh CLI

**Documentation**: See `scripts/README.md`

---

## Using the Reveal Tool

### Quick Examples

```bash
# Explore README structure
./scripts/reveal.sh 1 README.md

# Preview chemistry specification
./scripts/reveal.sh 2 docs/specifications/chemistry.md

# View full guide with line numbers
./scripts/reveal.sh 3 docs/guides/domain-implementation.md

# Check metadata of large file
./scripts/reveal.sh 0 CHANGELOG.md
```

### Exploring Documentation Systematically

```bash
# List all specifications with structure
for f in docs/specifications/*.md; do
    echo "=== $f ==="
    ./scripts/reveal.sh 1 "$f" | head -40
    echo ""
done

# Find all architecture docs
find docs/architecture -name "*.md" -exec scripts/reveal.sh 0 {} \;

# Preview all guides
for f in docs/guides/*.md; do
    ./scripts/reveal.sh 2 "$f"
done
```

---

## Documentation Statistics

**Root Documentation**: 5 major files (141KB total)
**Architecture**: 7 documents
**Specifications**: 27 documents
**Guides**: 5 documents
**Examples**: 6 documents
**Reference**: 18 documents
**ADRs**: 12 decision records
**Philosophy**: 5 documents
**Analysis**: 7 reports

**Total**: ~90 documentation files covering all aspects of the Morphogen platform

---

## Navigation Tips

1. **Start with README.md** for project overview
2. **Read docs/README.md** for documentation navigation
3. **Use reveal tool** to explore incrementally:
   - `./scripts/reveal.sh 1 <file>` — Structure first (recommended starting point)
   - `./scripts/reveal.sh 2 <file>` — Preview sample content
   - Only read full docs when you need complete details
4. **Check SPECIFICATION.md** for language reference
5. **Browse docs/specifications/** for domain details
6. **Read ADRs** to understand key decisions
7. **Use DOCUMENTATION_INDEX.md** (this file) as your map

**Efficient exploration workflow:**
```bash
# Survey a directory (structure of all files)
for f in docs/specifications/*.md; do
    ./scripts/reveal.sh 1 "$f" | head -40
done

# Preview interesting files
./scripts/reveal.sh 2 docs/guides/domain-implementation.md

# Read full content only when needed
./scripts/reveal.sh 3 SPECIFICATION.md
```

---

**Generated**: 2025-11-21
**Updated**: 2025-11-21
**Tool**: `scripts/reveal.py` and custom documentation mapping
**Maintenance**: Regenerate when documentation structure changes significantly

**Related Resources:**
- [docs/README.md](README.md) — Navigation guide with task-based finding
- [scripts/README.md](../scripts/README.md) — Complete reveal tool documentation
- [claude.md](../claude.md) — Claude AI assistant context and quick reference
- [../README.md](../README.md) — Project overview and getting started
