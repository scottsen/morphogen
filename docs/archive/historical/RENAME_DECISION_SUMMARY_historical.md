# The Morphogen & Philbrick Decision

**Date:** November 16, 2025
**Status:** Approved
**Full ADR:** [docs/adr/011-project-renaming-morphogen-philbrick.md](docs/adr/011-project-renaming-morphogen-philbrick.md)

---

## What We Decided

### **Kairo → Morphogen**
Digital temporal programming language renamed to honor **Alan Turing's morphogenesis** work.

### **Analog Platform → Philbrick**
Hardware platform named after **George A. Philbrick**, inventor of modular analog computing (1952).

### **Modules → Philbricks**
Individual function blocks called "Philbricks" (creates product category like LEGO bricks).

---

## Why These Names

### Morphogen
- **Turing's morphogenesis** (1952) = simple continuous rules → emergent patterns
- **Exactly what we do** = four primitives → complex cross-domain behavior
- **Unique positioning** = "emergence-focused continuous-time computation"
- **Educational value** = teaches Turing's often-overlooked biology work
- **Zero collision** = avoids Turing Award, Alan Turing Institute conflicts

### Philbrick
- **Historical accuracy** = we're literally reviving his 1952 modular analog computing vision
- **Perfect fit** = he invented what we're rebuilding
- **Educational legacy** = honors forgotten pioneer
- **"Philbricks"** = memorable product category
- **Zero collision** = unique, trademark-friendly

---

## The Pantheon Layer Naming

Both platforms honor inventors at each architectural layer:

### Morphogen (Digital)
```
Morphogen.Audio (user surface)
  ↓
Domain libraries (field, agent, audio...)
  ↓
Wiener Scheduler (Norbert Wiener - cybernetics)
Shannon Protocol (Claude Shannon - information theory)
Turing Core (Alan Turing - computation)
```

### Philbrick (Analog)
```
Moog Surface (Robert Moog - modular synthesis)
  ↓
Philbricks (composable modules)
  ↓
DeForest Layer (Lee de Forest - amplification)
Black Layer (Harold Black - negative feedback)
Shannon Bus (Claude Shannon - digital protocol)
Mead Processors (Carver Mead - neuromorphic)
```

---

## What Changes

### Immediate
- GitHub repo: `kairo` → `morphogen`
- Python package: `kairo` → `morphogen`
- All imports: `from kairo.stdlib` → `from morphogen.stdlib`
- All documentation updated
- New Philbrick repository created

### Timeline
- **Week 1:** Create Philbrick repo, move analog docs
- **Week 2:** Prepare migration guide, update docs
- **Week 3:** Execute Morphogen rename (v0.11.0)
- **Week 4:** Update branding, create bridge docs

---

## The Unified Vision

**Tagline:**
> **Morphogen** and **Philbrick**: Computational substrates for the continuous world

**Morphogen:**
> Where emergence meets engineering. Named after Turing's morphogenesis, Morphogen is a deterministic continuous-time language where simple primitives compose into complex patterns.

**Philbrick:**
> Modular analog computation. Named after George A. Philbrick, who invented modular analog computing in 1952. Philbricks are composable function blocks for continuous-time signal processing.

**Together:**
> Two projects, one vision. Simple primitives (sum, integrate, nonlinearity, events) compose into emergent complexity - in software and hardware.

---

## Next Steps

See [docs/adr/011-project-renaming-morphogen-philbrick.md](docs/adr/011-project-renaming-morphogen-philbrick.md) for:
- Complete rationale
- Alternatives considered
- Implementation plan
- Success metrics
- Rollback plan

---

**This is a major decision. Read the full ADR before proceeding.**
