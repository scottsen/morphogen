# Morphogen ↔ Philbrick Bridge

**Two Halves of One Vision**: Software and Hardware Duals

---

## Overview

**Morphogen** (formerly Morphogen) and **[Philbrick](https://github.com/scottsen/philbrick)** are sister projects that implement the same compositional vision in different substrates:

- **Morphogen** = Digital simulation of continuous-time phenomena
- **Philbrick** = Physical embodiment of continuous-time dynamics

They share:
- The same four primitives (sum, integrate, nonlinearity, events)
- The same compositional philosophy
- Compatible type systems and execution models
- The goal of eventual bidirectional compilation

---

## Document Map

### [01-OVERVIEW.md](01-OVERVIEW.md)
**How the Platforms Connect**

- Architectural mirror: software ↔ hardware
- Layer-by-layer mapping
- Shared design principles
- Integration vision

### [02-SHARED-PRIMITIVES.md](02-SHARED-PRIMITIVES.md)
**The Four Core Operations**

- Sum / Linear transforms
- Integration / Dynamics over time
- Nonlinearity / Shaping & activation
- Events / Sampling & discrete transitions
- How they map across substrates

### [03-WORKFLOWS.md](03-WORKFLOWS.md)
**Design → Build → Validate**

- Morphogen as design tool for Philbrick modules
- Philbrick as Morphogen accelerator
- Bidirectional testing
- Hybrid execution models

### [04-COMPILATION.md](04-COMPILATION.md)
**Morphogen → Philbrick Firmware**

- MLIR lowering to embedded targets
- Firmware code generation
- Module descriptor generation
- Future: Hardware synthesis from Morphogen operators

---

## Quick Comparison

| Aspect | Morphogen (Software) | Philbrick (Hardware) |
|--------|---------------------|---------------------|
| **Substrate** | CPU/GPU computation | Analog circuits, DSP, neural chips |
| **Primitives** | Operators (FFT, integrate, advect) | Modules (sum, integrate, clip, trigger) |
| **Type Safety** | Domain/rate/unit checking | Pin/voltage/impedance contracts |
| **Scheduling** | Multirate deterministic | Latency-aware routing |
| **Composition** | Computational graphs | Signal chains & feedback loops |
| **Determinism** | Strict/repro/live modes | Latency classes & measurement |

**Same Philosophy**: Computation = composition. Emergence from simple primitives.

---

## The Integration Vision

### Phase 1: Shared Vocabulary (Current)
- Define unified descriptor protocol
- Map operators to hardware primitives
- Establish type system compatibility

### Phase 2: Bidirectional Workflow (Months 3-6)
- Design in Morphogen → test in simulation
- Build in Philbrick → validate against Morphogen
- Generate test vectors for module calibration

### Phase 3: Compilation Pipeline (Months 6-12)
- Morphogen operators → Philbrick firmware
- MLIR lowering to Cortex-M embedded targets
- Automatic module generation

### Phase 4: Hybrid Execution (Year 2+)
- Philbrick modules as Morphogen accelerators
- Offload specific operators to hardware
- Mixed software/hardware execution graphs

---

## Why This Matters

**Nobody else has this pairing**:
- Eurorack: Hardware-only, no software abstraction
- DAWs: Software-only, no physical substrate
- ML accelerators: Narrow focus, no creative platform
- **Morphogen + Philbrick**: Both worlds unified

You can:
- **Design** a continuous-time system in Morphogen
- **Simulate** and optimize it in software
- **Build** it physically with Philbrick modules
- **Validate** that hardware matches simulation
- **Extend** with hybrid software/hardware execution

---

## Example: Guitar Body Modeling

### Design Phase (Morphogen)
```python
# Morphogen: Design and train body model
from morphogen.stdlib import audio, acoustics, optimization

# Define guitar body as acoustic resonator
body = acoustics.ModalResonator(modes=12, decay_times=...)

# Train to match real guitar impulse response
optimizer.fit(body, target_ir=real_guitar.wav)

# Export weights and parameters
body.export("j45_body_model.json")
```

### Build Phase (Philbrick)
- Load trained parameters into analog-neural inference chip
- Create physical module implementing the body model
- 3-4ms latency, real-time capable

### Validate Phase
- Feed test signals through Morphogen simulation
- Feed same signals through Philbrick hardware
- Compare outputs, measure accuracy

### Deploy Phase
- Cheap electric guitar → Philbrick body model → sounds like $3000 acoustic
- Real-time, low-latency, musically expressive

---

## Getting Started

**New to Morphogen?**
- Start with the [main README](../../README.md)
- Try the [Quick Start guide](../getting-started.md)

**New to Philbrick?**
- Visit the [Philbrick repository](https://github.com/scottsen/philbrick)
- Read the [Vision documentation](https://github.com/scottsen/philbrick/tree/main/docs/vision)

**Want to integrate them?**
- Read [01-OVERVIEW.md](01-OVERVIEW.md) for the big picture
- Study [02-SHARED-PRIMITIVES.md](02-SHARED-PRIMITIVES.md) for technical mapping
- Explore [03-WORKFLOWS.md](03-WORKFLOWS.md) for practical integration

---

## Community

**Discussions**:
- [Morphogen Discussions](https://github.com/scottsen/morphogen/discussions)
- [Philbrick Discussions](https://github.com/scottsen/philbrick/discussions)

**Cross-Platform Topics**:
- Operator → module mapping
- Firmware compilation strategies
- Hybrid execution models
- Educational curriculum development

---

*"The universe computes in analog. We model it in Morphogen. We embody it in Philbrick. This is the full circle."*

---

**Last Updated**: 2025-11-16
**Status**: Documentation in progress, technical integration roadmap defined
