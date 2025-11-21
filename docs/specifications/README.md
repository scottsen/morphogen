# Technical Specifications

Detailed technical specifications for Morphogen's language, infrastructure, and domains.

## Language Specifications

- [KAX Language](kax-language.md) - The Morphogen language specification
- [Type System](type-system.md) - Type system design and semantics

## Core Infrastructure

- [Graph IR](graph-ir.md) - Intermediate representation for operator graphs
- [MLIR Dialects](mlir-dialects.md) - MLIR dialect specifications
- [Operator Registry](operator-registry.md) - Operator registration and discovery system
- [Scheduler](scheduler.md) - Execution scheduling and ordering
- [Transform](transform.md) - Graph transformation system
- [Profiles](profiles.md) - Execution profiles and optimization levels
- [Snapshot ABI](snapshot-abi.md) - Snapshot serialization format
- [Coordinate Frames](coordinate-frames.md) - Coordinate system handling
- [Geometry](geometry.md) - Geometric primitives and operations

## Domain Specifications

### Physics & Simulation
- [Chemistry](chemistry.md) - Chemistry and chemical engineering domain
- [Circuit](circuit.md) - Electrical circuit modeling
- [Physics Domains](physics-domains.md) - Physics simulation domains
- [Emergence](emergence.md) - Emergence and complex systems domain

### Procedural & Creative
- [Procedural Generation](procedural-generation.md) - Procedural content generation
- [BI Domain](bi-domain.md) - GPU-first buffer imaging domain

### Media & Audio
- [Video/Audio Encoding](video-audio-encoding.md) - Media encoding and processing
- [Timbre Extraction](timbre-extraction.md) - Audio timbre analysis

---

## Navigation Tips

- **New to Morphogen?** Start with [KAX Language](kax-language.md) and [Graph IR](graph-ir.md)
- **Implementing a domain?** Read the relevant domain spec, then check [Domain Implementation Guide](../guides/domain-implementation.md)
- **Understanding why?** See [ADRs](../adr/) for the reasoning behind these designs
- **Need examples?** Check [Examples](../examples/) for working code

[← Back to Documentation Home](../README.md)
