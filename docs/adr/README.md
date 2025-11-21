# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records documenting key architectural decisions in Morphogen's development.

## What is an ADR?

An ADR captures:
- **Context**: What problem we're solving
- **Decision**: What we decided to do
- **Consequences**: What the decision means for the project

ADRs explain *why* decisions were made, not *how* to implement them (see [Specifications](../specifications/) for implementation details).

## Current ADRs

### [ADR-001: Unified Reference Model](001-unified-reference-model.md)
Establishes Morphogen's unified reference model for cross-domain simulation.

### [ADR-002: Cross-Domain Architectural Patterns](002-cross-domain-architectural-patterns.md)
Defines patterns for integrating multiple domains in a single system.

### [ADR-003: Circuit Modeling Domain](003-circuit-modeling-domain.md)
Decision to add circuit modeling as a first-class domain.

### [ADR-004: Instrument Modeling Domain](004-instrument-modeling-domain.md)
Decision to add audio instrument modeling capabilities.

### [ADR-005: Emergence Domain](005-emergence-domain.md)
Adds emergence and complex systems simulation domain.

### [ADR-006: Chemistry Domain](006-chemistry-domain.md)
Decision to support chemistry and chemical engineering workflows.

### [ADR-007: GPU-First Domains](007-gpu-first-domains.md)
Major paradigm shift: GPU-first execution for certain domains (BI domain).

### [ADR-008: Procedural Generation Domain](008-procedural-generation.md)
Adds procedural content generation as a core domain.

---

## ADR Numbering

**Note**: This directory was recently reorganized (2025-11-15) to fix numbering collisions:
- Previously had two files numbered "003" and four numbered "005"
- ADRs have been renumbered sequentially
- The "multiphysics success patterns" document was moved to [Reference](../reference/) as it's a patterns catalog, not a decision record

All file history has been preserved using `git mv`.

---

## Related Documentation

- **Implementation details?** See [Specifications](../specifications/)
- **High-level architecture?** See [Architecture](../architecture/)
- **How to implement?** See [Guides](../guides/)
- **Battle-tested patterns?** See [Multiphysics Success Patterns](../reference/multiphysics-success-patterns.md)

[← Back to Documentation Home](../README.md)
