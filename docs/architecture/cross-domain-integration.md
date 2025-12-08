# Cross-Domain Integration Guide

**Status**: Production-Ready
**Last Updated**: 2025-12-06 (Phase 3 Migration Complete)
**Consolidated From**: CROSS_DOMAIN_API.md, CROSS_DOMAIN_MESH*.md
**Mesh Catalog**: `../reference/domain-mesh-catalog.md`
**Visualizer**: `../../tools/visualize_mesh.py`

## Overview

Morphogen's cross-domain composition enables seamless data flow between computational domains (Field, Agent, Audio, Physics, Geometry, etc.).

**Key Features:**
- Type-safe transforms between domain pairs
- Automatic registration and discovery
- Bidirectional coupling (Field ↔ Agent, Physics ↔ Audio, etc.)
- Performance-optimized with NumPy/SciPy backends
- Validation at transform boundaries

## Quick Reference

See the following specialized guides:
- **API Reference**: This document (architecture)
- **Mesh Topology**: `../reference/domain-mesh-catalog.md` (catalog of all connections)
- **User Guide**: Implementation patterns and examples

## Cross-Domain API

For complete API details, see original `archive/CROSS_DOMAIN_API.md`

## Domain Mesh

The domain mesh visualizes all possible cross-domain transforms. See `../reference/domain-mesh-catalog.md` for the complete catalog.

**Current Mesh State** (2025-12-06):
- **17 domains** across 8 categories
- **18 transforms** (5 bidirectional pairs)
- **4 weakly connected components**
- **Top hub**: `field` (7 connections)
- **Visualization**: Run `python tools/visualize_mesh.py`

**Recent Updates** (Phase 3 Migration):
- Added operator decorators to 3 specialized domains:
  - `fluid_jet` (7 operators): Jet flow dynamics
  - `audio_analysis` (9 operators): Audio signal processing
  - `instrument_model` (5 operators): Musical instrument modeling
- Total: **12 domains, 126 operators** migrated with `@operator` decorators

**Consolidated from:**
- CROSS_DOMAIN_API.md (26KB)
- CROSS_DOMAIN_MESH_CATALOG.md (27KB)
- CROSS_DOMAIN_MESH.mermaid.md (3.6KB)
- CROSS_DOMAIN_MESH_ASCII.md (4.8KB)

