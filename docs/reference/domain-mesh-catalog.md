# Morphogen Cross-Domain Mesh Catalog

**Last Updated**: 2025-12-06 (Phase 3 Migration Complete)
**Visualization**: `tools/visualize_mesh.py`
**Status**: Production-Ready

## Overview

The Morphogen cross-domain mesh represents all registered transformation paths between computational domains. This catalog documents the current mesh topology, connectivity patterns, and domain relationships.

## Mesh Statistics

**Current State** (as of 2025-12-06):

```
Domains:              17
Transforms:           18
Bidirectional Pairs:  5
Connected Components: 4 (weakly connected)
```

**Graph Connectivity**:
- **Weakly connected**: No (4 separate components)
- **Largest component**: Contains `field`, `audio`, `agent`, `physics`, `visual`, `time`, `cepstral`, `wavelet`
- **Isolated components**: `spatial`, `cartesian/polar`, and individual source nodes

## Domain Categories

Domains are categorized by their primary computational purpose:

### Field/Spatial
- **field**: Spatial field simulation and computation
- **spatial**: Spatial data structures
- **terrain**: Procedural terrain generation

### Agents/Behavior
- **agent**: Agent-based modeling and simulation

### Physics/Simulation
- **physics**: Physics simulation (general)
- **fluid**: Fluid dynamics
- **acoustics**: Room acoustics and propagation

### Audio/Sound
- **audio**: Audio synthesis and processing
- **time**: Time-domain audio signals
- **cepstral**: Cepstral domain (quefrency)
- **wavelet**: Wavelet domain

### Visual/Graphics
- **visual**: Visual rendering and graphics
- **graph**: Graph and network algorithms

### Biology
- **cellular**: Cellular automata

### Perception
- **vision**: Computer vision and image processing

### Geometry
- **cartesian**: Cartesian coordinate system
- **polar**: Polar coordinate system

## Hub Analysis

### Top Hubs (by total degree)

1. **field** (7 connections)
   - Outbound: 3 (agent, audio, terrain)
   - Inbound: 4 (agent, terrain, vision, cellular)
   - **Role**: Primary integration hub

2. **audio** (4 connections)
   - Outbound: 1 (visual)
   - Inbound: 3 (physics, field, time)
   - **Role**: Audio processing hub

3. **time** (3 connections)
   - Outbound: 2 (cepstral, wavelet)
   - Inbound: 1 (cepstral)
   - **Role**: Time-domain audio gateway

4. **agent** (2 connections)
   - Outbound: 1 (field)
   - Inbound: 1 (field)
   - **Role**: Bidirectional field coupling

5. **visual** (2 connections)
   - Outbound: 0
   - Inbound: 2 (audio, graph)
   - **Role**: Visualization sink

### Sink Nodes (0 outbound)
- **visual**: Rendering endpoint
- **wavelet**: Wavelet analysis endpoint

### Source Nodes (0 inbound)
- **physics**: Physics simulation source
- **vision**: Computer vision source
- **graph**: Graph algorithms source
- **cellular**: Cellular automata source
- **fluid**: Fluid dynamics source

## Transform Catalog

### Bidirectional Transforms (5 pairs)

These transform pairs enable full bidirectional coupling:

1. **field ↔ agent**
   - Spatial fields to agent behaviors and vice versa
   - Use case: Agent-based field interactions

2. **field ↔ terrain**
   - Terrain generation from fields
   - Use case: Procedural terrain synthesis

3. **time ↔ cepstral**
   - Time-domain to cepstral domain analysis
   - Use case: Audio timbre extraction

4. **cartesian ↔ polar**
   - Coordinate system conversions
   - Use case: Geometric transformations

5. **spatial ↔ spatial**
   - Spatial data structure transformations
   - Use case: Spatial indexing and queries

### Unidirectional Transforms (8)

1. **physics → audio**: Physics-driven audio synthesis
2. **audio → visual**: Audio visualization (spectrograms, waveforms)
3. **field → audio**: Field-to-sound synthesis
4. **vision → field**: Image-to-field conversion
5. **graph → visual**: Graph visualization
6. **cellular → field**: Cellular automata to fields
7. **time → wavelet**: Wavelet decomposition
8. **fluid → (not registered)**: Fluid dynamics output

## Recent Updates

### Phase 3 Migration (2025-12-06)

Added operator decorators to 3 specialized domains:
- **fluid_jet** (7 operators): Jet flow dynamics
- **audio_analysis** (9 operators): Audio signal processing
- **instrument_model** (5 operators): Musical instrument modeling

**Note**: These domains are not yet integrated into the cross-domain mesh. Future work includes:
- Adding fluid_jet ↔ physics transforms
- Adding audio_analysis ↔ audio transforms
- Adding instrument_model ↔ audio_analysis transforms

## Connectivity Gaps

### Isolated Components

**Component 1** (Main cluster): field, agent, audio, physics, visual, time, cepstral, wavelet, terrain, vision, graph, cellular, fluid

**Component 2**: spatial (self-loop only)

**Component 3**: cartesian ↔ polar

**Component 4**: Individual source nodes without outbound edges

### Missing High-Value Transforms

Priority transforms to improve mesh connectivity:

1. **audio → time**: Audio to time-domain extraction
2. **visual → graph**: Visual scene to graph structure
3. **physics → field**: Physics simulation to field representation
4. **terrain → visual**: Direct terrain rendering
5. **cellular → agent**: Cellular automata to agent behaviors
6. **fluid → field**: Fluid simulation to field representation

## Visualization

### Generating Mesh Diagrams

Use the built-in visualizer:

```bash
cd /home/scottsen/src/projects/morphogen

# Basic visualization
python tools/visualize_mesh.py

# Custom layout and format
python tools/visualize_mesh.py --output mesh.svg --format svg --layout spring

# High-resolution PDF
python tools/visualize_mesh.py --output mesh.pdf --format pdf --dpi 600

# Statistics only
python tools/visualize_mesh.py --stats
```

### Layout Algorithms

- **kamada_kawai** (default): Force-directed, good for general topology
- **spring**: Spring layout, emphasizes clusters
- **circular**: Nodes arranged in a circle
- **shell**: Concentric shells by connectivity

### Color Coding

Nodes are colored by category:
- 🔵 Field/Spatial (blue)
- 🟣 Agents/Behavior (purple)
- 🟠 Physics/Simulation (orange)
- 🟢 Audio/Sound (green)
- 🩷 Visual/Graphics (pink)
- 🟡 Biology (yellow)
- 🔷 Geometry (teal)
- 🟩 Perception (lime)

Edges:
- Gray: Unidirectional transform
- Blue: Bidirectional transform pair

## Implementation Guide

### Registering New Transforms

Add transforms to `morphogen/cross_domain/registry.py`:

```python
from morphogen.cross_domain import CrossDomainRegistry

@CrossDomainRegistry.register("source_domain", "target_domain")
def source_to_target(data):
    """Transform source domain data to target domain."""
    # Implementation
    return transformed_data
```

### Best Practices

1. **Validate inputs**: Check domain-specific constraints
2. **Type safety**: Use proper type hints and validation
3. **Bidirectional**: Implement reverse transforms when semantically valid
4. **Performance**: Optimize with NumPy/vectorization
5. **Documentation**: Document expected data shapes and semantics

## References

- **Cross-Domain Integration**: `../architecture/cross-domain-integration.md`
- **Implementation Patterns**: `../adr/002-cross-domain-architectural-patterns.md`
- **Examples**: `../examples/emergence-cross-domain.md`
- **Visualizer**: `../../tools/visualize_mesh.py`

## Warnings

The visualizer currently reports duplicate transform registrations:
- field → agent (overwriting)
- agent → field (overwriting)
- physics → audio (overwriting)
- audio → visual (overwriting)
- field → audio (overwriting)
- terrain → field (overwriting)
- field → terrain (overwriting)
- vision → field (overwriting)
- graph → visual (overwriting)
- cellular → field (overwriting)
- time → cepstral (overwriting)
- cepstral → time (overwriting)
- time → wavelet (overwriting)
- spatial → spatial (overwriting)
- cartesian → polar (overwriting)
- polar → cartesian (overwriting)

**Action Item**: Investigate duplicate registrations and consolidate to single canonical transforms.

---

**Maintenance**: Update this catalog after:
- Adding new domains
- Registering new transforms
- Completing domain operator migrations
- Major connectivity changes
