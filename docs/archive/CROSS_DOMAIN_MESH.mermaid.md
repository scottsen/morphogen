# Morphogen Cross-Domain Mesh (Mermaid Diagram)

This diagram shows all cross-domain transforms in the Morphogen mesh.

**Legend:**
- **`-->`** = Unidirectional transform
- **`<-->`** = Bidirectional transform (can go both ways)
- **Colors** = Domain categories (blue=field/spatial, orange=physics, green=audio, etc.)

```mermaid
graph LR
    %% Morphogen Cross-Domain Transformation Mesh

    %% Node styling by category
    classDef fieldSpatial fill:#E3F2FD,stroke:#2196F3,stroke-width:2px
    classDef physics fill:#FFF3E0,stroke:#FF9800,stroke-width:2px
    classDef audio fill:#E8F5E9,stroke:#4CAF50,stroke-width:2px
    classDef visual fill:#FCE4EC,stroke:#E91E63,stroke-width:2px
    classDef agent fill:#F3E5F5,stroke:#9C27B0,stroke-width:2px
    classDef biology fill:#FFF9C4,stroke:#FFEB3B,stroke-width:2px
    classDef geometry fill:#E0F2F1,stroke:#009688,stroke-width:2px
    classDef perception fill:#F1F8E9,stroke:#8BC34A,stroke-width:2px

    %% Transforms
    acoustics --> audio
    agent <--> field
    audio --> visual
    cartesian <--> polar
    cellular --> field
    cepstral <--> time
    field --> audio
    field <--> terrain
    fluid --> acoustics
    graph --> visual
    physics --> audio
    spatial <--> spatial
    time --> wavelet
    vision --> field

    %% Apply styling
    class field,terrain,spatial fieldSpatial
    class physics,fluid,acoustics physics
    class audio,time,cepstral,wavelet audio
    class visual,graph visual
    class agent agent
    class cellular biology
    class cartesian,polar geometry
    class vision perception
```

## Connected Components

### Component 1: Main Network (12 domains)
Contains: field, agent, terrain, vision, cellular, audio, visual, graph, physics, fluid, acoustics

**Key paths:**
- `fluid → acoustics → audio → visual` (3 hops)
- `terrain → field → audio → visual` (3 hops)
- `vision → field → agent` (2 hops)
- `cellular → field → audio` (2 hops)

### Component 2: Time-Frequency (3 domains) - ISOLATED ❌
Contains: time, cepstral, wavelet

**Missing:** Connection to main network (needs `audio → time`)

### Component 3: Coordinate Systems (2 domains) - ISOLATED ❌
Contains: cartesian, polar

**Missing:** Connection to geometry or field (needs `polar → field` or `geometry → cartesian`)

### Component 4: Spatial (1 domain) - ISOLATED ❌
Contains: spatial (self-loop only)

**Missing:** Connection to anything (needs `spatial → field` or `field → spatial`)

## Usage

**In GitHub/GitLab:** This renders automatically in markdown previews!

**In other markdown viewers:** Install a Mermaid plugin, or use:
- [Mermaid Live Editor](https://mermaid.live)
- [GitHub Markdown Preview](https://github.com)
- VS Code with Mermaid extension

## Quick Examples

```python
from morphogen.cross_domain.composer import TransformComposer

composer = TransformComposer()

# Terrain to visual (3 hops)
pipeline = composer.compose_path("terrain", "visual")
print(pipeline.visualize())
# terrain → field → audio → visual

# Vision to agent (2 hops)
pipeline = composer.compose_path("vision", "agent")
print(pipeline.visualize())
# vision → field → agent

# Time to wavelet (1 hop, but isolated from main network!)
pipeline = composer.compose_path("time", "wavelet")
print(pipeline.visualize())
# time → wavelet
```

## Related Documentation

- **Topology Guide:** `MESH_TOPOLOGY_GUIDE.md` - Deep dive into connectivity
- **User Guide:** `MESH_USER_GUIDE.md` - Tutorials and examples
- **Mesh Catalog:** `CROSS_DOMAIN_MESH_CATALOG.md` - Complete reference
- **API Docs:** `CROSS_DOMAIN_API.md` - Implementation details
