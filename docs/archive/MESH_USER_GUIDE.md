# Morphogen Cross-Domain Mesh: User Guide

**For:** Developers, researchers, and creative coders
**Level:** Beginner to Advanced
**Version:** 1.0 (2025-12-06)

> **⚠️ ARCHIVED DOCUMENTATION**
>
> This guide contains outdated initialization patterns. As of 2025-12-07:
> - Transforms are **auto-registered on module import** (no explicit call needed)
> - `registry.register_builtin_transforms()` is now **idempotent** (safe to call multiple times)
> - Explicit calls are harmless but unnecessary for normal usage
>
> See `morphogen/cross_domain/registry.py` for current API.

---

## What is the Cross-Domain Mesh?

The Morphogen Cross-Domain Mesh is a **transformation network** that lets you convert data between different computational domains automatically. Think of it as a "universal translator" for computational data.

**Example:**
```python
# Want to turn terrain data into sound? The mesh finds the path:
terrain_data → field → audio
                ↓
        "Terrain → Field → Audio" pipeline created automatically!
```

**Key Benefits:**
- ✅ **Automatic path finding** - No need to manually chain transforms
- ✅ **Type safety** - Validates data compatibility
- ✅ **Composable** - Build complex multi-domain workflows
- ✅ **Extensible** - Add new transforms easily

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Basic Workflows](#basic-workflows)
3. [Advanced Techniques](#advanced-techniques)
4. [Common Patterns](#common-patterns)
5. [Troubleshooting](#troubleshooting)
6. [Real-World Examples](#real-world-examples)
7. [Performance Tips](#performance-tips)
8. [FAQ](#faq)

---

## Quick Start

### Installation

Morphogen mesh is included in the Morphogen package:

```bash
pip install morphogen
```

### Your First Transform

```python
from morphogen.cross_domain import registry
from morphogen.cross_domain.composer import TransformComposer

# Initialize the registry
registry.register_builtin_transforms()

# Create a composer (handles path finding)
composer = TransformComposer(enable_caching=True)

# Find a path from terrain to audio
pipeline = composer.compose_path("terrain", "audio")

# Use it!
import numpy as np
terrain_data = np.random.rand(64, 64)  # Example terrain

audio_output = pipeline(terrain_data)
print(f"✓ Generated audio from terrain: {audio_output.shape}")
```

**Output:**
```
✓ Generated audio from terrain: (44100,)
```

### Check Available Transforms

```python
from morphogen.cross_domain.registry import CrossDomainRegistry

# List all transforms
transforms = CrossDomainRegistry.list_all()
print(f"Available transforms: {len(transforms)}")

for src, tgt in sorted(transforms)[:5]:
    print(f"  {src} → {tgt}")

# Output:
# Available transforms: 18
#   acoustics → audio
#   agent → field
#   audio → visual
#   cartesian → polar
#   cellular → field
```

### Visualize a Pipeline

```python
pipeline = composer.compose_path("vision", "visual")

# See the transformation steps
print(pipeline.visualize())

# Output:
# vision → field (VisionToFieldInterface)
# field → audio (FieldToAudioInterface)
# audio → visual (AudioToVisualInterface)
```

---

## Basic Workflows

### Workflow 1: Direct Transform (1 hop)

**Scenario:** You have a direct transform between domains.

```python
# Direct: field → audio
pipeline = composer.compose_path("field", "audio")

# Create field data
field_data = np.random.rand(64, 64)

# Transform to audio
audio = pipeline(field_data)
print(f"Audio shape: {audio.shape}")
```

**When to use:** When domains are directly connected (see mesh catalog).

---

### Workflow 2: Multi-Hop Transform (2-3 hops)

**Scenario:** No direct path exists, but mesh can route through intermediate domains.

```python
# Terrain → Audio requires routing through field
pipeline = composer.compose_path("terrain", "audio")

print(f"Path length: {pipeline.length} hops")
print(pipeline.visualize())

# Output:
# Path length: 2 hops
# terrain → field (TerrainToFieldInterface)
# field → audio (FieldToAudioInterface)

terrain_heightmap = np.random.rand(128, 128)
audio = pipeline(terrain_heightmap)
```

**When to use:** When automatic path finding is acceptable.

---

### Workflow 3: Constrained Routing

**Scenario:** You want to force the path through specific intermediate domains.

```python
# Force vision → field → agent (instead of auto-finding shortest path)
pipeline = composer.compose_path("vision", "agent", via=["field"])

print(pipeline.visualize())

# Output:
# vision → field (VisionToFieldInterface)
# field → agent (FieldToAgentInterface)
```

**When to use:**
- You need a specific transformation sequence
- For debugging or testing
- When semantics of intermediate steps matter

---

### Workflow 4: Bidirectional Round-Trip

**Scenario:** Transform data, process it, then transform back.

```python
# Agent positions → field → process → agent positions
agent_to_field = composer.compose_path("agent", "field")
field_to_agent = composer.compose_path("field", "agent")

# Forward transform
agent_positions = np.random.rand(100, 2)  # 100 agents, 2D positions
field = agent_to_field(agent_positions)

# Process field (e.g., diffusion, forces)
processed_field = apply_diffusion(field)

# Backward transform
new_positions = field_to_agent(processed_field)
```

**When to use:** Round-trip workflows with bidirectional transforms.

---

### Workflow 5: Batch Processing

**Scenario:** Transform multiple inputs efficiently.

```python
from morphogen.cross_domain.composer import BatchTransformComposer

batch_composer = BatchTransformComposer()
pipeline = composer.compose_path("cellular", "audio")

# Multiple cellular automata states
cellular_states = [state1, state2, state3, state4]

# Batch transform
audio_outputs = batch_composer.batch_transform(pipeline, cellular_states)

print(f"Processed {len(audio_outputs)} inputs")
```

**When to use:** Processing datasets, batch inference, parameter sweeps.

---

## Advanced Techniques

### 1. Path Finding with Max Hops

Control how far the algorithm searches:

```python
# Try short path first
path = composer.find_path("fluid", "visual", max_hops=2)
if not path:
    print("No path within 2 hops")

    # Expand search
    path = composer.find_path("fluid", "visual", max_hops=4)
    if path:
        print(f"Found path with {len(path)} hops")
```

**Use case:** Performance optimization, bounded search.

---

### 2. Inspecting Transforms

```python
# Get transform class directly
from morphogen.cross_domain.registry import CrossDomainRegistry

transform_class = CrossDomainRegistry.get("field", "audio")
print(f"Transform class: {transform_class.__name__}")

# Get metadata
metadata = CrossDomainRegistry.get_metadata("field", "audio")
print(f"Metadata: {metadata}")
```

**Use case:** Understanding transform characteristics, debugging.

---

### 3. Manual Transform Composition

```python
from morphogen.cross_domain.composer import compose

# Create individual transforms
field_to_audio_transform = CrossDomainRegistry.get("field", "audio")()
audio_to_visual_transform = CrossDomainRegistry.get("audio", "visual")()

# Manual composition
manual_pipeline = compose(field_to_audio_transform, audio_to_visual_transform)

# Use composed function
field_data = np.random.rand(64, 64)
visual_output = manual_pipeline(field_data)
```

**Use case:** Fine-grained control, custom pipelines.

---

### 4. Error Handling

```python
try:
    # Try to create a pipeline for unreachable domains
    pipeline = composer.compose_path("polar", "audio")
except ValueError as e:
    print(f"❌ No path: {e}")

    # Fallback strategy
    print("Fallback: Using manual conversion...")
    # polar → cartesian (Component 3)
    # Then manually import to field
```

**Use case:** Robust applications, handling disconnected components.

---

### 5. Caching Performance

```python
composer = TransformComposer(enable_caching=True)

# First call: cache miss
pipeline1 = composer.compose_path("terrain", "audio")

# Second call: cache hit (much faster!)
pipeline2 = composer.compose_path("terrain", "audio")

# Check stats
stats = composer.get_stats()
print(f"Cache hits: {stats['cache_hits']}")
print(f"Cache misses: {stats['cache_misses']}")
print(f"Cache size: {stats['cache_size']}")
```

**Use case:** Interactive applications, repeated queries.

---

## Common Patterns

### Pattern 1: Perception → Simulation → Perception Loop

```python
# Vision → Field → Agent → Field → Audio → Visual

# Stage 1: Vision to field
vision_to_field = composer.compose_path("vision", "field")
field = vision_to_field(image_data)

# Stage 2: Field influences agents
field_to_agent = composer.compose_path("field", "agent")
agent_positions = field_to_agent(field)

# Stage 3: Agents create new field
agent_to_field = composer.compose_path("agent", "field")
new_field = agent_to_field(agent_positions)

# Stage 4: Sonify field
field_to_audio = composer.compose_path("field", "audio")
audio = field_to_audio(new_field)

# Stage 5: Visualize audio
audio_to_visual = composer.compose_path("audio", "visual")
visualization = audio_to_visual(audio)
```

**Use case:** Creative coding, generative art, feedback systems.

---

### Pattern 2: Multi-Source Audio Synthesis

```python
# Combine multiple domain sources into audio

# Source 1: Physics simulation
physics_to_audio = composer.compose_path("physics", "audio")
audio1 = physics_to_audio(physics_state)

# Source 2: Field dynamics
field_to_audio = composer.compose_path("field", "audio")
audio2 = field_to_audio(field_data)

# Source 3: Terrain sonification
terrain_to_audio = composer.compose_path("terrain", "audio")
audio3 = terrain_to_audio(terrain_map)

# Mix audio sources
mixed_audio = (audio1 + audio2 + audio3) / 3
```

**Use case:** Multi-modal synthesis, audio mixing.

---

### Pattern 3: Terrain → Audio → Visual Pipeline

```python
# Complete generative pipeline

# Generate terrain
terrain = generate_terrain(size=256)

# Sonify terrain
terrain_to_audio_pipeline = composer.compose_path("terrain", "visual")
print(terrain_to_audio_pipeline.visualize())
# terrain → field → audio → visual

# Execute full pipeline
visualization = terrain_to_audio_pipeline(terrain)

# Display
import matplotlib.pyplot as plt
plt.imshow(visualization)
plt.title("Terrain Sonification Visualization")
plt.show()
```

**Use case:** Procedural generation, data visualization.

---

### Pattern 4: Cellular Automata → Agent Swarm

```python
# Use CA patterns to drive agent behavior

# Run cellular automaton
ca_state = run_cellular_automaton(steps=100)

# Convert CA to field
ca_to_field = composer.compose_path("cellular", "field")
field = ca_to_field(ca_state)

# Field drives agents
field_to_agent = composer.compose_path("field", "agent")
agent_positions = field_to_agent(field)

# Simulate agents
agents = create_agents(agent_positions)
```

**Use case:** AI behavior, swarm intelligence, generative systems.

---

## Troubleshooting

### Problem 1: "No transform path found"

**Error:**
```python
ValueError: No transform path found from polar to audio
```

**Causes:**
1. Domains are in different connected components
2. Max hops too low
3. Transform doesn't exist

**Solutions:**

**A) Check connectivity:**
```python
# Use topology guide to identify connected components
# See: MESH_TOPOLOGY_GUIDE.md → "Connected Components"
```

**B) Increase max hops:**
```python
path = composer.find_path("source", "target", max_hops=5)
```

**C) Route through field:**
```python
# Two-stage pipeline
stage1 = composer.compose_path("source", "field")
stage2 = composer.compose_path("field", "target")

intermediate = stage1(data)
result = stage2(intermediate)
```

**D) Implement missing transform:**
```python
# See CROSS_DOMAIN_API.md for how to add new transforms
```

---

### Problem 2: Slow path finding

**Symptom:** `find_path()` takes long time

**Solutions:**

**A) Enable caching:**
```python
composer = TransformComposer(enable_caching=True)
```

**B) Reduce max hops:**
```python
# Default is 3, which is usually sufficient
path = composer.find_path(src, tgt, max_hops=2)
```

---

### Problem 3: Invalid data format

**Error:**
```python
CrossDomainValidationError: Field data must be 2D or 3D array
```

**Cause:** Input data doesn't match expected format for domain

**Solution:** Check validation requirements
```python
from morphogen.cross_domain import validators

# Validate before transforming
validators.validate_field_data(field_data)  # Raises error if invalid
```

See [`CROSS_DOMAIN_MESH_CATALOG.md`](CROSS_DOMAIN_MESH_CATALOG.md) → "Validation & Type Safety" for format requirements.

---

### Problem 4: Unexpected transformation result

**Symptom:** Output doesn't look right

**Debug steps:**

**A) Visualize pipeline:**
```python
print(pipeline.visualize())
# Check if path is what you expected
```

**B) Inspect intermediate steps:**
```python
# Break pipeline into stages
stage1 = composer.compose_path("source", "intermediate")
stage2 = composer.compose_path("intermediate", "target")

intermediate_result = stage1(data)
print(f"Intermediate result shape: {intermediate_result.shape}")

final_result = stage2(intermediate_result)
```

**C) Check transform metadata:**
```python
metadata = CrossDomainRegistry.get_metadata("source", "target")
print(metadata)
```

---

## Real-World Examples

### Example 1: Generative Music from Terrain

```python
import numpy as np
import soundfile as sf
from morphogen.cross_domain.composer import TransformComposer

# Setup
composer = TransformComposer(enable_caching=True)

# Generate fractal terrain
def generate_fractal_terrain(size=256):
    # ... fractal generation code ...
    return terrain_heightmap

# Create pipeline
terrain_to_audio = composer.compose_path("terrain", "audio")

# Generate terrain
terrain = generate_fractal_terrain(size=256)

# Transform to audio
audio = terrain_to_audio(terrain)

# Save audio file
sf.write("terrain_music.wav", audio, 44100)
print("✓ Saved terrain_music.wav")
```

---

### Example 2: Vision-Driven Agent Simulation

```python
import cv2
from morphogen.cross_domain.composer import TransformComposer

composer = TransformComposer()

# Load image
image = cv2.imread("input.jpg")

# Vision → Field → Agent pipeline
pipeline = composer.compose_path("vision", "agent")
print(f"Pipeline: {pipeline.visualize()}")

# Extract agent positions from image
agent_positions = pipeline(image)

# Create agent simulation
agents = [Agent(pos) for pos in agent_positions]

# Run simulation
for step in range(100):
    for agent in agents:
        agent.update()

    render_agents(agents)
```

---

### Example 3: Audio Visualization

```python
from morphogen.cross_domain.composer import TransformComposer
import matplotlib.pyplot as plt

composer = TransformComposer()

# Load audio
audio_data = load_audio("song.wav")

# Audio → Visual
pipeline = composer.compose_path("audio", "visual")

# Generate visualization
visual = pipeline(audio_data)

# Display
plt.figure(figsize=(12, 8))
plt.imshow(visual, cmap='viridis')
plt.title("Audio Visualization")
plt.axis('off')
plt.tight_layout()
plt.savefig("audio_viz.png", dpi=300)
```

---

### Example 4: Physics-Based Sonification

```python
from morphogen.cross_domain.composer import TransformComposer

composer = TransformComposer()

# Run physics simulation
physics_state = run_physics_simulation(num_particles=100, steps=1000)

# Physics → Audio → Visual
pipeline = composer.compose_path("physics", "visual")
print(pipeline.visualize())

# Output:
# physics → audio (PhysicsToAudioInterface)
# audio → visual (AudioToVisualInterface)

# Execute
visualization = pipeline(physics_state)

# Save results
save_visualization(visualization, "physics_viz.png")
```

---

## Performance Tips

### Tip 1: Reuse Pipelines

❌ **Don't do this:**
```python
for data in dataset:
    pipeline = composer.compose_path("terrain", "audio")  # Recreated every loop!
    result = pipeline(data)
```

✅ **Do this:**
```python
pipeline = composer.compose_path("terrain", "audio")  # Created once

for data in dataset:
    result = pipeline(data)
```

---

### Tip 2: Use Batch Processing

❌ **Don't do this:**
```python
results = []
for data in dataset:
    result = pipeline(data)
    results.append(result)
```

✅ **Do this:**
```python
from morphogen.cross_domain.composer import BatchTransformComposer

batch_composer = BatchTransformComposer()
results = batch_composer.batch_transform(pipeline, dataset)
```

---

### Tip 3: Enable Caching

```python
# Always enable caching for interactive applications
composer = TransformComposer(enable_caching=True)
```

---

### Tip 4: Limit Search Depth

```python
# If you know path is short, limit search
path = composer.find_path(src, tgt, max_hops=2)
```

---

## FAQ

**Q: How do I know which domains are available?**

```python
from morphogen.cross_domain.registry import CrossDomainRegistry

transforms = CrossDomainRegistry.list_all()
domains = set()
for src, tgt in transforms:
    domains.add(src)
    domains.add(tgt)

print(f"Available domains: {sorted(domains)}")
```

**Q: Can I add my own transforms?**

Yes! See [`CROSS_DOMAIN_API.md`](CROSS_DOMAIN_API.md) → "Adding New Transforms"

**Q: What if two domains aren't connected?**

Options:
1. Route through `field` (universal hub)
2. Implement a new transform
3. Use two-stage pipeline with manual data preparation

**Q: How do I visualize the entire mesh?**

```bash
cd /path/to/morphogen
python tools/visualize_mesh.py --output mesh.png
```

Or see [`CROSS_DOMAIN_MESH.png`](CROSS_DOMAIN_MESH.png)

**Q: What's the longest possible path?**

Currently 3 hops: `fluid → acoustics → audio → visual`

**Q: Are all transforms bidirectional?**

No. Only 5 pairs are bidirectional:
- agent ↔ field
- field ↔ terrain
- time ↔ cepstral
- cartesian ↔ polar
- spatial ↔ spatial

**Q: How fast is path finding?**

BFS-based, very fast for small meshes (< 1ms typically). With caching, subsequent queries are near-instant.

---

## Next Steps

- **Explore the mesh:** See [`MESH_TOPOLOGY_GUIDE.md`](MESH_TOPOLOGY_GUIDE.md)
- **Learn the API:** See [`CROSS_DOMAIN_API.md`](CROSS_DOMAIN_API.md)
- **Browse examples:** See `/examples/cross_domain/`
- **View the catalog:** See [`CROSS_DOMAIN_MESH_CATALOG.md`](CROSS_DOMAIN_MESH_CATALOG.md)

---

**Happy transforming!** 🎨🔊🌍
