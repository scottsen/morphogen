# Cross-Domain Operator Composition API

**Version:** 1.0
**Status:** Production-Ready
**Last Updated:** 2025-11-16

---

## Overview

Morphogen's cross-domain composition infrastructure enables seamless data flow between different computational domains (Field, Agent, Audio, Physics, Geometry, etc.). This document describes the API for creating, registering, and using cross-domain transforms.

**Key Features:**
- **Type-safe transforms** between domain pairs
- **Automatic registration** and discovery
- **Bidirectional coupling** (Field ↔ Agent, Physics ↔ Audio, etc.)
- **Performance-optimized** with NumPy/SciPy backends
- **Validation** at transform boundaries

---

## Architecture

### Domain Interface Pattern

Every cross-domain transform implements the `DomainInterface` base class:

```python
class DomainInterface(ABC):
    source_domain: str  # e.g., "field"
    target_domain: str  # e.g., "agent"

    @abstractmethod
    def transform(self, source_data: Any) -> Any:
        """Convert source domain data to target domain format."""
        ...

    @abstractmethod
    def validate(self) -> bool:
        """Ensure data types are compatible across domains."""
        ...
```

### Transform Registry

All transforms are registered in the global `CrossDomainRegistry`:

```python
# Check if transform exists
if CrossDomainRegistry.has_transform("field", "agent"):
    # Get transform class
    TransformClass = CrossDomainRegistry.get("field", "agent")

    # Create instance and apply
    transform = TransformClass(field_data, agent_positions)
    result = transform(source_data)
```

---

## Built-In Transforms

### 1. Field → Agent

**Purpose:** Sample field values at agent positions

**Use Cases:**
- Flow field → particle forces
- Temperature field → agent behavior
- Density field → agent sensing

**API:**
```python
from morphogen.cross_domain.interface import FieldToAgentInterface

# Create transform
transform = FieldToAgentInterface(
    field=velocity_field,  # (H, W) or (H, W, C) numpy array
    positions=agent_positions,  # (N, 2) numpy array
    property_name="velocity"  # Optional name
)

# Sample field at agent positions (bilinear interpolation)
sampled_values = transform(field)
```

**Parameters:**
- `field`: Numpy array, shape `(H, W)` (scalar) or `(H, W, C)` (vector)
- `positions`: Numpy array, shape `(N, 2)`, coordinates in grid space
- `property_name`: String, optional property identifier

**Returns:**
- Scalar field: `(N,)` array of sampled values
- Vector field: `(N, C)` array of sampled vector components

**Example:**
```python
import numpy as np
from morphogen.cross_domain.interface import FieldToAgentInterface

# Create velocity field (vortex)
y, x = np.mgrid[0:100, 0:100]
vx = -(y - 50) / 100
vy = (x - 50) / 100
velocity_field = np.stack([vx, vy], axis=2)

# Agent positions
positions = np.array([[25, 25], [75, 75]], dtype=np.float32)

# Sample velocities at agent positions
interface = FieldToAgentInterface(velocity_field, positions)
velocities = interface.transform(velocity_field)

print(velocities.shape)  # (2, 2) - two agents, two velocity components
```

---

### 2. Agent → Field

**Purpose:** Deposit agent properties onto field grid

**Use Cases:**
- Particle positions → density field
- Agent velocities → velocity field
- Agent heat → temperature sources

**API:**
```python
from morphogen.cross_domain.interface import AgentToFieldInterface

# Create transform
transform = AgentToFieldInterface(
    positions=agent_positions,  # (N, 2) array
    values=agent_properties,    # (N,) array of values to deposit
    field_shape=(128, 128),     # Output field shape
    method="accumulate"         # "accumulate", "average", or "max"
)

# Deposit to field
field = transform((positions, values))
```

**Parameters:**
- `positions`: Numpy array, shape `(N, 2)`, agent positions
- `values`: Numpy array, shape `(N,)`, values to deposit per agent
- `field_shape`: Tuple `(H, W)`, output field dimensions
- `method`: String, deposition method:
  - `"accumulate"`: Sum all values at each grid cell
  - `"average"`: Average values at each grid cell
  - `"max"`: Take maximum value at each grid cell

**Returns:**
- Numpy array, shape `(H, W)`, deposited field

**Example:**
```python
# Create particles
positions = np.random.rand(100, 2) * 128
values = np.ones(100)  # Unit density per particle

# Deposit to field
interface = AgentToFieldInterface(
    positions, values,
    field_shape=(128, 128),
    method="accumulate"
)
density_field = interface.transform((positions, values))

print(density_field.shape)  # (128, 128)
print(density_field.max())  # Maximum density (how many particles in a cell)
```

---

### 3. Physics → Audio

**Purpose:** Sonification of physical events

**Use Cases:**
- Collision forces → percussion synthesis
- Body velocities → pitch/volume
- Contact points → spatial audio

**API:**
```python
from morphogen.cross_domain.interface import PhysicsToAudioInterface

# Create transform with mapping
transform = PhysicsToAudioInterface(
    events=collision_events,  # List of event objects
    mapping={
        "impulse": "amplitude",   # Force magnitude → volume
        "body_id": "pitch",       # Object ID → frequency
        "position": "pan",        # Position → stereo pan
    },
    sample_rate=48000
)

# Convert to audio parameters
audio_params = transform(events)
```

**Parameters:**
- `events`: List of event objects with properties (impulse, body_id, position, time, etc.)
- `mapping`: Dict mapping physics properties to audio parameters
- `sample_rate`: Integer, audio sample rate (Hz)

**Physics Properties:**
- `impulse`: Collision impulse magnitude
- `body_id`: Unique object identifier
- `position`: 2D/3D position tuple
- `velocity`: Velocity vector
- `time`: Event timestamp

**Audio Parameters:**
- `amplitude`: Volume (0.0 to 1.0)
- `pitch`: Frequency (Hz)
- `pan`: Stereo position (-1.0 to 1.0)
- `duration`: Note length (seconds)

**Returns:**
Dict with keys:
- `triggers`: List of sample indices when events occur
- `amplitudes`: List of volume levels
- `frequencies`: List of frequencies (Hz)
- `positions`: List of positions (for spatial audio)

**Example:**
```python
# Mock collision events
class CollisionEvent:
    def __init__(self, impulse, body_id, position, time):
        self.impulse = impulse
        self.body_id = body_id
        self.position = position
        self.time = time

events = [
    CollisionEvent(impulse=50.0, body_id=0, position=(0, 0), time=0.0),
    CollisionEvent(impulse=100.0, body_id=1, position=(10, 5), time=0.1),
]

# Sonify
interface = PhysicsToAudioInterface(
    events,
    mapping={"impulse": "amplitude", "body_id": "pitch"},
    sample_rate=48000
)
audio_params = interface.transform(events)

print(audio_params["amplitudes"])  # [0.5, 1.0]
print(audio_params["frequencies"])  # [261.63, 293.66, ...]
```

---

## Creating Custom Transforms

### Method 1: Subclass `DomainInterface`

```python
from morphogen.cross_domain.interface import DomainInterface
from morphogen.cross_domain.registry import register_transform
import numpy as np

@register_transform("geometry", "field", metadata={"version": "1.0"})
class GeometryToFieldInterface(DomainInterface):
    """Convert geometry mesh to signed distance field."""

    source_domain = "geometry"
    target_domain = "field"

    def __init__(self, mesh, grid_size=128):
        super().__init__()
        self.mesh = mesh
        self.grid_size = grid_size

    def transform(self, source_data):
        """Convert mesh to SDF."""
        mesh = source_data if source_data else self.mesh

        # Compute signed distance field
        sdf = self._compute_sdf(mesh, self.grid_size)
        return sdf

    def validate(self):
        """Check mesh is valid."""
        if not hasattr(self.mesh, 'vertices'):
            raise ValueError("Mesh must have vertices attribute")
        return True

    def _compute_sdf(self, mesh, grid_size):
        """Compute signed distance field from mesh."""
        # Implementation...
        sdf = np.zeros((grid_size, grid_size), dtype=np.float32)
        # ... compute distances ...
        return sdf
```

### Method 2: Use `@DomainTransform` Decorator

```python
from morphogen.cross_domain.interface import DomainTransform
import numpy as np

@DomainTransform(
    source="field",
    target="image",
    name="field_to_image",
    description="Convert field to RGB image"
)
def field_to_image(field, cmap="viridis"):
    """Convert scalar field to RGB image."""
    # Normalize to [0, 1]
    normalized = (field - field.min()) / (field.max() - field.min() + 1e-10)

    # Apply colormap
    from matplotlib.cm import get_cmap
    cmap_fn = get_cmap(cmap)
    rgb = cmap_fn(normalized)[:, :, :3]  # Drop alpha

    return (rgb * 255).astype(np.uint8)
```

---

## Language Support

### `compose()` Statement

Parallel composition of cross-domain modules:

```morphogen
compose(module1, module2, module3)
```

**Example:**
```morphogen
// Define modules
module FluidField(dt: f32) {
    @state vel : Field2D<Vec2<f32>> = zeros((256, 256))
    // ...
}

module ParticleSystem(dt: f32) {
    @state agents : Agents<Particle> = alloc(count=1000)
    // ...
}

// Compose in parallel
compose(
    FluidField(dt=0.01),
    ParticleSystem(dt=0.01)
)
```

### `link()` Statement

Declare dependency metadata (no runtime cost):

```morphogen
link module_name { metadata... }
```

**Example:**
```morphogen
link AudioDomain {
    version: 1.0,
    required: true,
    provides: ["oscillators", "filters", "effects"]
}
```

---

## Validation

### Type Checking

```python
from morphogen.cross_domain.validators import validate_cross_domain_flow

# Validate a cross-domain flow
is_valid = validate_cross_domain_flow(
    source_domain="field",
    target_domain="agent",
    source_data=field_data
)
```

### Field Validation

```python
from morphogen.cross_domain.validators import validate_field_data

validate_field_data(field, allow_vector=True)  # Raises on error
```

### Agent Position Validation

```python
from morphogen.cross_domain.validators import validate_agent_positions

validate_agent_positions(positions, ndim=2)  # Raises on error
```

### Dimensional Compatibility

```python
from morphogen.cross_domain.validators import check_dimensional_compatibility

check_dimensional_compatibility(field_shape=(128, 128), positions=agent_pos)
```

---

## Registry Operations

### List All Transforms

```python
from morphogen.cross_domain.registry import CrossDomainRegistry

# List all registered transforms
all_transforms = CrossDomainRegistry.list_all()
print(all_transforms)
# [('field', 'agent'), ('agent', 'field'), ('physics', 'audio'), ...]
```

### List Transforms for a Domain

```python
# List all transforms where "field" is source
field_outputs = CrossDomainRegistry.list_transforms("field", direction="source")

# List all transforms where "agent" is target
agent_inputs = CrossDomainRegistry.list_transforms("agent", direction="target")

# List all transforms involving "audio" (either direction)
audio_transforms = CrossDomainRegistry.list_transforms("audio", direction="both")
```

### Visualize Transform Graph

```python
print(CrossDomainRegistry.visualize())
```

Output:
```
Cross-Domain Transform Graph:

  agent → field
  field → agent
  physics → audio
```

---

## Performance Tips

1. **Reuse Interface Objects**
   ```python
   # Good: Reuse interface for multiple transforms
   interface = FieldToAgentInterface(field, positions)
   for timestep in range(1000):
       values = interface.transform(field)  # Fast
   ```

2. **Batch Operations**
   ```python
   # Good: Transform all agents at once
   all_values = interface.transform(all_positions)

   # Bad: Transform agents one by one (slow)
   for pos in positions:
       value = interface.transform(pos)  # Slow!
   ```

3. **Use Appropriate Methods**
   ```python
   # For sparse deposition: use "max" or "average" to avoid overflow
   interface = AgentToFieldInterface(..., method="max")
   ```

---

## Complete Example: Field-Agent Coupling

```python
import numpy as np
from morphogen.cross_domain.interface import FieldToAgentInterface, AgentToFieldInterface

# Setup
grid_size = 128
num_agents = 500

# Create flow field (vortex)
y, x = np.mgrid[0:grid_size, 0:grid_size]
dx, dy = x - 64, y - 64
r = np.sqrt(dx**2 + dy**2) + 1e-10
vx = -dy / r * np.exp(-r / 30)
vy = dx / r * np.exp(-r / 30)
velocity_field = np.stack([vy, vx], axis=2).astype(np.float32)

# Initialize agents
positions = np.random.rand(num_agents, 2) * grid_size

# Create transforms
field_to_agent = FieldToAgentInterface(velocity_field, positions)
agent_to_field = AgentToFieldInterface(
    positions, np.ones(num_agents),
    field_shape=(grid_size, grid_size),
    method="accumulate"
)

# Simulation loop
for step in range(100):
    # Field → Agent: Sample velocity
    velocities = field_to_agent.transform(velocity_field)

    # Update positions
    positions += velocities * 0.5
    positions %= grid_size  # Periodic boundary

    # Agent → Field: Deposit density
    field_to_agent.positions = positions
    agent_to_field.positions = positions
    density = agent_to_field.transform((positions, np.ones(num_agents)))

    if step % 20 == 0:
        print(f"Step {step}: max density = {density.max():.2f}")
```

---

## Error Handling

### Common Errors

1. **CrossDomainTypeError**
   ```python
   try:
       interface.transform(invalid_data)
   except CrossDomainTypeError as e:
       print(f"Type mismatch: {e}")
   ```

2. **CrossDomainValidationError**
   ```python
   try:
       interface.validate()
   except CrossDomainValidationError as e:
       print(f"Validation failed: {e}")
   ```

3. **Transform Not Found**
   ```python
   try:
       transform = CrossDomainRegistry.get("unknown", "domain")
   except KeyError as e:
       print(f"Transform not registered: {e}")
   ```

---

## Phase 2 Transforms (New in v0.11.0)

### Audio → Visual

**Purpose**: Audio-reactive visual generation

**API**:
```python
from morphogen.cross_domain import AudioToVisualInterface

transform = AudioToVisualInterface(
    audio_signal,
    sample_rate=44100,
    fft_size=2048,
    mode="spectrum"  # "spectrum", "waveform", "energy", "beat"
)

visual_params = transform(audio_signal)
```

**Modes**:
- `spectrum`: FFT analysis → frequency content, spectral brightness
- `waveform`: Raw waveform for oscilloscope visuals
- `energy`: RMS energy → intensity/emission triggers
- `beat`: Onset detection → beat times

**Use Cases**:
- Music visualization, VJ systems
- Audio-reactive particle emission
- Spectrum-driven color palettes

---

### Field → Audio

**Purpose**: Field-driven audio synthesis

**API**:
```python
from morphogen.cross_domain import FieldToAudioInterface

transform = FieldToAudioInterface(
    field,
    mapping={
        "mean": "frequency",
        "std": "amplitude",
        "gradient_mean": "modulation"
    },
    sample_rate=44100,
    duration=1.0
)

audio_params = transform(field)
```

**Mappings**: mean, std, min, max, range, gradient_mean, gradient_max → frequency, amplitude, modulation, filter_cutoff

**Use Cases**:
- Procedural soundscapes from simulation
- Temperature field → synthesis
- Vorticity → frequency modulation

---

### Terrain ↔ Field

**Purpose**: Bidirectional heightmap/field conversion

**API**:
```python
from morphogen.cross_domain import TerrainToFieldInterface, FieldToTerrainInterface

# Terrain → Field
t2f = TerrainToFieldInterface(heightmap, normalize=True)
field = t2f(heightmap)

# Field → Terrain
f2t = FieldToTerrainInterface(field, height_scale=100.0)
terrain_data = f2t(field)
```

**Use Cases**:
- Procedural terrain generation from noise
- Simulation results → landscape
- Elevation → potential field for PDEs

---

### Vision → Field

**Purpose**: Computer vision features to field conversion

**API**:
```python
from morphogen.cross_domain import VisionToFieldInterface

transform = VisionToFieldInterface(
    image,
    mode="edges"  # "edges", "gradient", "intensity"
)

field = transform(image)
```

**Modes**:
- `edges`: Sobel edge detection → scalar field
- `gradient`: Image gradient → vector field
- `intensity`: Direct grayscale → field

**Use Cases**:
- Edge detection → field patterns
- Optical flow → vector field
- Feature maps → PDE initial conditions

---

### Graph → Visual

**Purpose**: Network graph visualization

**API**:
```python
from morphogen.cross_domain import GraphToVisualInterface

transform = GraphToVisualInterface(
    graph_data={'nodes': [...], 'edges': [...]},
    width=512,
    height=512,
    layout="spring"  # "spring", "circular", "random"
)

visual_data = transform(graph_data)
# Returns: node_positions, edge_list, dimensions
```

**Use Cases**:
- Social network visualization
- Dependency graphs
- Flow networks

---

### Cellular → Field

**Purpose**: Cellular automata state to field

**API**:
```python
from morphogen.cross_domain import CellularToFieldInterface

transform = CellularToFieldInterface(ca_state, normalize=True)
field = transform(ca_state)
```

**Use Cases**:
- Game of Life → PDE initial conditions
- CA patterns → field diffusion
- Discrete state → continuous field

---

## Transform Composition Engine

**New in Phase 2**: Automatic multi-hop pipeline construction

### Automatic Path Finding

```python
from morphogen.cross_domain import find_transform_path

# Find path from source to target domain
path = find_transform_path("terrain", "audio", max_hops=3)
print(path)  # ['terrain', 'field', 'audio']
```

### Pipeline Creation

```python
from morphogen.cross_domain import TransformComposer

composer = TransformComposer(enable_caching=True)

# Automatic routing
pipeline = composer.compose_path("terrain", "audio")

# Explicit routing
pipeline = composer.compose_path("terrain", "audio", via=["field"])

# Execute pipeline
result = pipeline(terrain_data)

# Inspect pipeline
print(pipeline.visualize())  # "terrain → field → audio"
print(pipeline.length)  # 2
```

### Composition Utilities

```python
from morphogen.cross_domain import auto_compose, compose

# Shorthand for automatic composition
pipeline = auto_compose("field", "audio")

# Manual composition of transform instances
from morphogen.cross_domain import FieldToAgentInterface, AgentToFieldInterface

t1 = FieldToAgentInterface(field, positions)
t2 = AgentToFieldInterface(positions, values, field_shape)

composed = compose(t1, t2, validate=True)
result = composed(field_data)
```

### Performance Monitoring

```python
composer = TransformComposer(enable_caching=True)

# ... execute transforms ...

stats = composer.get_stats()
print(f"Transforms executed: {stats['transforms_executed']}")
print(f"Cache hits: {stats['cache_hits']}")
print(f"Cache misses: {stats['cache_misses']}")

composer.clear_cache()  # Reset cache
```

---

## Complete Phase 2 Example

```python
"""
Multi-domain workflow: Terrain → Field → Audio → Visual

Demonstrates automatic pipeline composition and execution.
"""

import numpy as np
from morphogen.cross_domain import TransformComposer, AudioToVisualInterface

# Generate procedural terrain
terrain = np.random.rand(256, 256) * 100.0

# Create composer
composer = TransformComposer()

# Build multi-hop pipeline: Terrain → Field → Audio
pipeline = composer.compose_path("terrain", "audio", via=["field"])

print(pipeline.visualize())  # "terrain → field → audio"

# Execute pipeline
audio_params = pipeline(terrain)

print(f"Generated audio with frequency {audio_params['frequency']:.2f} Hz")

# Continue to visual
# (Audio → Visual requires audio signal, not just parameters)
# This demonstrates the full cross-domain workflow
```

---

## Registry Inspection

```python
from morphogen.cross_domain import CrossDomainRegistry

# List all transforms
all_transforms = CrossDomainRegistry.list_all()
print(f"Total transforms: {len(all_transforms)}")

# Visualize transform graph
print(CrossDomainRegistry.visualize())

# Check if transform exists
has_transform = CrossDomainRegistry.has_transform("field", "audio")

# Get transform metadata
metadata = CrossDomainRegistry.get_metadata("field", "audio")
print(metadata['description'])
print(metadata['use_cases'])

# List transforms for a domain
field_transforms = CrossDomainRegistry.list_transforms("field", direction="source")
audio_inputs = CrossDomainRegistry.list_transforms("audio", direction="target")
```

---

## Domain Translation Gap Analysis

### Current Translation Coverage

**Total Implemented: 18 Transforms**

| Source | Target | Status | Use Case |
|--------|--------|--------|----------|
| Field | Agent | ✅ | Flow field → particle forces |
| Agent | Field | ✅ | Particles → density field |
| Physics | Audio | ✅ | Collision forces → percussion |
| Audio | Visual | ✅ | FFT spectrum → colors |
| Field | Audio | ✅ | Temperature → synthesis |
| Terrain | Field | ✅ | Heightmap → scalar field |
| Field | Terrain | ✅ | Field → landscape generation |
| Vision | Field | ✅ | Edge detection → field |
| Graph | Visual | ✅ | Network → visualization |
| Cellular | Field | ✅ | Game of Life → PDE init |
| Time | Cepstral | ✅ | Audio → DCT (MP3/AAC) |
| Cepstral | Time | ✅ | DCT → audio reconstruction |
| Time | Wavelet | ✅ | Signal → time-frequency |
| Spatial | Spatial | ✅ | Affine transformations |
| Cartesian | Polar | ✅ | Coordinate conversion |
| Polar | Cartesian | ✅ | Coordinate conversion |
| Fluid | Acoustics | ✅ | Pressure → acoustic waves |
| Acoustics | Audio | ✅ | Acoustic → waveform |

**Registered Domains (24+):**
field, agent, audio, visual, rigidbody, acoustics, geometry, circuit, terrain, cellular, signal, noise, temporal, optimization, graph, vision, image, color, palette, io_storage, sparse_linalg, neural, genetic, statemachine, integrators

### Critical Missing Translations

#### 1. Geometry Domain (High Priority)
**Status:** Domain exists, no translations implemented

- **Geometry → Physics**: Mesh → collision geometry
  - *Use Case:* CAD models → rigid body simulation
  - *Impact:* Enables design-to-simulation workflow

- **Geometry → Field**: SDF generation from mesh
  - *Use Case:* Mesh → signed distance field for PDEs
  - *Impact:* Enables geometry-driven fluid simulation

- **Geometry → Visual**: Mesh → renderable scene
  - *Use Case:* Parametric shapes → 3D visualization
  - *Impact:* Complete CAD visualization pipeline

- **Geometry → Agent**: Mesh vertices → spawn points
  - *Use Case:* Surface sampling for particle systems
  - *Impact:* Geometry-driven particle placement

#### 2. Circuit Domain (Medium Priority)
**Status:** Domain mentioned, no translations

- **Circuit → Audio**: Analog circuit modeling
  - *Use Case:* Virtual analog synthesis (Moog, 303)
  - *Impact:* Physics-based audio synthesis
  - *Unique Value:* Impossible in traditional audio engines

- **Circuit → Field**: Voltage/current fields
  - *Use Case:* Circuit simulation visualization
  - *Impact:* Educational and analysis tools

#### 3. Neural Domain (High Priority)
**Status:** Domain registered, no translations

- **Neural → Geometry**: GAN → 3D shapes
  - *Use Case:* AI-generated procedural models
  - *Impact:* ML-driven design

- **Visual → Neural**: Feature extraction
  - *Use Case:* Image → embeddings
  - *Impact:* Vision-based control systems

- **Audio → Neural**: Audio embeddings
  - *Use Case:* Music analysis, classification
  - *Impact:* ML-driven audio processing

#### 4. Temporal/Optimization Domains
**Status:** Domains registered, no translations

- **Temporal → Audio**: Time series → sound
  - *Use Case:* Data sonification
  - *Impact:* Accessible data analysis

- **Optimization → Geometry**: Optimal shapes
  - *Use Case:* Topology optimization results
  - *Impact:* Engineering applications

#### 5. Missing Reverse Translations

- **Audio → Field**: Reverse of Field → Audio
  - *Use Case:* Music → simulation initial conditions
  - *Impact:* Audio-driven generative art

- **Visual → Audio**: Reverse of Audio → Visual
  - *Use Case:* Image sonification
  - *Impact:* Accessibility, data exploration

- **Agent → Audio**: Swarm sonification
  - *Use Case:* Collective behavior → sound
  - *Impact:* Complex system monitoring

- **Physics → Field**: Rigid bodies → fields
  - *Use Case:* Particle positions → density/velocity fields
  - *Impact:* Hybrid particle-field simulation

#### 6. Color/Palette Domain
**Status:** Domain registered, no translations

- **Field → Color**: Data → color mapping
  - *Use Case:* Scientific visualization
  - *Impact:* Better data interpretation

- **Audio → Color**: Spectrum → palette
  - *Use Case:* Synesthesia visualization
  - *Impact:* Audio-visual art

---

## Multi-Hop Translation Chains

### The Power of Composition

Multi-hop translations enable **emergent capabilities** that are impossible with single transforms. The composition system automatically finds paths between domains, creating workflows that were previously impossible.

### Killer Demo Chains

#### 1. **Fluid → Acoustics → Audio** (3-Domain Physics Sonification)
```python
# Already implemented - the "killer demo"
pipeline = composer.compose_path("fluid", "audio", via=["acoustics"])
```

**Workflow:**
1. CFD pressure field → acoustic wave propagation
2. Acoustic waves → audio waveform synthesis
3. Result: Physical sound synthesis from fluid dynamics

**Unique Value:** Aeroacoustic simulation in a single pipeline - impossible in traditional systems where CFD and audio are separate worlds.

**Applications:**
- Wind noise simulation
- Fluid instrument modeling (flutes, organ pipes)
- Environmental sound design

---

#### 2. **Geometry → Field → Audio → Visual** (4-Domain Shape Sonification)
```python
pipeline = composer.compose_path("geometry", "visual", via=["field", "audio"])
```

**Workflow:**
1. Mesh → signed distance field
2. Field statistics → audio parameters (mean → freq, std → amplitude)
3. Audio spectrum → visual colors/particles

**Unique Value:** "See the sound of shape" - morphological sonification with visual feedback

**Applications:**
- Shape comparison through sound
- Accessibility tools for 3D modeling
- Generative art exploring form/sound relationships

---

#### 3. **Vision → Field → Terrain → Physics → Audio** (5-Domain Image-to-Sound)
```python
pipeline = composer.compose_path("vision", "audio",
                                via=["field", "terrain", "physics"])
```

**Workflow:**
1. Image edge detection → scalar field
2. Field → 3D terrain heightmap
3. Terrain → rigid body collision geometry
4. Physics collisions → percussion synthesis

**Unique Value:** Transform images into physically simulated soundscapes

**Applications:**
- Image-driven sound design
- Data sonification with spatial meaning
- Artistic exploration of visual/sonic textures

---

#### 4. **Genetic → Agent → Field → Visual** (Evolutionary Visualization)
```python
# Requires: Genetic → Agent translation (not yet implemented)
pipeline = composer.compose_path("genetic", "visual", via=["agent", "field"])
```

**Workflow:**
1. Evolution parameters → agent behavior
2. Agent swarms → density fields
3. Field → heat map visualization

**Unique Value:** Real-time visualization of evolutionary dynamics

**Applications:**
- Genetic algorithm monitoring
- Swarm intelligence research
- Evolutionary computation visualization

---

#### 5. **Audio → Visual → Field → Terrain** (Music-Generated Worlds)
```python
# Requires: Visual → Field translation (not yet implemented)
pipeline = composer.compose_path("audio", "terrain", via=["visual", "field"])
```

**Workflow:**
1. Music FFT → spectrum visualization
2. Visual → scalar field
3. Field → terrain landscape

**Unique Value:** Generate explorable 3D worlds from music

**Applications:**
- Music visualization in 3D
- Procedural generation driven by audio
- VR experiences from songs

---

#### 6. **Circuit → Audio → Visual → Agent** (Analog-Driven Particles)
```python
# Requires: Circuit → Audio, Visual → Agent translations
pipeline = composer.compose_path("circuit", "agent", via=["audio", "visual"])
```

**Workflow:**
1. Virtual analog circuit → audio waveform
2. Audio spectrum → visual parameters
3. Visual features → agent behavior

**Unique Value:** Physics-based synthesis driving generative systems

**Applications:**
- Modular synthesis visualization
- Circuit-driven art installations
- Educational tools for analog synthesis

---

#### 7. **Temporal → Field → Acoustics → Audio** (Data Sonification via Physics)
```python
# Requires: Temporal → Field translation
pipeline = composer.compose_path("temporal", "audio", via=["field", "acoustics"])
```

**Workflow:**
1. Time series → pressure field
2. Pressure → acoustic wave propagation
3. Acoustic → audio waveform

**Unique Value:** Physically-grounded data sonification

**Applications:**
- Scientific data listening
- Stock market sonification
- Climate data audification

---

#### 8. **Neural → Geometry → Field → Agent** (AI-Driven Swarms)
```python
# Requires: Neural → Geometry, Geometry → Field, Field → Agent translations
pipeline = composer.compose_path("neural", "agent", via=["geometry", "field"])
```

**Workflow:**
1. GAN generates 3D shape
2. Shape → SDF guidance field
3. Field gradients → agent navigation

**Unique Value:** ML-generated forms driving collective behavior

**Applications:**
- AI-designed swarm behaviors
- Procedural animation from neural nets
- Evolved guidance fields

---

#### 9. **Cellular → Field → Terrain → Physics** (CA-Generated Worlds)
```python
pipeline = composer.compose_path("cellular", "physics", via=["field", "terrain"])
```

**Workflow:**
1. Game of Life state → scalar field
2. Field → terrain heightmap
3. Terrain → rigid body physics

**Unique Value:** Emergent patterns become physical spaces

**Applications:**
- Procedurally generated landscapes
- Pattern-based level design
- Physics playgrounds from cellular automata

---

#### 10. **Optimization → Geometry → Physics → Audio** (Optimal Sound)
```python
# Requires: Optimization → Geometry translation
pipeline = composer.compose_path("optimization", "audio",
                                via=["geometry", "physics"])
```

**Workflow:**
1. Topology optimization → optimal shape
2. Shape → collision geometry
3. Physics collisions → percussion

**Unique Value:** Hear the sound of optimized structures

**Applications:**
- Engineering design sonification
- Structural acoustics
- Form-finding through audio feedback

---

### Composition Patterns

#### Pattern 1: Bidirectional Loops
Create feedback systems with bidirectional transforms:

```python
# Field ↔ Agent feedback loop
field → agent (sample)
agent → field (deposit)
# Iterative coupling for emergent behavior
```

**Enabled by:**
- Field ↔ Agent (implemented ✅)
- Field ↔ Terrain (implemented ✅)
- Time ↔ Cepstral (implemented ✅)
- Cartesian ↔ Polar (implemented ✅)

**Future Loops:**
- Audio ↔ Visual (needs Visual → Audio)
- Geometry ↔ Field (needs Geometry → Field)
- Neural ↔ Optimization (needs both directions)

---

#### Pattern 2: Domain Convergence
Multiple sources → single target:

```python
# Multiple paths to audio
field → audio (direct)
physics → audio (events)
circuit → audio (synthesis)
temporal → audio (sonification)

# Result: Unified audio output from diverse sources
```

**Applications:**
- Multi-source sonification
- Hybrid synthesis (physical + circuit models)
- Data-driven music generation

---

#### Pattern 3: Domain Divergence
Single source → multiple targets:

```python
# Field as hub
field → agent (forces)
field → audio (parameters)
field → terrain (landscape)
field → visual (color mapping)

# Result: Synchronized multi-modal outputs
```

**Applications:**
- Synchronized AV performances
- Multi-sensory simulations
- Cross-modal data exploration

---

### Implementation Priorities

Based on unique value and multi-hop enablement:

**Priority 1 (Highest Impact):**
1. **Geometry → Field** - Unlocks 8+ new chains
2. **Visual → Audio** - Completes Audio ↔ Visual bidirectionality
3. **Neural → Geometry** - Enables ML-driven workflows
4. **Circuit → Audio** - Unique physical modeling capability

**Priority 2 (High Value):**
5. **Geometry → Physics** - CAD-to-simulation workflow
6. **Agent → Audio** - Swarm sonification
7. **Temporal → Field** - Data-driven simulation
8. **Field → Color** - Better scientific visualization

**Priority 3 (Nice to Have):**
9. **Optimization → Geometry** - Engineering workflows
10. **Genetic → Agent** - Evolutionary visualization

---

## Automatic Path Finding

The composition system uses BFS to find shortest paths:

```python
from morphogen.cross_domain import find_transform_path

# Automatic routing
path = find_transform_path("terrain", "audio", max_hops=3)
print(path)  # ['terrain', 'field', 'audio']

# Create and execute
pipeline = composer.compose_path("terrain", "audio")
result = pipeline(terrain_data)
```

**Path Caching:** Frequently used paths are cached for performance.

**Path Validation:** Ensures type compatibility at each hop.

**Path Visualization:**
```python
print(pipeline.visualize())  # "terrain → field → audio"
print(f"Pipeline length: {pipeline.length}")  # 2 hops
```

---

## Future Extensions

### Near-Term (v0.12)

- **Geometry → Physics**: Mesh → collision geometry
- **Geometry → Field**: SDF generation
- **Visual → Audio**: Image sonification
- **Circuit → Audio**: Analog modeling

### Mid-Term (v0.13)

- **Neural → Geometry**: GAN → 3D shapes
- **Optimization → Geometry**: Optimal forms
- **Temporal → Audio**: Time series sonification
- **Agent → Audio**: Swarm sonification

### Long-Term (v1.0+)

- **Pattern → Audio**: Euclidean rhythms → events
- **ML → Any**: Neural operator integration
- **Quantum → Field**: Quantum simulation visualization
- **Financial → Visual**: Market data visualization

---

## References

- **ADR-002**: Cross-Domain Architectural Patterns
- **SPECIFICATION.md**: Language specification (compose/link syntax)
- **examples/cross_domain/**: Complete working examples
  - `01_transform_composition.py`: Pipeline composition and path finding
  - `02_audio_reactive_visuals.py`: Audio analysis for visual generation
  - `fluid_acoustics_audio.py`: 3-domain killer demo

---

**Last Updated:** 2025-11-21 (Multi-Hop Analysis Complete)
**Maintainer:** Morphogen Development Team
**Version:** v0.11.0 (Cross-Domain Infrastructure Phase 2)
