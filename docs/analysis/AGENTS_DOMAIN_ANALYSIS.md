# Comprehensive Agents Domain Analysis - Morphogen DSL

## Executive Summary

The Morphogen agents domain is a **sparse particle/agent simulation system** built with NumPy backend and MLIR dialect support. It provides foundation-level operations for agent allocation, property mapping, filtering, force calculations, and field coupling, but currently lacks advanced **visual effects (VFX) and particle effect systems**.

---

## 1. Agents Domain Location & Structure

### Core Files
- **Primary Implementation**: `/home/user/morphogen/morphogen/stdlib/agents.py` (544 lines)
- **MLIR Dialect**: `/home/user/morphogen/morphogen/mlir/dialects/agent.py` (526 lines)
- **MLIR Lowering**: `/home/user/morphogen/morphogen/mlir/lowering/agent_to_scf.py` (424 lines)
- **Examples**: `/home/user/morphogen/examples/agents/boids.ccdsl`, `/home/user/morphogen/examples/phase4_agent_operations.py`
- **Tests**: `/home/user/morphogen/tests/test_agents_*.py` (5+ test files)

### Directory Structure
```
morphogen/
├── morphogen/stdlib/agents.py          # NumPy backend implementation
├── morphogen/mlir/dialects/agent.py    # High-level agent operations
├── morphogen/mlir/lowering/agent_to_scf.py  # Lowering to control flow
├── morphogen/stdlib/visual.py          # Visualization (with agent rendering)
├── examples/agents/boids.ccdsl     # Example simulation
├── examples/phase4_agent_operations.py  # MLIR phase 4 examples
└── tests/test_agents_*.py          # Comprehensive tests
```

---

## 2. Current Functionality

### 2.1 Core Agent System (`agents.py`)

#### Agents Class (Sparse Particle Collection)
```python
class Agents:
    """Sparse agent collection with per-agent properties."""
    
    Properties:
    - count: Total allocated agents
    - alive_mask: Boolean mask for active/dead agents
    - properties: Dict[str, np.ndarray] - per-agent data
    
    Methods:
    - get(property): Get values for alive agents
    - get_all(property): Get values for ALL agents
    - set/update: Modify agent properties
    - copy(): Deep copy with immutability
    - alive_count: Count of active agents
```

#### AgentOperations Namespace
Core operations available through `agents.*`:

| Operation | Purpose | Notes |
|-----------|---------|-------|
| `alloc(count, properties)` | Create agent population | Handles scalar broadcast, arrays |
| `map(agents, property, func)` | Apply function per agent | Vectorized or element-wise |
| `filter(agents, property, condition)` | Keep agents matching condition | Updates alive_mask |
| `reduce(agents, property, operation)` | Aggregate (sum, mean, min, max, prod) | On alive agents only |
| `compute_pairwise_forces(agents, radius, force_func, ...)` | N-body interactions | O(n) spatial hashing or O(n²) brute force |
| `sample_field(agents, field, position_property)` | Bilinear interpolation at agent positions | Agent-field coupling |

#### Force Calculation
- **Spatial Hashing**: O(n) performance with grid-based neighbor queries
- **Brute Force**: O(n²) fallback for small counts or 1D
- **Customizable**: User provides `force_func(pos_i, pos_j, [mass_i, mass_j])` callback

### 2.2 MLIR Dialect (`agent.py`)

#### High-Level Operations
- `morphogen.agent.spawn`: Create agents with initial properties
- `morphogen.agent.update`: Modify agent properties (position, velocity, state)
- `morphogen.agent.query`: Read agent property values
- `morphogen.agent.behavior`: Apply movement/interaction rules

#### Type System
```
AgentType: !morphogen.agent<T>
  - Phase 4: Opaque type (UnrealizedConversionCast)
  - Future: Proper IRDL dialect definition
```

#### Standard Property Layout
```
Agent Properties (5 base):
  [0] position_x
  [1] position_y
  [2] velocity_x
  [3] velocity_y
  [4] state (scalar custom value)
  [5+] custom properties
```

### 2.3 Lowering Pass (`agent_to_scf.py`)

Transforms high-level agent ops to executable code:

```
morphogen.agent.spawn → memref.alloc + initialization loop
morphogen.agent.update → memref.store (in-place mutation)
morphogen.agent.query → memref.load
morphogen.agent.behavior → scf.for loop + behavior logic
```

#### Behavior Types Implemented
1. **"move"**: Simple position += velocity integration
2. **"bounce"**: Boundary collision with velocity reversal
3. **"seek"**: Move towards target position with speed control
4. **Default**: Falls back to move

### 2.4 Visual Rendering (`visual.py`)

#### Agent Visualization
```python
visual.agents(
    agents, 
    width=512, height=512,
    position_property='pos',
    color_property=None,          # Optional: colorize by property
    size_property=None,           # Optional: size by property
    color=(1.0, 1.0, 1.0),        # Default RGB color
    size=2.0,                     # Default point size (pixels)
    palette="viridis",            # Color palette for property mapping
    background=(0.0, 0.0, 0.0),
    bounds=None,                  # Auto-compute if None
    trail=False,                  # TRAIL SUPPORT (but not fully used)
    trail_length=10,              # Trail point count
    trail_alpha=0.5               # Trail transparency
) → Visual
```

**Current Rendering**:
- Simple circle rasterization (per-pixel loop)
- Color mapping from property values
- Size scaling from property values
- **Note**: Trail parameters are defined but not implemented

#### Visualization Output
- `visual.colorize(field, palette)` - Convert field to color image
- `visual.output(visual, path)` - Save to PNG/JPG
- `visual.display(frame_generator)` - Interactive pygame window
- `visual.video(frames, path)` - Export to MP4/GIF
- `visual.composite(*layers, mode)` - Blend multiple layers (over, add, multiply, screen, overlay)

---

## 3. Agent Structure & Data Flow

### Data Layout
```
Agents (sparse collection):
  ├── properties: Dict[str, ndarray]
  │   ├── 'pos': (N, 2) float32 - 2D positions
  │   ├── 'vel': (N, 2) float32 - 2D velocities
  │   ├── 'mass': (N,) or (N, 1) float32
  │   ├── 'color': (N, 3) float32 - RGB
  │   ├── 'age': (N,) float32 - lifetime
  │   ├── 'size': (N,) float32 - render scale
  │   └── ...custom properties...
  │
  ├── alive_mask: (N,) bool - which agents are active
  └── count: int - total allocated

Field2D (dense grid):
  ├── data: (height, width) or (height, width, channels)
  ├── shape: (height, width)
  ├── dx, dy: grid spacing
```

### Typical Simulation Loop
```python
# Phase 1: Allocation
agents = agents.alloc(count=1000, properties={
    'pos': init_positions,
    'vel': init_velocities,
    'age': np.zeros(1000),
    'color': np.ones((1000, 3))
})

# Phase 2: Update per timestep
for step in range(num_steps):
    # Compute forces
    forces = agents.compute_pairwise_forces(agents, radius=10.0, force_func=gravity)
    
    # Update velocities
    new_vel = agents.get('vel') + forces * dt
    agents = agents.update('vel', new_vel)
    
    # Update positions
    new_pos = agents.get('pos') + new_vel * dt
    agents = agents.update('pos', new_pos)
    
    # Sample field influence
    field_values = agents.sample_field(agents, temperature_field)
    
    # Age agents and filter dead ones
    ages = agents.get('age') + dt
    agents = agents.update('age', ages)
    agents = agents.filter('age', lambda a: a < max_age)
    
    # Render
    vis = visual.agents(agents, color_property='vel')
    visual.output(vis, f"frame_{step:04d}.png")
```

---

## 4. Particle Effects/VFX Integration Opportunities

### 4.1 Missing Capabilities

Currently **NOT implemented**:
- Particle emission systems
- Life/death particle cycles with age-based removal
- Velocity-based trails (marked as `trail=False` default)
- Sprite/glyph rendering (only circles)
- Blend modes for particles (only for compositing)
- Velocity-direction arrows/lines
- Rotation/orientation per agent
- Rotation during render based on velocity
- GPU particle simulation
- Instanced rendering
- Particle pools/reuse
- Explosion/burst emission patterns
- Wind/global forces field
- Collision detection between agents
- Soft particles / fade-out at boundaries
- Velocity streaks / motion blur

### 4.2 Integration Points (Where VFX Can Be Added)

#### A. **In the Agents Domain** (`agents.py`)
Add new property management and operations:

```python
# New properties for particle effects
'emission_rate': (N,) float32 - particles/timestep per agent
'lifetime': (N,) float32 - max age before death
'rotation': (N,) float32 - orientation angle
'angular_velocity': (N,) float32 - rotation speed
'scale': (N,) float32 - size over lifetime curve
'alpha': (N,) float32 - transparency
'trail_history': (N, max_trail_length, 2) - positional history
'sprite_id': (N,) int32 - which sprite/glyph to render

# New operations
agents.emit(parent_agents, emission_rate, ...)  # Create child particles
agents.apply_lifetime(agents, max_age, ...)  # Auto-kill old particles
agents.apply_force_field(agents, force_field, ...)  # Global forces
agents.collision_detect(agents, radius, callback)  # Pairwise collisions
agents.sort_by_property(agents, property)  # For depth sorting
```

#### B. **In the Visual Domain** (`visual.py`)
Enhance agent rendering:

```python
# Particle effect rendering modes
visual.particles(agents, 
    mode='sprite',              # sprite, glyph, quad, line, trail
    sprite_sheet_path='particles.png',
    sprite_size=16,
    sprite_uv_property='sprite_id',
    rotation_property='rotation',
    scale_property='scale',
    alpha_property='alpha',
    blend_mode='add',           # add, multiply, screen, over
    motion_blur=True,
    motion_blur_samples=4,
    soft_particle_fade=True,
    fade_distance=10.0
) → Visual

# Trail rendering
visual.trails(agents,
    position_property='pos',
    trail_history_property='trail_history',
    color_property='color',
    alpha_fade=True,            # Fade older trail points
    line_width=1.0
) → Visual
```

#### C. **In the Behavior System** (`agent_to_scf.py`)
Add behavior templates:

```
Behavior Types (extend "move", "bounce", "seek"):
  - "explode": Burst particles outward from center
  - "gravity": Apply constant downward force
  - "wind": Apply directional force
  - "vortex": Circular orbital motion
  - "fade": Decrease alpha over lifetime
  - "trail": Record path history
  - "collision": Bounce off boundaries/agents
  - "damping": Velocity decay
```

#### D. **Workflow Integration**
New simulation patterns:

```python
# Fire/smoke simulation
emitter = Agents(count=1, properties={
    'pos': [(width/2, height/2)],
    'emission_rate': [50],
    'lifetime': [1.0],
})

# Loop: Emit → Update → Filter → Render
for step in range(steps):
    # Emit new particles from emitter
    new_particles = agents.emit(
        emitter,
        rate=emitter.get('emission_rate')[0],
        template_properties={
            'pos': particle_offset,
            'vel': velocity_distribution,
            'color': color_by_temperature,
        }
    )
    
    # Merge with existing particles
    particles = Agents.merge(particles, new_particles)
    
    # Apply global force (wind)
    vel = particles.get('vel')
    vel[:, 0] += wind_force * dt
    particles = particles.update('vel', vel)
    
    # Update position
    pos = particles.get('pos') + particles.get('vel') * dt
    particles = particles.update('pos', pos)
    
    # Age and remove dead
    age = particles.get('age') + dt
    particles = particles.update('age', age)
    particles = particles.filter('age', lambda a: a < 1.0)
    
    # Render with effects
    effect_vis = visual.particles(particles, mode='sprite', blend_mode='add')
    trail_vis = visual.trails(particles)
    combined = visual.composite(effect_vis, trail_vis, mode='add')
    visual.output(combined, f'frame_{step:04d}.png')
```

---

## 5. MLIR Phase 4 Implementation Status

### Current Phase 4 Agents
```
Operations:
  ✓ morphogen.agent.spawn - Allocate agents with properties
  ✓ morphogen.agent.update - Modify properties
  ✓ morphogen.agent.query - Read properties
  ✓ morphogen.agent.behavior - Apply rules
  
Behaviors:
  ✓ move - position += velocity
  ✓ bounce - boundary reflection
  ✓ seek - move towards target
  
Lowering:
  ✓ Converts to memref arrays
  ✓ Implements scf.for loops
  ✓ Handles property store/load
```

### Phase 5+ (Future)
- Audio operations (ALREADY IN DEVELOPMENT)
- JIT/AOT compilation
- **GPU particle simulation** (potential)
- **Advanced particle effects** (potential)

---

## 6. Test Coverage & Examples

### Test Files
```
tests/
├── test_agents_basic.py        # Allocation, properties, update
├── test_agents_operations.py   # Map, filter, reduce
├── test_agents_forces.py       # Pairwise forces, spatial hashing
├── test_agents_integration.py  # Runtime integration, simulation loops
├── test_agent_dialect.py       # MLIR operations
└── test_io_integration.py      # Particle animation workflows
```

### Example Implementations
- `examples/phase4_agent_operations.py`: 8 comprehensive MLIR examples
- `examples/agents/boids.ccdsl`: Flocking simulation DSL
- `examples/showcase/02_physics_visualizer.py`: Particle-based physics

### Existing Particle-Related Code
```python
# In tests/test_io_integration.py:
def test_particle_animation_workflow():
    """Complete particle animation workflow."""
    # Initialize, simulate, render 20 frames of 50 particles
    # Current: simple position update + wrapping

# In tests/test_agents_integration.py:
def test_simple_particle_simulation():
    """Particle simulation with position update."""
    
def test_filter_out_of_bounds_particles():
    """Filter particles leaving bounds."""
```

---

## 7. Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│              User DSL / Python API                      │
│  agents.alloc() | agents.map() | agents.filter()       │
└────────────────────────────┬────────────────────────────┘
                             │
        ┌────────────────────┴────────────────────┐
        │                                         │
┌───────▼────────────────────┐   ┌───────────────▼──────────┐
│  NumPy Backend             │   │  MLIR Dialect (Phase 4)   │
│  (agents.py)               │   │  (agent.py)               │
│                            │   │                           │
│ • Agents class             │   │ • agent.spawn             │
│ • Properties (dict)        │   │ • agent.update            │
│ • Alive mask               │   │ • agent.query             │
│ • Operations: map, filter, │   │ • agent.behavior          │
│   reduce, forces, sample   │   │ • Behaviors: move,bounce, │
│                            │   │             seek, ...     │
└───────┬────────────────────┘   └───────┬──────────────────┘
        │                                 │
        │      ┌────────────────────────┐ │
        │      │  MLIR Lowering Pass    │ │
        │      │  (agent_to_scf.py)     │ │
        │      │                        │ │
        │      │ Transforms to:         │ │
        │      │ • memref.alloc/store   │ │
        │      │ • scf.for loops        │ │
        │      │ • arith operations     │ │
        │      └────────────────────────┘ │
        │                                 │
        └─────────────────┬───────────────┘
                          │
        ┌─────────────────▼─────────────────┐
        │   Visual Rendering (visual.py)    │
        │                                   │
        │ • visual.agents() - circles       │
        │ • visual.colorize() - field       │
        │ • visual.composite() - blending   │
        │ • visual.output() - save/display  │
        │ • visual.video() - export         │
        │                                   │
        │ (Trails marked but not fully      │
        │  implemented)                     │
        └───────────────────────────────────┘
```

---

## 8. Recommended Integration Plan

### Phase A: Core Particle Effects (Immediate)
1. **Age/Lifetime Management**
   - Add `'lifetime'` property auto-management
   - Extend `filter()` to support lambda with age checking
   - Auto-removal of dead particles

2. **Trail System**
   - Implement `agents.record_trail()` to populate trail history
   - Complete `visual.agents(..., trail=True)` rendering
   - Add fade-out effect for older trail points

3. **Enhanced Rendering**
   - Add sprite/texture support (optional: sprite sheets)
   - Implement rotation per agent
   - Add motion blur (sample previous positions)
   - Implement alpha/transparency blending

### Phase B: Emission & Dynamics (Short-term)
4. **Particle Emission**
   - Add `agents.emit(parent_agents, rate, template)` operation
   - Support burst/continuous emission patterns
   - Velocity distribution options (cone, sphere, sphere surface)

5. **Global Forces**
   - Add `agents.apply_forces(agents, force_field)` for wind/gravity
   - Implement damping/drag
   - Collision response framework

6. **Enhanced Behaviors**
   - Implement behavior trees for complex patterns
   - Add more behavior templates (explode, vortex, fade)
   - Behavior parameter system

### Phase C: Advanced VFX (Medium-term)
7. **GPU Acceleration**
   - Port particle simulation to SPIR-V/Metal
   - Instanced rendering support
   - Compute shaders for forces

8. **Advanced Visual Effects**
   - Soft particles (fade at boundaries)
   - Depth sorting/OIT
   - Screen-space effects (bloom, glow)

9. **Physics Integration**
   - Full collision detection (agent-agent, agent-boundary)
   - Constraint systems
   - Compound particles

---

## 9. Code Examples

### Current Usage Pattern
```python
from morphogen.stdlib.agents import agents
from morphogen.stdlib.visual import visual
import numpy as np

# Create 100 agents
a = agents.alloc(
    count=100,
    properties={
        'pos': np.random.rand(100, 2) * 100,
        'vel': np.random.randn(100, 2) * 0.1
    }
)

# Simulate
for t in range(100):
    # Update position
    pos = a.get('pos')
    vel = a.get('vel')
    pos = pos + vel * 0.1
    a = a.update('pos', pos)
    
    # Render
    vis = visual.agents(a, color=(1, 1, 1), size=3)
    visual.output(vis, f'frame_{t:03d}.png')
```

### Proposed Future Usage
```python
# Emission system
emitter = agents.alloc(count=1, properties={
    'pos': [(50, 50)],
    'emission_rate': [20],  # particles per frame
})

particles = agents.alloc(count=0, properties={...})

for t in range(100):
    # Emit new particles
    new = agents.emit(
        emitter, 
        rate=20,
        template={'vel': np.random.randn(20, 2) * 2}
    )
    particles = Agents.merge(particles, new)
    
    # Apply gravity
    vel = particles.get('vel')
    vel[:, 1] -= 0.1  # gravity
    particles = particles.update('vel', vel)
    
    # Update position with lifetime
    pos = particles.get('pos') + vel * 0.1
    particles = particles.update('pos', pos)
    
    # Age and remove
    age = particles.get('age') + 0.1
    particles = particles.update('age', age)
    particles = particles.filter('age', lambda a: a < 3.0)
    
    # Render with effects
    vis = visual.particles(
        particles,
        mode='sprite',
        alpha_property='alpha',
        blend_mode='add',
        motion_blur=True
    )
    visual.output(vis, f'frame_{t:03d}.png')
```

---

## 10. Key Findings & Recommendations

### Strengths
- ✓ Solid foundational agent system (NumPy + MLIR)
- ✓ Flexible property system (arbitrary per-agent data)
- ✓ Efficient spatial hashing for forces
- ✓ Field coupling via bilinear interpolation
- ✓ Good test coverage
- ✓ MLIR integration for JIT compilation

### Gaps (VFX/Particle Focus)
- ✗ No particle emission system
- ✗ No automatic lifetime management
- ✗ Trail visualization marked but not implemented
- ✗ Basic circle-only rendering (no sprites, no glyphs)
- ✗ No rotation/orientation support
- ✗ No velocity-direction visualization
- ✗ No collision detection
- ✗ No motion blur or particle effects
- ✗ No GPU particle simulation

### Quick Wins (Immediate Integration)
1. **Complete trail rendering** - Already marked in API, just needs implementation
2. **Rotation property** - Add 'rotation' field, render arrow/line direction
3. **Alpha/transparency** - Implement 'alpha' property with proper blending
4. **Velocity visualization** - Arrow glyphs or velocity-colored particles

### Strategic Recommendations
1. **Keep NumPy backend simple** - Focus on CPU-side particle pools
2. **Extend visual.py** - Add `particles()` function with sprite support
3. **Add emission helpers** - Utility functions for burst/continuous patterns
4. **Consider agent pools** - Optimize allocation/deallocation cycles
5. **Plan GPU port** - Design particle system with GPU in mind from start

---

## Appendix: File Sizes

```
agents.py           544 lines
agent.py (dialect)  526 lines
agent_to_scf.py     424 lines
visual.py           781 lines
field.py            690 lines
-----
Total stdlib: 12,075 lines

Key ratios:
- Agents: 544/12075 = 4.5% of stdlib
- Visual: 781/12075 = 6.5% of stdlib
- Together: 11% of stdlib
```

This analysis provides a complete roadmap for integrating particle effects into the Morphogen agents domain.
