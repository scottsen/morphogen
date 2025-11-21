# Agents Domain: VFX Integration Quick Reference

## Domain Overview at a Glance

The Morphogen agents domain provides **sparse particle/agent simulation** with:
- NumPy-based backend (CPU execution)
- MLIR dialect for future compilation
- Flexible per-agent properties
- Field sampling and force calculations
- Basic circle-based visualization

**Status**: Foundation-level complete, VFX features sparse

---

## Key Files & Their Roles

| File | Size | Purpose | Integration Point |
|------|------|---------|-------------------|
| `morphogen/stdlib/agents.py` | 544 LOC | Core agent system | Add new operations (emit, lifetime, collision) |
| `morphogen/mlir/dialects/agent.py` | 526 LOC | High-level ops | Add behavior types (fade, explode, vortex) |
| `morphogen/mlir/lowering/agent_to_scf.py` | 424 LOC | Lowering to loops | Extend behavior implementation |
| `morphogen/stdlib/visual.py` | 781 LOC | Rendering | Add `particles()` and `trails()` functions |
| `tests/test_agents_*.py` | Multi-file | Coverage | Add VFX behavior tests |

---

## Core Data Structure

```python
class Agents:
    count: int                           # Total allocated
    alive_mask: np.ndarray (N,) bool    # Which agents active
    properties: Dict[str, np.ndarray]   # Flexible per-agent data
        - 'pos': (N, 2) float32 - position
        - 'vel': (N, 2) float32 - velocity
        - 'mass': (N,) float32 - mass
        - ...any custom properties...
```

### Custom Properties Available for VFX
```python
# Add these to properties dict for effects:
'age': (N,) float32          # Lifetime counter
'lifetime': (N,) float32     # Max age before death
'alpha': (N,) float32        # 0.0-1.0 transparency
'rotation': (N,) float32     # Angle in radians
'angular_velocity': (N,) float32  # Rotation speed
'scale': (N,) float32        # Size multiplier
'color': (N, 3) float32      # RGB per agent
'trail_history': (N, T, 2) float32  # Previous positions
'sprite_id': (N,) int32      # Sprite index
```

---

## Quick Integration Checklist

### Phase A: Immediate (1-2 days)

- [ ] **Trail Rendering** - Complete existing `visual.agents(..., trail=True)`
  - File: `morphogen/stdlib/visual.py` line 399-401
  - Implementation: Draw faded line segments from trail history
  
- [ ] **Alpha Support** - Add transparency property
  - File: `morphogen/stdlib/visual.py` line 530-537 (circle drawing)
  - Implementation: Multiply alpha into color channels
  
- [ ] **Rotation Property** - Add orientation rendering
  - File: `morphogen/stdlib/visual.py` visual.agents()
  - Implementation: Draw arrow or line at rotation angle
  
- [ ] **Lifetime Filtering** - Auto-cleanup utility
  - File: `morphogen/stdlib/agents.py` after line 200
  - Implementation: Helper to filter by age < lifetime

### Phase B: Short-term (1 week)

- [ ] **Emission System** - `agents.emit()` operation
  - File: `morphogen/stdlib/agents.py` new method
  - Creates particles from template, supports burst/continuous
  
- [ ] **Enhanced Rendering** - New `visual.particles()` function
  - File: `morphogen/stdlib/visual.py` new function (after agents)
  - Supports: sprite mode, motion blur, soft particles
  
- [ ] **Global Forces** - Apply force fields
  - File: `morphogen/stdlib/agents.py` new method
  - Implements wind, gravity, damping

### Phase C: Medium-term (2-3 weeks)

- [ ] **Collision Detection** - Agent-agent/boundary collisions
  - File: `morphogen/stdlib/agents.py` new method
  - Use spatial hashing infrastructure
  
- [ ] **Behavior Templates** - Extended behavior types
  - File: `morphogen/mlir/lowering/agent_to_scf.py` lines 322-394
  - Add: fade, explode, vortex, damping behaviors
  
- [ ] **Particle Pools** - Efficient allocation reuse
  - File: `morphogen/stdlib/agents.py` new class
  - Pre-allocate and reuse agent slots

---

## Current API Reference

### Agent Allocation
```python
agents.alloc(count: int, properties: Dict[str, Any]) -> Agents
```

### Agent Operations
```python
agents.map(agents, property_name, func) -> np.ndarray
agents.filter(agents, property_name, condition) -> Agents
agents.reduce(agents, property_name, operation) -> scalar
agents.compute_pairwise_forces(agents, radius, force_func) -> np.ndarray
agents.sample_field(agents, field2d, position_property) -> np.ndarray
```

### Property Access
```python
agents.get(property_name) -> np.ndarray          # Get alive agents only
agents.get_all(property_name) -> np.ndarray      # Get all agents
agents.set(property_name, values) -> Agents      # In-place modification
agents.update(property_name, values) -> Agents   # Returns new instance
```

### Visualization
```python
visual.agents(agents, width=512, height=512, color_property=None, 
             size_property=None, trail=False, trail_length=10) -> Visual

visual.colorize(field2d, palette="fire") -> Visual
visual.composite(*visuals, mode="over") -> Visual
visual.output(visual, path) -> None
visual.display(frame_generator) -> None
visual.video(frames, path, fps=30) -> None
```

---

## Example: Simple Particle Effect (Current)

```python
from morphogen.stdlib.agents import agents
from morphogen.stdlib.visual import visual
import numpy as np

# Create particles
particles = agents.alloc(count=100, properties={
    'pos': np.random.rand(100, 2) * 100,
    'vel': np.random.randn(100, 2) * 0.1,
})

# Simulate 50 frames
for step in range(50):
    # Physics
    pos = particles.get('pos')
    vel = particles.get('vel')
    
    # Gravity
    vel[:, 1] -= 0.001
    particles = particles.update('vel', vel)
    
    # Position update
    new_pos = pos + vel
    particles = particles.update('pos', new_pos)
    
    # Boundary wrap
    particles.get('pos')[:] = particles.get('pos') % 100
    
    # Render
    vis = visual.agents(particles, color=(1, 1, 1), size=3)
    visual.output(vis, f'frame_{step:03d}.png')
```

---

## Example: Proposed Future Pattern (With VFX)

```python
# Create emitter
emitter = agents.alloc(count=1, properties={
    'pos': [(50, 50)],
    'emission_rate': [20],
})

particles = agents.alloc(count=0, properties={
    'pos': np.empty((0, 2), dtype=np.float32),
    'vel': np.empty((0, 2), dtype=np.float32),
    'age': np.empty(0, dtype=np.float32),
    'alpha': np.empty(0, dtype=np.float32),
    'rotation': np.empty(0, dtype=np.float32),
})

# Simulate with emission & effects
for step in range(100):
    # Emit new particles
    rate = int(emitter.get('emission_rate')[0])
    if rate > 0:
        new_particles = agents.emit(
            emitter,
            rate=rate,
            template={
                'vel': np.random.randn(rate, 2) * 2,
                'age': np.zeros(rate),
                'alpha': np.ones(rate),
                'lifetime': np.ones(rate) * 2.0,
            }
        )
        particles = agents.merge(particles, new_particles)
    
    # Apply gravity
    vel = particles.get('vel')
    vel[:, 1] -= 0.05
    particles = particles.update('vel', vel)
    
    # Update position
    pos = particles.get('pos')
    pos = pos + vel * 0.016
    particles = particles.update('pos', pos)
    
    # Age particles
    age = particles.get('age') + 0.016
    particles = particles.update('age', age)
    
    # Fade out based on age
    lifetime = particles.get('lifetime')
    alpha = 1.0 - (age / lifetime)
    alpha = np.clip(alpha, 0, 1)
    particles = particles.update('alpha', alpha)
    
    # Remove dead
    particles = particles.filter('age', lambda a: a < lifetime)
    
    # Render with effects
    vis = visual.particles(
        particles,
        alpha_property='alpha',
        rotation_property='rotation',
        blend_mode='add',
        motion_blur=True
    )
    visual.output(vis, f'frame_{step:03d}.png')
```

---

## MLIR Phase 4 Operations

Current high-level operations that lower to memref:

```
morphogen.agent.spawn(count, pos_x, pos_y, vel_x, vel_y, state)
  → memref.alloc(%count, %c5) + initialization loop

morphogen.agent.update(agents, index, property, value)
  → memref.store %value, %agents[%index, %property]

morphogen.agent.query(agents, index, property)
  → memref.load %agents[%index, %property]

morphogen.agent.behavior(agents, behavior_type, params)
  → scf.for loop with behavior logic
    - "move": x += vx; y += vy
    - "bounce": check bounds, reverse velocity
    - "seek": move towards target
```

### Extending Behaviors
File: `morphogen/mlir/lowering/agent_to_scf.py` lines 322-394

Add new behavior types in `_lower_agent_behavior()`:
```python
elif behavior_type == "fade":
    # Decrease alpha over time
    alpha = memref.LoadOp(agents, [i, alpha_idx]).result
    new_alpha = arith.MulFOp(alpha, decay_rate).result
    memref.StoreOp(new_alpha, agents, [i, alpha_idx])
    
elif behavior_type == "vortex":
    # Rotate around center with spiral inward
    # ... orbital mechanics ...
```

---

## Testing Strategy

### Existing Test Coverage
- `tests/test_agents_basic.py` - Allocation, properties
- `tests/test_agents_operations.py` - Map, filter, reduce
- `tests/test_agents_forces.py` - Pairwise forces
- `tests/test_agents_integration.py` - Simulation loops
- `tests/test_io_integration.py` - Particle animation workflows

### Where to Add VFX Tests
1. **Trail tests** in `test_visual_operations.py`
   - Verify trail history recording
   - Check fade-out effect
   
2. **Emission tests** in new `test_particle_effects.py`
   - Test emit operation
   - Verify rate control
   - Check lifetime management
   
3. **Behavior tests** in `test_agent_dialect.py`
   - Test fade behavior
   - Test explode behavior
   - Test vortex behavior

---

## Performance Notes

### Spatial Hashing (Current O(n) Performance)
- Grid cell size = force radius
- Neighbor offset: 3x3 grid (2D) or 3x3x3 (3D)
- Good for: 1000+ agents with local interactions

### Rendering (Current Bottleneck)
- Per-pixel circle rasterization: O(n * size²)
- Optimization: Use NumPy vectorized operations
- Future: GPU instancing, compute shaders

### Property Storage
- Dict of numpy arrays: cache-friendly
- Good for vectorized operations
- Easy to add custom properties

---

## Common Pitfalls to Avoid

1. **Alive Mask Synchronization**
   - Always use `alive_mask` when filtering
   - `get()` returns only alive agents
   - `get_all()` returns all including dead

2. **Array Shape Mismatches**
   - Ensure properties have shape (count, ...) first dimension
   - Check shape when updating
   - Use broadcasting carefully

3. **Immutability in Updates**
   - `update()` returns new Agents instance
   - Don't forget assignment: `agents = agents.update(...)`
   - Use copy() for defensive programming

4. **Trail History Size**
   - Allocate with max size upfront
   - Use circular buffer pattern
   - Monitor memory with many agents

5. **Rendering Bounds**
   - Auto-compute or specify bounds explicitly
   - Include padding for velocity visualization
   - Handle edge case of single agent

---

## Related Domains

### Field Coupling
- `agents.sample_field(agents, field2d, 'pos')`
- Bilinear interpolation at agent positions
- Use for fluid coupling, temperature influence

### Optimization Domain
- `optimization.particle_swarm()` - Already uses "particles"
- Different purpose: optimization not visualization
- Can share infrastructure if refactored

### Audio Domain
- Can drive particle parameters (freq → velocity, etc.)
- Sonify particle positions (fun cross-domain)
- See: `examples/showcase/05_audio_visualizer.py`

### Cellular Automata
- Could emit particles based on cell state
- Combined discrete/continuous simulation
- Boundary interaction patterns

---

## Resources

### Documentation
- `ARCHITECTURE.md` - System-wide design
- `ECOSYSTEM_MAP.md` - Domain relationships
- `SPECIFICATION.md` - Full language spec
- `STATUS.md` - Current implementation status

### Code Examples
- `examples/phase4_agent_operations.py` - MLIR examples
- `examples/agents/boids.ccdsl` - Flocking simulation
- `examples/showcase/02_physics_visualizer.py` - Physics demo
- `tests/test_io_integration.py::test_particle_animation_workflow()`

### Related Work
- Particle systems: Classic GPU implementation patterns
- Spatial hashing: Standard acceleration structure
- Agent-based modeling: Mesa, NetLogo, AnyLogic
- VFX tools: Houdini (COPs), RealityKit, Unreal Niagara

