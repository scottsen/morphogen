# Interactive Physics Sandbox - Cross-Domain Demo

**2D physics simulation with real-time visualization**

## Overview

This example demonstrates a comprehensive 2D physics sandbox that combines rigid body simulation with visual rendering. It showcases multiple physics scenarios including gravity, collisions, friction, and many-body dynamics.

## Domains Integrated

- **RigidBody**: 2D physics simulation with collision detection and response
- **Field**: Velocity field visualization
- **Palette**: Color mapping for physical quantities
- **Visual**: Rendering and composition
- **Image**: Output generation

## Physics Scenarios

### Scenario 1: Falling Objects
Demonstrates gravity and elastic collisions with bouncing balls.

**Physics features**:
- Gravitational acceleration
- Elastic collisions (high restitution)
- Circle-circle collisions
- Ground contact

**Visual features**:
- Color-coded velocity (blue=slow, red=fast)
- Real-time rendering
- Trajectory visualization

### Scenario 2: Box Stack
Shows friction and stability with stacked boxes.

**Physics features**:
- Friction coefficients
- Contact forces
- Static equilibrium
- Box-box collisions
- Moment of inertia

**Visual features**:
- Box rotation visualization
- Velocity color mapping
- Stability analysis through frames

### Scenario 3: Particle Rain
Many-body dynamics with particle interactions.

**Physics features**:
- Many-body simulation (30+ bodies)
- Particle-particle collisions
- Emergent behavior
- Energy dissipation

**Visual features**:
- Particle rendering
- Collective motion visualization
- Density patterns

### Scenario 4: Velocity Field
Field-based visualization of body velocities.

**Physics features**:
- Orbital motion
- Conservation of momentum
- Velocity vectors

**Visual features**:
- Velocity field interpolation
- Plasma color mapping
- Spatial velocity distribution

## Usage

```bash
# Run the demo
python examples/interactive_physics_sandbox/demo.py
```

## Output

The demo generates frame sequences for each scenario:

**Scenario 1 (Falling Objects)**:
- `scenario1_frame00.png` - Initial state
- `scenario1_frame01.png` - Mid-fall
- `scenario1_frame02.png` - After first bounce
- `scenario1_frame03.png` - Final state

**Scenario 2 (Box Stack)**:
- `scenario2_frame00.png` - Initial drop
- `scenario2_frame01.png` - Settling
- `scenario2_frame02.png` - Final stable configuration

**Scenario 3 (Particle Rain)**:
- `scenario3_frame00.png` through `scenario3_frame04.png` - Particle evolution

**Scenario 4 (Velocity Field)**:
- `scenario4_field_step000.png` through `scenario4_field_step090.png` - Field evolution

## Technical Details

### Physics Engine
- **Integration**: Semi-implicit Euler (stable for rigid bodies)
- **Collision Detection**: Broad-phase + narrow-phase
- **Collision Response**: Impulse-based with friction
- **Constraint Solver**: Iterative contact resolution
- **Timestep**: Fixed at 1/60s (60 FPS)

### Rendering
- **Resolution**: 800×800 pixels
- **World Bounds**: -10 to +10 meters in X and Y
- **Color Coding**: Velocity magnitude (0-10 m/s)
- **Shapes**: Circles and oriented rectangles

### Performance
- **Simulation Speed**: Real-time capable
- **Body Capacity**: Tested with 30+ bodies
- **Memory**: ~50MB per scenario
- **Runtime**: 5-15 seconds per scenario

## Physics Concepts Demonstrated

1. **Newton's Laws**:
   - Force = mass × acceleration
   - Action-reaction pairs in collisions
   - Conservation of momentum

2. **Collision Physics**:
   - Elastic vs inelastic collisions
   - Restitution coefficients
   - Friction forces
   - Contact resolution

3. **Rigid Body Dynamics**:
   - Linear and angular motion
   - Moment of inertia
   - Torque generation
   - Rotation dynamics

4. **Many-Body Systems**:
   - Emergent behavior
   - Energy dissipation
   - Pattern formation

## Extension Ideas

- Add user interaction (mouse clicks to add bodies)
- Implement joints and constraints (springs, hinges)
- Add force fields (wind, magnetism)
- Create compound shapes (ragdolls, vehicles)
- Add soft body physics
- Implement fluid-rigid body coupling
- Add particle effects (trails, explosions)
- Create game scenarios (angry birds, pinball)

## Cross-Domain Patterns

### Pattern 1: Physics → Rendering
```python
# Simulate physics
world = step_world(world)

# Render current state
img = render_world(world)

# Save visualization
image.save(img, "frame.png")
```

### Pattern 2: Physics → Field → Visual
```python
# Extract velocity field from bodies
vel_field = create_velocity_field_visualization(world)

# Apply color mapping
colored = palette.map(palette.plasma(), vel_field)

# Render
visual_output = visual.Visual(colored)
```

### Pattern 3: Multi-Frame Animation
```python
frames = []
for step in range(n_steps):
    world = step_world(world)
    if step % capture_interval == 0:
        frames.append(render_world(world))

# Create animation
visual.video(frames, "animation.mp4")
```

## Related Examples

- `examples/rigidbody_physics/` - More rigid body examples
- `examples/showcase/02_physics_visualizer.py` - Physics visualizations
- `examples/integrators/` - Time integration methods
- `examples/field/` - Field operations

## Educational Value

This example is excellent for:
- **Physics Education**: Visualizing classical mechanics
- **Game Development**: Understanding physics engines
- **Scientific Computing**: Numerical simulation techniques
- **Computer Graphics**: Real-time rendering

## References

- Rigid body dynamics: Baraff, D. (1997). "An Introduction to Physically Based Modeling"
- Collision detection: Ericson, C. (2004). "Real-Time Collision Detection"
- Game physics: Millington, I. (2007). "Game Physics Engine Development"
