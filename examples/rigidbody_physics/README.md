# Rigid Body Physics Examples

This directory contains examples demonstrating Kairo's 2D rigid body physics capabilities.

## Examples

### 01_bouncing_balls.py
Demonstrates basic rigid body physics with gravity, collisions, and restitution.

**Concepts:**
- Static vs dynamic bodies
- Gravity and integration
- Elastic and inelastic collisions (restitution coefficient)
- Friction effects
- Multiple body simulation

**Output:** Simulation of 4 balls with different properties bouncing on a ground.

**Run:**
```bash
python 01_bouncing_balls.py
```

### 02_collision_demo.py
Comprehensive collision demonstrations with different scenarios.

**Concepts:**
- Elastic collision (e=1.0) - perfect bounce
- Inelastic collision (e=0.0) - objects stick together
- Mass ratio effects (heavy vs light)
- Friction during collision (tangential impulse)
- Momentum and energy conservation

**Demonstrations:**
1. Two equal-mass balls in elastic collision (velocities swap)
2. Inelastic collision (momentum conserved, energy lost)
3. Heavy ball vs light ball (light ball bounces back fast)
4. Friction reducing horizontal velocity

**Run:**
```bash
python 02_collision_demo.py
```

### 03_box_stack.py
Stack simulation demonstrating stability and multi-body interactions.

**Concepts:**
- Multi-body constraint solving
- Stable stacking (high friction, low restitution)
- Iterative collision resolution
- Settling behavior

**Note:** Currently uses circles. Box-box collisions coming soon.

**Run:**
```bash
python 03_box_stack.py
```

## Rigid Body Physics Features

### Implemented ✅
- **Shapes**: Circle, Box (box rendering only, collision WIP)
- **Collision Detection**: Circle-circle
- **Collision Response**: Impulse-based with restitution and friction
- **Integration**: Semi-implicit Euler
- **Forces**: Apply force/impulse at arbitrary points
- **Static Bodies**: Infinite mass/inertia
- **World Simulation**: Fixed timestep, iterative solver
- **Utilities**: Raycast, vertex extraction

### Planned 🚧
- **Collision Detection**: Box-box, circle-box, polygon-polygon
- **Constraints**: Distance joint, hinge joint, spring joint
- **Advanced Integration**: Verlet, RK4 (via integrators module)
- **Spatial Hashing**: O(n) broad-phase collision detection
- **Convex Hulls**: General polygon support
- **Continuous Collision Detection**: Prevent tunneling

## Physics Parameters

### Restitution (Bounciness)
- `0.0` = Perfectly inelastic (objects don't bounce)
- `0.5` = Moderate bounce (realistic for most objects)
- `1.0` = Perfectly elastic (perfect bounce, energy conserved)

### Friction
- `0.0` = Frictionless (ice-like)
- `0.3` = Low friction (smooth surfaces)
- `0.5` = Moderate friction (wood on wood)
- `0.8+` = High friction (rubber on concrete)

### Damping
- `0.95` = High damping (underwater-like)
- `0.99` = Low damping (realistic air resistance)
- `1.0` = No damping (space-like)

### Solver Iterations
- `1-5` = Fast but unstable (simple scenes)
- `10` = Default (good balance)
- `20+` = High stability (stacking, complex constraints)

## Performance Notes

- Current implementation: O(n²) collision detection
- Recommended: <100 bodies for real-time (60 FPS)
- Future: Spatial hashing will enable 1000+ bodies

## Integration with Other Kairo Domains

Rigid body physics can be integrated with:

- **Fields**: Bodies can sample field values (flow fields, temperature)
- **Agents**: Hybrid particle/rigid-body systems
- **Visual**: Render body positions and rotations
- **Integrators**: Use RK4/Verlet for higher accuracy

## References

- **Impulse-based collision response**: Erin Catto's GDC talks
- **Physics simulation**: "Game Physics Engine Development" by Ian Millington
- **Rigid body dynamics**: "Physics for Game Developers" by David M. Bourg

## Next Steps

1. Run examples to understand basic concepts
2. Experiment with different restitution/friction values
3. Create custom scenes with multiple bodies
4. Integrate with visual domain for rendering (coming soon)
