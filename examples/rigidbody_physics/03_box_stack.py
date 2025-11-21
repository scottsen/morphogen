"""Example 3: Box Stack (Coming Soon)

Demonstrates:
- Box-box collisions
- Stable stacking
- Rotational dynamics
- Angular momentum conservation

Note: Currently only circle-circle collisions are implemented.
Box collision detection will be added in future updates.
"""

import sys
import numpy as np

sys.path.insert(0, '../..')

from morphogen.stdlib.rigidbody import (
    PhysicsWorld2D,
    create_circle_body,
    step_world,
)


def main():
    print("=" * 60)
    print("EXAMPLE 3: Stack Simulation (Circles)")
    print("=" * 60)
    print()

    # Create physics world
    world = PhysicsWorld2D(
        gravity=np.array([0.0, -9.81]),
        damping=0.98,
        dt=1.0 / 120.0,  # Higher timestep for stability
        iterations=20  # More iterations for stability
    )

    # Ground
    ground = create_circle_body(
        position=np.array([0.0, -10.0]),
        radius=10.0,
        mass=0.0,
        restitution=0.1,  # Low bounce
        friction=0.9  # High friction for stability
    )
    world.add_body(ground)

    # Create a tower of circles
    num_layers = 5
    layer_spacing = 1.1  # Slightly more than 2 * radius

    for i in range(num_layers):
        y = i * layer_spacing
        circle = create_circle_body(
            position=np.array([0.0, y]),
            radius=0.5,
            mass=1.0,
            restitution=0.1,
            friction=0.9
        )
        world.add_body(circle)
        print(f"Layer {i + 1}: y = {y:.2f}m")

    print()
    print(f"Simulating 10 seconds...")
    print()

    # Simulate
    duration = 10.0
    steps = int(duration / world.dt)
    sample_interval = steps // 10

    for step in range(steps):
        world = step_world(world)

        if step % sample_interval == 0:
            t = world.time
            print(f"t = {t:.1f}s:")
            for i, body in enumerate(world.bodies[1:], 1):  # Skip ground
                print(f"  Layer {i}: y={body.position[1]:6.2f}m, vy={body.velocity[1]:6.2f}m/s")
            print()

    print("=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print()

    # Check stability
    print("Final stability check:")
    stable = True
    for i, body in enumerate(world.bodies[1:], 1):
        speed = np.linalg.norm(body.velocity)
        is_stable = speed < 0.1
        status = "STABLE" if is_stable else "MOVING"
        print(f"  Layer {i}: speed={speed:.3f} m/s [{status}]")
        if not is_stable:
            stable = False

    print()
    if stable:
        print("✓ Tower is stable!")
    else:
        print("⚠ Tower still settling...")


if __name__ == "__main__":
    main()
