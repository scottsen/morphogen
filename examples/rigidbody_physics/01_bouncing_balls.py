"""Example 1: Bouncing Balls

Demonstrates basic rigid body physics with gravity, collisions, and restitution.
Shows multiple balls bouncing on a static ground.
"""

import sys
import numpy as np

# Add parent directory to path
sys.path.insert(0, '../..')

from morphogen.stdlib.rigidbody import (
    PhysicsWorld2D,
    create_circle_body,
    step_world,
)


def main():
    print("=" * 60)
    print("EXAMPLE 1: Bouncing Balls")
    print("=" * 60)
    print()

    # Create physics world with gravity
    world = PhysicsWorld2D(
        gravity=np.array([0.0, -9.81]),  # Earth gravity
        damping=0.995,  # Slight air resistance
        dt=1.0 / 60.0  # 60 FPS
    )

    # Create ground (static body with large radius = flat surface)
    ground = create_circle_body(
        position=np.array([0.0, -10.0]),
        radius=10.0,
        mass=0.0,  # Static (infinite mass)
        restitution=0.8,
        friction=0.2
    )
    world.add_body(ground)

    # Create several bouncing balls with different properties
    balls = [
        # (position, radius, mass, restitution, friction, name)
        (np.array([0.0, 10.0]), 0.5, 1.0, 0.9, 0.1, "Bouncy Ball"),
        (np.array([2.0, 8.0]), 0.3, 0.5, 0.7, 0.3, "Medium Ball"),
        (np.array([-2.0, 12.0]), 0.7, 2.0, 0.5, 0.5, "Heavy Ball"),
        (np.array([1.0, 15.0]), 0.4, 0.8, 1.0, 0.0, "Super Bouncy"),
    ]

    ball_ids = []
    for pos, radius, mass, restitution, friction, name in balls:
        ball = create_circle_body(
            position=pos,
            radius=radius,
            mass=mass,
            restitution=restitution,
            friction=friction
        )
        ball_id = world.add_body(ball)
        ball_ids.append((ball_id, name))
        print(f"Added {name}:")
        print(f"  Position: {pos}")
        print(f"  Radius: {radius}m, Mass: {mass}kg")
        print(f"  Restitution: {restitution}, Friction: {friction}")
        print()

    # Simulate for 5 seconds
    duration = 5.0
    steps = int(duration / world.dt)

    print(f"Simulating {duration} seconds ({steps} steps)...")
    print()

    # Track sample positions
    sample_times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    sample_step_indices = [int(t / world.dt) for t in sample_times]

    for step in range(steps):
        world = step_world(world)

        # Print status at sample times
        if step in sample_step_indices:
            t = world.time
            print(f"Time = {t:.2f}s:")
            for ball_id, name in ball_ids:
                body = world.get_body(ball_id)
                print(f"  {name}: y={body.position[1]:6.2f}m, vy={body.velocity[1]:6.2f}m/s")
            print()

    print("=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print()

    # Final statistics
    print("Final state:")
    for ball_id, name in ball_ids:
        body = world.get_body(ball_id)
        speed = np.linalg.norm(body.velocity)
        print(f"  {name}:")
        print(f"    Position: [{body.position[0]:.2f}, {body.position[1]:.2f}]")
        print(f"    Velocity: [{body.velocity[0]:.2f}, {body.velocity[1]:.2f}]")
        print(f"    Speed: {speed:.2f} m/s")


if __name__ == "__main__":
    main()
