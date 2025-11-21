"""Example 2: Collision Demo

Demonstrates collisions between moving bodies with different properties.
Shows elastic and inelastic collisions, friction effects.
"""

import sys
import numpy as np

sys.path.insert(0, '../..')

from morphogen.stdlib.rigidbody import (
    PhysicsWorld2D,
    create_circle_body,
    create_box_body,
    step_world,
)


def demo_elastic_collision():
    """Demo: Two balls with elastic collision (restitution = 1.0)."""
    print("\n" + "=" * 60)
    print("DEMO 1: Elastic Collision (e = 1.0)")
    print("=" * 60)

    world = PhysicsWorld2D(
        gravity=np.array([0.0, 0.0]),  # No gravity
        damping=1.0,  # No damping
        dt=0.01
    )

    # Two identical balls moving toward each other
    ball_a = create_circle_body(
        position=np.array([-5.0, 0.0]),
        radius=0.5,
        mass=1.0,
        restitution=1.0,
        friction=0.0
    )
    ball_a.velocity = np.array([3.0, 0.0])

    ball_b = create_circle_body(
        position=np.array([5.0, 0.0]),
        radius=0.5,
        mass=1.0,
        restitution=1.0,
        friction=0.0
    )
    ball_b.velocity = np.array([-3.0, 0.0])

    world.add_body(ball_a)
    world.add_body(ball_b)

    print("Initial state:")
    print(f"  Ball A: pos=({ball_a.position[0]:.2f}, {ball_a.position[1]:.2f}), "
          f"vel=({ball_a.velocity[0]:.2f}, {ball_a.velocity[1]:.2f})")
    print(f"  Ball B: pos=({ball_b.position[0]:.2f}, {ball_b.position[1]:.2f}), "
          f"vel=({ball_b.velocity[0]:.2f}, {ball_b.velocity[1]:.2f})")

    # Simulate until collision and aftermath
    for _ in range(400):
        world = step_world(world)

    ball_a = world.bodies[0]
    ball_b = world.bodies[1]

    print("\nFinal state (after collision):")
    print(f"  Ball A: pos=({ball_a.position[0]:.2f}, {ball_a.position[1]:.2f}), "
          f"vel=({ball_a.velocity[0]:.2f}, {ball_a.velocity[1]:.2f})")
    print(f"  Ball B: pos=({ball_b.position[0]:.2f}, {ball_b.position[1]:.2f}), "
          f"vel=({ball_b.velocity[0]:.2f}, {ball_b.velocity[1]:.2f})")

    print("\nExpected: Velocities should swap (elastic collision, equal masses)")


def demo_inelastic_collision():
    """Demo: Inelastic collision (restitution = 0.0)."""
    print("\n" + "=" * 60)
    print("DEMO 2: Inelastic Collision (e = 0.0)")
    print("=" * 60)

    world = PhysicsWorld2D(
        gravity=np.array([0.0, 0.0]),
        damping=1.0,
        dt=0.01
    )

    ball_a = create_circle_body(
        position=np.array([-5.0, 0.0]),
        radius=0.5,
        mass=1.0,
        restitution=0.0,  # Inelastic
        friction=0.0
    )
    ball_a.velocity = np.array([5.0, 0.0])

    ball_b = create_circle_body(
        position=np.array([5.0, 0.0]),
        radius=0.5,
        mass=1.0,
        restitution=0.0,
        friction=0.0
    )
    ball_b.velocity = np.array([0.0, 0.0])  # Stationary

    world.add_body(ball_a)
    world.add_body(ball_b)

    print("Initial state:")
    print(f"  Ball A: vel=({ball_a.velocity[0]:.2f}, {ball_a.velocity[1]:.2f})")
    print(f"  Ball B: vel=({ball_b.velocity[0]:.2f}, {ball_b.velocity[1]:.2f})")

    for _ in range(400):
        world = step_world(world)

    ball_a = world.bodies[0]
    ball_b = world.bodies[1]

    print("\nFinal state:")
    print(f"  Ball A: vel=({ball_a.velocity[0]:.2f}, {ball_a.velocity[1]:.2f})")
    print(f"  Ball B: vel=({ball_b.velocity[0]:.2f}, {ball_b.velocity[1]:.2f})")

    print("\nExpected: Both balls moving together at ~2.5 m/s (momentum conserved)")


def demo_mass_difference():
    """Demo: Collision between heavy and light objects."""
    print("\n" + "=" * 60)
    print("DEMO 3: Heavy vs Light Collision")
    print("=" * 60)

    world = PhysicsWorld2D(
        gravity=np.array([0.0, 0.0]),
        damping=1.0,
        dt=0.01
    )

    # Heavy ball (bowling ball)
    heavy = create_circle_body(
        position=np.array([-5.0, 0.0]),
        radius=0.3,
        mass=5.0,  # 5 kg
        restitution=0.8,
        friction=0.0
    )
    heavy.velocity = np.array([2.0, 0.0])

    # Light ball (ping pong ball)
    light = create_circle_body(
        position=np.array([5.0, 0.0]),
        radius=0.2,
        mass=0.1,  # 0.1 kg
        restitution=0.8,
        friction=0.0
    )
    light.velocity = np.array([-1.0, 0.0])

    world.add_body(heavy)
    world.add_body(light)

    print("Initial state:")
    print(f"  Heavy ball (5kg): vel={heavy.velocity[0]:.2f} m/s")
    print(f"  Light ball (0.1kg): vel={light.velocity[0]:.2f} m/s")

    for _ in range(600):
        world = step_world(world)

    heavy = world.bodies[0]
    light = world.bodies[1]

    print("\nFinal state:")
    print(f"  Heavy ball: vel={heavy.velocity[0]:.2f} m/s")
    print(f"  Light ball: vel={light.velocity[0]:.2f} m/s")

    print("\nExpected: Heavy ball barely affected, light ball bounces back fast")


def demo_friction():
    """Demo: Collision with friction (tangential effects)."""
    print("\n" + "=" * 60)
    print("DEMO 4: Friction Effects")
    print("=" * 60)

    world = PhysicsWorld2D(
        gravity=np.array([0.0, -9.81]),
        damping=0.99,
        dt=0.01
    )

    # Ground
    ground = create_circle_body(
        position=np.array([0.0, -5.0]),
        radius=5.0,
        mass=0.0,
        restitution=0.3,
        friction=0.8  # High friction
    )

    # Ball with horizontal and vertical velocity
    ball = create_circle_body(
        position=np.array([0.0, 2.0]),
        radius=0.5,
        mass=1.0,
        restitution=0.5,
        friction=0.8
    )
    ball.velocity = np.array([5.0, 0.0])  # Moving sideways

    world.add_body(ground)
    world.add_body(ball)

    print("Initial state:")
    print(f"  Ball: vel=({ball.velocity[0]:.2f}, {ball.velocity[1]:.2f}) m/s")

    # Simulate bounces
    for _ in range(500):
        world = step_world(world)

    ball = world.bodies[1]

    print("\nFinal state:")
    print(f"  Ball: vel=({ball.velocity[0]:.2f}, {ball.velocity[1]:.2f}) m/s")

    print("\nExpected: Horizontal velocity reduced by friction during bounces")


def main():
    print("=" * 60)
    print("RIGID BODY PHYSICS: Collision Demonstrations")
    print("=" * 60)

    demo_elastic_collision()
    demo_inelastic_collision()
    demo_mass_difference()
    demo_friction()

    print("\n" + "=" * 60)
    print("ALL DEMOS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
