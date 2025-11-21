"""Phase Space Visualization Demo

Demonstrates the new visual.phase_space() function for analyzing dynamical systems.
Shows position-velocity relationships for various particle systems.
"""

import numpy as np
from morphogen.stdlib import agents as agent_ops
from morphogen.stdlib import visual


def simulate_harmonic_oscillator(n_steps=1000, dt=0.01, omega=2.0, damping=0.1, n_agents=100):
    """Simulate damped harmonic oscillators with different initial conditions."""
    # Initialize agents with random positions and velocities
    positions = np.random.uniform(-2.0, 2.0, (n_agents, 1))
    velocities = np.random.uniform(-2.0, 2.0, (n_agents, 1))

    particles = agent_ops.create(n_agents, pos=positions)
    particles = agent_ops.set(particles, 'vel', velocities)

    # Simulate for a while to get interesting trajectories
    for step in range(n_steps):
        pos = particles.get('pos')
        vel = particles.get('vel')

        # Damped harmonic oscillator: F = -k*x - c*v
        acceleration = -omega**2 * pos - 2 * damping * omega * vel

        # Update velocity and position
        vel = vel + acceleration * dt
        pos = pos + vel * dt

        particles = agent_ops.set(particles, 'vel', vel)
        particles = agent_ops.set(particles, 'pos', pos)

    # Add energy for coloring
    kinetic = 0.5 * particles.get('vel')**2
    potential = 0.5 * omega**2 * particles.get('pos')**2
    energy = (kinetic + potential).flatten()
    particles = agent_ops.set(particles, 'energy', energy)

    return particles


def simulate_double_pendulum_endpoints(n_agents=500, time_range=10.0):
    """Generate phase space data from double pendulum simulations."""
    # Simplified double pendulum endpoint positions/velocities
    np.random.seed(42)

    # Sample different initial conditions
    theta1_init = np.random.uniform(-np.pi, np.pi, n_agents)
    theta2_init = np.random.uniform(-np.pi, np.pi, n_agents)

    # Simulate (simplified - just showing the concept)
    t = time_range
    g, L = 9.81, 1.0

    # Approximate dynamics (this is simplified for demonstration)
    positions = np.column_stack([
        L * (np.sin(theta1_init) + np.sin(theta2_init)),
        -L * (np.cos(theta1_init) + np.cos(theta2_init))
    ])

    velocities = np.column_stack([
        L * (theta1_init * np.cos(theta1_init) + theta2_init * np.cos(theta2_init)) / t,
        L * (theta1_init * np.sin(theta1_init) + theta2_init * np.sin(theta2_init)) / t
    ])

    particles = agent_ops.create(n_agents, pos=positions)
    particles = agent_ops.set(particles, 'vel', velocities)

    # Add chaos indicator (how far from origin)
    chaos = np.linalg.norm(positions, axis=1) + np.linalg.norm(velocities, axis=1)
    particles = agent_ops.set(particles, 'chaos', chaos)

    return particles


def simulate_planetary_system(n_orbits=8, points_per_orbit=100):
    """Simulate circular orbital dynamics."""
    n_agents = n_orbits * points_per_orbit
    positions = []
    velocities = []

    for orbit_idx in range(n_orbits):
        radius = (orbit_idx + 1) * 0.5
        v_orbital = np.sqrt(1.0 / radius)  # Simplified orbital velocity

        for point_idx in range(points_per_orbit):
            angle = 2 * np.pi * point_idx / points_per_orbit

            # Position on circular orbit
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)

            # Tangential velocity
            vx = -v_orbital * np.sin(angle)
            vy = v_orbital * np.cos(angle)

            positions.append([x, y])
            velocities.append([vx, vy])

    positions = np.array(positions)
    velocities = np.array(velocities)

    particles = agent_ops.create(n_agents, pos=positions)
    particles = agent_ops.set(particles, 'vel', velocities)

    # Add orbital radius for coloring
    radii = np.linalg.norm(positions, axis=1)
    particles = agent_ops.set(particles, 'radius', radii)

    return particles


def simulate_brownian_motion(n_agents=300, n_steps=100, dt=0.1, temperature=1.0):
    """Simulate Brownian motion with thermal fluctuations."""
    np.random.seed(123)

    # Start at origin
    positions = np.zeros((n_agents, 2))
    velocities = np.random.randn(n_agents, 2) * np.sqrt(temperature)

    particles = agent_ops.create(n_agents, pos=positions)
    particles = agent_ops.set(particles, 'vel', velocities)

    # Simulate random walk
    for step in range(n_steps):
        pos = particles.get('pos')
        vel = particles.get('vel')

        # Random force (thermal noise)
        random_force = np.random.randn(n_agents, 2) * np.sqrt(2 * temperature * dt)

        # Drag force
        drag = -0.5 * vel

        # Update
        vel = vel + (drag + random_force) * dt
        pos = pos + vel * dt

        particles = agent_ops.set(particles, 'pos', pos)
        particles = agent_ops.set(particles, 'vel', vel)

    # Add speed for coloring
    speed = np.linalg.norm(particles.get('vel'), axis=1)
    particles = agent_ops.set(particles, 'speed', speed)

    return particles


def main():
    print("Phase Space Visualization Demo")
    print("=" * 50)

    # Example 1: Damped harmonic oscillator
    print("\n1. Damped harmonic oscillator...")
    particles = simulate_harmonic_oscillator(
        n_steps=500,
        dt=0.01,
        omega=2.0,
        damping=0.1,
        n_agents=100
    )

    print("   Creating phase space diagram (position vs velocity)...")
    vis = visual.phase_space(
        particles,
        width=700,
        height=700,
        color_property='energy',
        palette='fire',
        point_size=3.0,
        alpha=0.7,
        background=(0.0, 0.0, 0.0)
    )

    metrics = {
        "System": "Damped Oscillator",
        "Agents": 100,
        "Damping": 0.1,
        "Frequency": 2.0
    }
    vis = visual.add_metrics(vis, metrics, position="top-left")

    visual.output(vis, "output_phase_space_oscillator.png")
    print("   Saved: output_phase_space_oscillator.png")

    # Example 2: Double pendulum (chaotic system)
    print("\n2. Double pendulum endpoints...")
    particles = simulate_double_pendulum_endpoints(n_agents=500, time_range=10.0)

    print("   Creating phase space with chaos indicator...")
    vis = visual.phase_space(
        particles,
        width=800,
        height=800,
        color_property='chaos',
        palette='viridis',
        point_size=2.0,
        alpha=0.6,
        show_trajectories=False,
        background=(0.05, 0.05, 0.1)
    )

    visual.output(vis, "output_phase_space_pendulum.png")
    print("   Saved: output_phase_space_pendulum.png")

    # Example 3: Planetary orbits
    print("\n3. Planetary orbital system...")
    particles = simulate_planetary_system(n_orbits=8, points_per_orbit=100)

    print("   Creating phase space diagram...")
    vis = visual.phase_space(
        particles,
        width=700,
        height=700,
        color_property='radius',
        palette='coolwarm',
        point_size=2.5,
        alpha=0.8,
        background=(0.0, 0.0, 0.05)
    )

    metrics = {
        "System": "Orbital Dynamics",
        "Orbits": 8,
        "Points": 800,
        "Type": "Circular"
    }
    vis = visual.add_metrics(vis, metrics, position="top-right")

    visual.output(vis, "output_phase_space_orbits.png")
    print("   Saved: output_phase_space_orbits.png")

    # Example 4: Brownian motion
    print("\n4. Brownian motion (thermal system)...")
    particles = simulate_brownian_motion(
        n_agents=300,
        n_steps=100,
        dt=0.1,
        temperature=1.0
    )

    print("   Creating phase space with speed coloring...")
    vis = visual.phase_space(
        particles,
        width=600,
        height=600,
        color_property='speed',
        palette='grayscale',
        point_size=3.0,
        alpha=0.7,
        background=(0.1, 0.1, 0.15)
    )

    visual.output(vis, "output_phase_space_brownian.png")
    print("   Saved: output_phase_space_brownian.png")

    # Example 5: Multiple oscillators with trajectories
    print("\n5. Harmonic oscillators with trajectories...")
    particles = simulate_harmonic_oscillator(
        n_steps=200,
        dt=0.02,
        omega=3.0,
        damping=0.05,
        n_agents=50
    )

    print("   Creating phase space with trajectory lines...")
    vis = visual.phase_space(
        particles,
        width=700,
        height=700,
        color_property='energy',
        palette='viridis',
        point_size=4.0,
        alpha=0.8,
        show_trajectories=True,
        background=(0.0, 0.0, 0.0)
    )

    visual.output(vis, "output_phase_space_trajectories.png")
    print("   Saved: output_phase_space_trajectories.png")

    print("\n" + "=" * 50)
    print("Demo complete! Phase space visualizations showcase:")
    print("  - Position-velocity analysis")
    print("  - Energy and chaos indicators")
    print("  - Oscillatory and chaotic systems")
    print("  - Orbital dynamics")
    print("  - Trajectory visualization")
    print("  - Property-based coloring")


if __name__ == "__main__":
    main()
