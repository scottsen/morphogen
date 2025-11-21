"""Example: N-body gravitational simulation using Verlet integrator.

Demonstrates symplectic integration for Hamiltonian systems (physics).
Simulates a simple 3-body gravitational system with energy conservation.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/morphogen')

from morphogen.stdlib.integrators import verlet, rk4


def gravitational_acceleration(t, positions, velocities, masses, G=1.0, softening=0.1):
    """Compute gravitational acceleration for N-body system.

    Args:
        t: Time (unused, for API compatibility)
        positions: Array of shape (N, 2) with particle positions
        velocities: Array of shape (N, 2) with particle velocities (unused)
        masses: Array of shape (N,) with particle masses
        G: Gravitational constant
        softening: Softening length to prevent singularities

    Returns:
        Acceleration array of shape (N, 2)
    """
    n = len(positions)
    accel = np.zeros_like(positions)

    for i in range(n):
        for j in range(i + 1, n):
            # Vector from i to j
            r_vec = positions[j] - positions[i]
            r_dist = np.linalg.norm(r_vec)

            # Softened gravitational force: F = G*m1*m2 / (r^2 + eps^2)^(3/2)
            force_mag = G * masses[i] * masses[j] / (r_dist**2 + softening**2)**1.5
            force_vec = force_mag * r_vec

            # Newton's third law
            accel[i] += force_vec / masses[i]
            accel[j] -= force_vec / masses[j]

    return accel


def total_energy(positions, velocities, masses, G=1.0):
    """Compute total energy (kinetic + potential) of N-body system"""
    n = len(positions)

    # Kinetic energy: T = 0.5 * sum(m * v^2)
    kinetic = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))

    # Potential energy: V = -G * sum(m_i * m_j / r_ij)
    potential = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[j] - positions[i])
            potential -= G * masses[i] * masses[j] / r

    return kinetic + potential


def main():
    print("=" * 80)
    print("N-BODY GRAVITATIONAL SIMULATION - Verlet vs RK4")
    print("=" * 80)
    print()

    # Setup: 3-body system (sun + 2 planets in circular orbits)
    G = 1.0

    # Initial conditions (sun at origin, 2 planets in orbit)
    masses = np.array([100.0, 1.0, 1.0])  # Sun, Planet 1, Planet 2
    positions_verlet = np.array([
        [0.0, 0.0],      # Sun at origin
        [1.0, 0.0],      # Planet 1 at distance 1
        [0.0, -1.5]      # Planet 2 at distance 1.5
    ])
    positions_rk4 = positions_verlet.copy()

    # Circular orbit velocities: v = sqrt(G*M/r)
    v1 = np.sqrt(G * masses[0] / 1.0)
    v2 = np.sqrt(G * masses[0] / 1.5)

    velocities_verlet = np.array([
        [0.0, 0.0],      # Sun stationary
        [0.0, v1],       # Planet 1 circular velocity
        [v2, 0.0]        # Planet 2 circular velocity
    ])
    velocities_rk4 = velocities_verlet.copy()

    # Simulation parameters
    dt = 0.01
    t_max = 20.0
    n_steps = int(t_max / dt)

    print(f"Initial configuration:")
    print(f"  Masses: {masses}")
    print(f"  Timestep: dt={dt}")
    print(f"  Duration: {t_max} time units ({n_steps} steps)")
    print()

    # Initial energy
    initial_energy = total_energy(positions_verlet, velocities_verlet, masses, G)
    print(f"Initial total energy: {initial_energy:.6f}")
    print()

    # Simulate with Verlet
    print("Simulating with Verlet integrator...")

    def accel_func(t, pos, vel):
        return gravitational_acceleration(t, pos, vel, masses, G)

    for step in range(n_steps):
        positions_verlet, velocities_verlet = verlet(
            step * dt,
            positions_verlet,
            velocities_verlet,
            accel_func,
            dt
        )

    final_energy_verlet = total_energy(positions_verlet, velocities_verlet, masses, G)
    energy_drift_verlet = abs(final_energy_verlet - initial_energy) / abs(initial_energy) * 100

    print(f"  Final energy: {final_energy_verlet:.6f}")
    print(f"  Energy drift: {energy_drift_verlet:.4f}%")
    print()

    # Simulate with RK4 for comparison
    print("Simulating with RK4 integrator (for comparison)...")

    def combined_derivative(t, state):
        """Combined [pos, vel] derivative for RK4"""
        n = len(masses)
        pos = state[:n].reshape(n, 2)
        vel = state[n:].reshape(n, 2)
        accel = gravitational_acceleration(t, pos, vel, masses, G)
        return np.concatenate([vel.flatten(), accel.flatten()])

    # Flatten state for RK4
    state_rk4 = np.concatenate([positions_rk4.flatten(), velocities_rk4.flatten()])

    for step in range(n_steps):
        state_rk4 = rk4(step * dt, state_rk4, combined_derivative, dt)

    # Unflatten state
    n = len(masses)
    positions_rk4 = state_rk4[:n*2].reshape(n, 2)
    velocities_rk4 = state_rk4[n*2:].reshape(n, 2)

    final_energy_rk4 = total_energy(positions_rk4, velocities_rk4, masses, G)
    energy_drift_rk4 = abs(final_energy_rk4 - initial_energy) / abs(initial_energy) * 100

    print(f"  Final energy: {final_energy_rk4:.6f}")
    print(f"  Energy drift: {energy_drift_rk4:.4f}%")
    print()

    # Results
    print("=" * 80)
    print("FINAL POSITIONS")
    print("=" * 80)
    print()
    print("Verlet integrator:")
    for i, (pos, vel) in enumerate(zip(positions_verlet, velocities_verlet)):
        print(f"  Body {i}: pos=({pos[0]:+.4f}, {pos[1]:+.4f}), "
              f"vel=({vel[0]:+.4f}, {vel[1]:+.4f})")
    print()
    print("RK4 integrator:")
    for i, (pos, vel) in enumerate(zip(positions_rk4, velocities_rk4)):
        print(f"  Body {i}: pos=({pos[0]:+.4f}, {pos[1]:+.4f}), "
              f"vel=({vel[0]:+.4f}, {vel[1]:+.4f})")
    print()

    print("=" * 80)
    print("KEY OBSERVATIONS:")
    print(f"  - Verlet energy drift: {energy_drift_verlet:.4f}%")
    print(f"  - RK4 energy drift:    {energy_drift_rk4:.4f}%")
    print("  - Verlet is SYMPLECTIC → conserves energy exactly (bounded drift)")
    print("  - RK4 is NOT symplectic → energy drifts monotonically")
    print("  - For long-term physics simulations, ALWAYS use symplectic integrators!")
    print("=" * 80)


if __name__ == "__main__":
    main()
