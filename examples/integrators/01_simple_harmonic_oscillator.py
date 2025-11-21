"""Example: Simple Harmonic Oscillator using different integrators.

Demonstrates the accuracy and energy conservation properties of different
integration methods on the classic SHO: d²x/dt² = -k*x
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/morphogen')

from morphogen.stdlib.integrators import euler, rk2, rk4, verlet, symplectic


def sho_derivative(t, state):
    """SHO as first-order ODE system: [x, v]' = [v, -k*x]"""
    x, v = state
    k = 1.0  # Spring constant
    return np.array([v, -k * x])


def sho_acceleration(t, pos, vel):
    """SHO acceleration for symplectic integrators"""
    k = 1.0
    return -k * pos


def sho_force(x):
    """SHO force for symplectic split-operator methods"""
    k = 1.0
    return -k * x


def energy(state, k=1.0, m=1.0):
    """Total energy of SHO"""
    x, v = state
    return 0.5 * k * x**2 + 0.5 * m * v**2


def main():
    # Initial conditions
    x0, v0 = 1.0, 0.0
    dt = 0.01
    n_steps = int(10 * 2 * np.pi / dt)  # 10 periods

    # Method comparison
    methods = [
        ("Euler", lambda t, s: euler(t, s, sho_derivative, dt)),
        ("RK2", lambda t, s: rk2(t, s, sho_derivative, dt)),
        ("RK4", lambda t, s: rk4(t, s, sho_derivative, dt)),
    ]

    print("=" * 80)
    print("SIMPLE HARMONIC OSCILLATOR - Method Comparison")
    print("=" * 80)
    print(f"Initial state: x={x0}, v={v0}")
    print(f"Timestep: dt={dt}")
    print(f"Simulating {n_steps} steps ({n_steps * dt / (2*np.pi):.1f} periods)")
    print()

    for name, method in methods:
        state = np.array([x0, v0])
        initial_energy = energy(state)
        t = 0.0

        for _ in range(n_steps):
            state = method(t, state)
            t += dt

        final_energy = energy(state)
        energy_drift = abs(final_energy - initial_energy) / initial_energy * 100

        print(f"{name:10s}: x={state[0]:+.6f}, v={state[1]:+.6f}, "
              f"energy drift={energy_drift:.4f}%")

    print()
    print("-" * 80)
    print("SYMPLECTIC INTEGRATORS (Energy Conserving)")
    print("-" * 80)

    # Verlet integrator
    pos = np.array([x0])
    vel = np.array([v0])
    initial_energy = energy(np.array([pos[0], vel[0]]))

    for _ in range(n_steps):
        pos, vel = verlet(0, pos, vel, sho_acceleration, dt)

    final_energy = energy(np.array([pos[0], vel[0]]))
    energy_drift = abs(final_energy - initial_energy) / initial_energy * 100

    print(f"Verlet    : x={pos[0]:+.6f}, v={vel[0]:+.6f}, "
          f"energy drift={energy_drift:.4f}%")

    # Symplectic 2nd order
    pos = np.array([x0])
    vel = np.array([v0])
    initial_energy = energy(np.array([pos[0], vel[0]]))

    for _ in range(n_steps):
        pos, vel = symplectic(0, pos, vel, sho_force, dt, order=2)

    final_energy = energy(np.array([pos[0], vel[0]]))
    energy_drift = abs(final_energy - initial_energy) / initial_energy * 100

    print(f"Symplectic: x={pos[0]:+.6f}, v={vel[0]:+.6f}, "
          f"energy drift={energy_drift:.4f}%")

    print()
    print("=" * 80)
    print("KEY OBSERVATIONS:")
    print("  - Euler: Largest error, poor energy conservation")
    print("  - RK2/RK4: Better accuracy, but energy still drifts")
    print("  - Verlet/Symplectic: Excellent energy conservation (< 0.1% drift)")
    print("  - For physics simulations, use symplectic integrators!")
    print("=" * 80)


if __name__ == "__main__":
    main()
