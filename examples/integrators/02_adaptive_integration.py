"""Example: Adaptive timestep integration with error control.

Demonstrates adaptive Dormand-Prince integrator on stiff and non-stiff systems.
Shows how the integrator automatically adjusts timestep to maintain accuracy.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/morphogen')

from morphogen.stdlib.integrators import adaptive_integrate, rk4


def van_der_pol(t, state):
    """Van der Pol oscillator (stiff for large mu).

    This is a classic stiff ODE that exhibits limit cycle behavior.
    For large mu, the system has both fast and slow dynamics.
    """
    x, v = state
    mu = 10.0  # Stiffness parameter (larger = stiffer)
    return np.array([v, mu * (1 - x**2) * v - x])


def lorenz(t, state):
    """Lorenz attractor (chaotic system)"""
    x, y, z = state
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ])


def exponential_decay(t, state):
    """Simple exponential decay: dy/dt = -y"""
    return -state


def main():
    print("=" * 80)
    print("ADAPTIVE INTEGRATION - Automatic Timestep Control")
    print("=" * 80)
    print()

    # Example 1: Exponential decay (simple, non-stiff)
    print("1. EXPONENTIAL DECAY (Non-Stiff)")
    print("-" * 80)

    state0 = np.array([1.0])
    times, states = adaptive_integrate(
        exponential_decay,
        state0,
        (0.0, 5.0),
        dt_initial=0.1,
        tol=1e-6
    )

    print(f"Time span: 0.0 → 5.0")
    print(f"Number of steps: {len(times)}")
    print(f"Average timestep: {5.0 / (len(times) - 1):.4f}")
    print(f"Final value: {states[-1][0]:.8f}")
    print(f"Expected value: {np.exp(-5.0):.8f}")
    print(f"Error: {abs(states[-1][0] - np.exp(-5.0)):.2e}")
    print()

    # Example 2: Van der Pol oscillator (stiff)
    print("2. VAN DER POL OSCILLATOR (Stiff, mu=10)")
    print("-" * 80)

    state0 = np.array([2.0, 0.0])
    times_adaptive, states_adaptive = adaptive_integrate(
        van_der_pol,
        state0,
        (0.0, 20.0),
        dt_initial=0.1,
        tol=1e-4
    )

    print(f"Time span: 0.0 → 20.0")
    print(f"Number of adaptive steps: {len(times_adaptive)}")
    print(f"Average timestep: {20.0 / (len(times_adaptive) - 1):.4f}")

    # Compare with fixed timestep RK4
    state_fixed = state0.copy()
    dt_fixed = 0.01
    n_steps_fixed = int(20.0 / dt_fixed)

    for i in range(n_steps_fixed):
        state_fixed = rk4(i * dt_fixed, state_fixed, van_der_pol, dt_fixed)

    print(f"Fixed RK4 steps (dt=0.01): {n_steps_fixed}")
    print(f"Final state (adaptive): x={states_adaptive[-1][0]:.6f}, "
          f"v={states_adaptive[-1][1]:.6f}")
    print(f"Final state (fixed):    x={state_fixed[0]:.6f}, "
          f"v={state_fixed[1]:.6f}")
    print()

    # Example 3: Lorenz attractor (chaotic)
    print("3. LORENZ ATTRACTOR (Chaotic)")
    print("-" * 80)

    state0 = np.array([1.0, 1.0, 1.0])
    times, states = adaptive_integrate(
        lorenz,
        state0,
        (0.0, 20.0),
        dt_initial=0.01,
        tol=1e-6
    )

    print(f"Time span: 0.0 → 20.0")
    print(f"Number of steps: {len(times)}")
    print(f"Average timestep: {20.0 / (len(times) - 1):.4f}")
    print(f"Final state: x={states[-1][0]:.4f}, y={states[-1][1]:.4f}, "
          f"z={states[-1][2]:.4f}")

    # Analyze timestep variation
    timesteps = np.diff(times)
    print(f"Min timestep: {np.min(timesteps):.6f}")
    print(f"Max timestep: {np.max(timesteps):.6f}")
    print(f"Timestep range: {np.max(timesteps) / np.min(timesteps):.1f}x variation")
    print()

    print("=" * 80)
    print("KEY OBSERVATIONS:")
    print("  - Adaptive integration uses fewer steps for smooth regions")
    print("  - Timestep automatically decreases in regions of rapid change")
    print("  - Error tolerance controls accuracy vs. computational cost")
    print("  - Stiff systems benefit greatly from adaptive stepping")
    print("=" * 80)


if __name__ == "__main__":
    main()
