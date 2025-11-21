"""Simple verification script for integrators (no pytest required)"""

import numpy as np
import sys
sys.path.insert(0, '/home/user/morphogen')

from morphogen.stdlib.integrators import (
    euler, rk2, rk4,
    verlet, leapfrog, symplectic,
    dormand_prince_step, adaptive_integrate,
    integrate
)


def simple_harmonic_oscillator(t, state):
    """SHO: d²x/dt² = -k*x, state = [x, v]"""
    x, v = state[0], state[1]
    k = 1.0
    return np.array([v, -k * x])


def exponential_decay(t, state):
    """dy/dt = -lambda * y"""
    return -state


def analytic_exponential(t, y0=1.0):
    """Analytic solution for exponential decay"""
    return y0 * np.exp(-t)


def test_euler():
    """Test Euler method"""
    print("Testing Euler method...")
    state = np.array([1.0])
    for _ in range(100):
        state = euler(0, state, exponential_decay, 0.01)
    expected = analytic_exponential(1.0)
    error = abs(state[0] - expected)
    print(f"  Euler error: {error:.6f}")
    assert error < 0.05, "Euler test failed"
    print("  ✓ Euler test passed")


def test_rk2():
    """Test RK2 method"""
    print("Testing RK2 method...")
    state = np.array([1.0])
    for _ in range(100):
        state = rk2(0, state, exponential_decay, 0.01)
    expected = analytic_exponential(1.0)
    error = abs(state[0] - expected)
    print(f"  RK2 error: {error:.6f}")
    assert error < 1e-3, "RK2 test failed"
    print("  ✓ RK2 test passed")


def test_rk4():
    """Test RK4 method"""
    print("Testing RK4 method...")
    state = np.array([1.0])
    for _ in range(100):
        state = rk4(0, state, exponential_decay, 0.01)
    expected = analytic_exponential(1.0)
    error = abs(state[0] - expected)
    print(f"  RK4 error: {error:.6f}")
    assert error < 1e-6, "RK4 test failed"
    print("  ✓ RK4 test passed")


def test_verlet():
    """Test Verlet integrator"""
    print("Testing Verlet method...")
    pos = np.array([1.0])
    vel = np.array([0.0])

    def accel(t, p, v):
        return -p  # SHO

    # Integrate for one period
    n_steps = int(2 * np.pi / 0.01)
    for _ in range(n_steps):
        pos, vel = verlet(0, pos, vel, accel, 0.01)

    error = abs(pos[0] - 1.0) + abs(vel[0])
    print(f"  Verlet error: {error:.6f}")
    assert error < 1e-2, "Verlet test failed"
    print("  ✓ Verlet test passed")


def test_symplectic():
    """Test symplectic integrator"""
    print("Testing Symplectic method...")
    pos = np.array([1.0])
    vel = np.array([0.0])

    def force(x):
        return -x

    n_steps = int(2 * np.pi / 0.01)
    for _ in range(n_steps):
        pos, vel = symplectic(0, pos, vel, force, 0.01, order=2)

    error = abs(pos[0] - 1.0) + abs(vel[0])
    print(f"  Symplectic error: {error:.6f}")
    assert error < 1e-2, "Symplectic test failed"
    print("  ✓ Symplectic test passed")


def test_adaptive():
    """Test adaptive integrator"""
    print("Testing Adaptive integrator...")
    state0 = np.array([1.0])
    times, states = adaptive_integrate(
        exponential_decay,
        state0,
        (0.0, 1.0),
        dt_initial=0.1,
        tol=1e-6
    )

    expected = analytic_exponential(1.0)
    error = abs(states[-1][0] - expected)
    print(f"  Adaptive error: {error:.6e}")
    print(f"  Number of steps: {len(times)}")
    assert error < 1e-6, "Adaptive test failed"
    print("  ✓ Adaptive test passed")


def test_generic_interface():
    """Test generic integrate() function"""
    print("Testing generic integrate() interface...")
    state = np.array([1.0])

    # Test with different methods
    result_euler = integrate(exponential_decay, state, 0.0, 0.01, method="euler")
    result_rk2 = integrate(exponential_decay, state, 0.0, 0.01, method="rk2")
    result_rk4 = integrate(exponential_decay, state, 0.0, 0.01, method="rk4")

    assert result_euler is not None
    assert result_rk2 is not None
    assert result_rk4 is not None
    print("  ✓ Generic interface test passed")


def test_determinism():
    """Test that integrators are deterministic"""
    print("Testing determinism...")
    state1 = np.array([1.0, 2.0])
    state2 = np.array([1.0, 2.0])

    for _ in range(100):
        state1 = rk4(0, state1, simple_harmonic_oscillator, 0.01)
        state2 = rk4(0, state2, simple_harmonic_oscillator, 0.01)

    assert np.allclose(state1, state2), "Determinism test failed"
    print("  ✓ Determinism test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("INTEGRATORS DIALECT VERIFICATION")
    print("=" * 60)
    print()

    try:
        test_euler()
        test_rk2()
        test_rk4()
        test_verlet()
        test_symplectic()
        test_adaptive()
        test_generic_interface()
        test_determinism()

        print()
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)

    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        sys.exit(1)
