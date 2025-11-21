"""Tests for integrators dialect (numerical ODE/SDE integration).

Tests all integration methods for correctness, accuracy, and determinism.
"""

import pytest
import numpy as np
from morphogen.stdlib.integrators import (
    euler, rk2, rk4,
    verlet, leapfrog, symplectic,
    dormand_prince_step, adaptive_integrate,
    integrate, IntegratorResult
)


# ============================================================================
# TEST FIXTURES AND HELPER FUNCTIONS
# ============================================================================

def simple_harmonic_oscillator(t, state):
    """SHO: d²x/dt² = -k*x, state = [x, v]"""
    x, v = state[0], state[1]
    k = 1.0
    return np.array([v, -k * x])


def simple_harmonic_oscillator_accel(t, pos, vel):
    """SHO acceleration for symplectic integrators"""
    k = 1.0
    return -k * pos


def exponential_decay(t, state):
    """dy/dt = -lambda * y"""
    lam = 1.0
    return -lam * state


def lorenz_system(t, state):
    """Lorenz attractor"""
    x, y, z = state
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    return np.array([
        sigma * (y - x),
        x * (rho - z) - y,
        x * y - beta * z
    ])


def analytic_sho(t, x0=1.0, v0=0.0, k=1.0):
    """Analytic solution for SHO"""
    omega = np.sqrt(k)
    A = x0
    B = v0 / omega
    x = A * np.cos(omega * t) + B * np.sin(omega * t)
    v = -A * omega * np.sin(omega * t) + B * omega * np.cos(omega * t)
    return np.array([x, v])


def analytic_exponential(t, y0=1.0, lam=1.0):
    """Analytic solution for exponential decay"""
    return y0 * np.exp(-lam * t)


def sho_energy(state, k=1.0, m=1.0):
    """Total energy of simple harmonic oscillator"""
    x, v = state[0], state[1]
    return 0.5 * k * x**2 + 0.5 * m * v**2


# ============================================================================
# TESTS: EXPLICIT METHODS
# ============================================================================

class TestExplicitMethods:
    """Tests for Euler, RK2, RK4"""

    def test_euler_basic(self):
        """Test Euler method on simple exponential decay"""
        state = np.array([1.0])
        t = 0.0
        dt = 0.01

        # Integrate for 100 steps
        for _ in range(100):
            state = euler(t, state, exponential_decay, dt)
            t += dt

        # Compare with analytic solution
        expected = analytic_exponential(1.0, y0=1.0, lam=1.0)
        assert np.abs(state[0] - expected) < 0.05  # Euler has large error

    def test_euler_sho(self):
        """Test Euler on simple harmonic oscillator"""
        state = np.array([1.0, 0.0])  # x=1, v=0
        t = 0.0
        dt = 0.01

        # Integrate for one period (T = 2π)
        n_steps = int(2 * np.pi / dt)
        for _ in range(n_steps):
            state = euler(t, state, simple_harmonic_oscillator, dt)
            t += dt

        # Should return close to initial state (but Euler has energy drift)
        expected = analytic_sho(2 * np.pi, x0=1.0, v0=0.0)
        assert np.linalg.norm(state - expected) < 0.5  # Euler accumulates error

    def test_rk2_exponential(self):
        """Test RK2 method on exponential decay"""
        state = np.array([1.0])
        t = 0.0
        dt = 0.01

        for _ in range(100):
            state = rk2(t, state, exponential_decay, dt)
            t += dt

        expected = analytic_exponential(1.0, y0=1.0, lam=1.0)
        assert np.abs(state[0] - expected) < 1e-3  # RK2 is much better

    def test_rk2_sho(self):
        """Test RK2 on simple harmonic oscillator"""
        state = np.array([1.0, 0.0])
        t = 0.0
        dt = 0.01

        n_steps = int(2 * np.pi / dt)
        for _ in range(n_steps):
            state = rk2(t, state, simple_harmonic_oscillator, dt)
            t += dt

        expected = analytic_sho(2 * np.pi, x0=1.0, v0=0.0)
        assert np.linalg.norm(state - expected) < 0.01

    def test_rk4_exponential(self):
        """Test RK4 method on exponential decay"""
        state = np.array([1.0])
        t = 0.0
        dt = 0.01

        for _ in range(100):
            state = rk4(t, state, exponential_decay, dt)
            t += dt

        expected = analytic_exponential(1.0, y0=1.0, lam=1.0)
        assert np.abs(state[0] - expected) < 1e-6  # RK4 is very accurate

    def test_rk4_sho(self):
        """Test RK4 on simple harmonic oscillator"""
        state = np.array([1.0, 0.0])
        t = 0.0
        dt = 0.01

        n_steps = int(2 * np.pi / dt)
        for _ in range(n_steps):
            state = rk4(t, state, simple_harmonic_oscillator, dt)
            t += dt

        expected = analytic_sho(2 * np.pi, x0=1.0, v0=0.0)
        # Note: int(2π/dt) doesn't give exact period, so some phase error expected
        assert np.linalg.norm(state - expected) < 5e-3

    def test_rk4_lorenz(self):
        """Test RK4 on chaotic Lorenz system"""
        state = np.array([1.0, 1.0, 1.0])
        t = 0.0
        dt = 0.01

        # Just verify it doesn't blow up
        for _ in range(1000):
            state = rk4(t, state, lorenz_system, dt)
            t += dt

        # Verify state is bounded (Lorenz attractor is bounded)
        assert np.all(np.abs(state) < 100)

    def test_method_comparison(self):
        """Compare accuracy of Euler vs RK2 vs RK4"""
        state_euler = np.array([1.0])
        state_rk2 = np.array([1.0])
        state_rk4 = np.array([1.0])
        t = 0.0
        dt = 0.1  # Larger timestep to see differences

        for _ in range(10):
            state_euler = euler(t, state_euler, exponential_decay, dt)
            state_rk2 = rk2(t, state_rk2, exponential_decay, dt)
            state_rk4 = rk4(t, state_rk4, exponential_decay, dt)
            t += dt

        expected = analytic_exponential(1.0, y0=1.0, lam=1.0)

        error_euler = np.abs(state_euler[0] - expected)
        error_rk2 = np.abs(state_rk2[0] - expected)
        error_rk4 = np.abs(state_rk4[0] - expected)

        # Verify order of accuracy: Euler > RK2 > RK4
        assert error_euler > error_rk2
        assert error_rk2 > error_rk4


# ============================================================================
# TESTS: SYMPLECTIC METHODS
# ============================================================================

class TestSymplecticMethods:
    """Tests for Verlet, Leapfrog, Symplectic"""

    def test_verlet_sho(self):
        """Test Verlet integrator on SHO"""
        pos = np.array([1.0])
        vel = np.array([0.0])
        t = 0.0
        dt = 0.01

        # Integrate for one period
        n_steps = int(2 * np.pi / dt)
        for _ in range(n_steps):
            pos, vel = verlet(t, pos, vel, simple_harmonic_oscillator_accel, dt)
            t += dt

        # Should return close to initial state
        # Note: int(2π/dt) doesn't give exact period, so some phase error expected
        assert np.abs(pos[0] - 1.0) < 5e-3
        assert np.abs(vel[0]) < 5e-3

    def test_verlet_energy_conservation(self):
        """Test that Verlet conserves energy in SHO"""
        pos = np.array([1.0])
        vel = np.array([0.0])
        t = 0.0
        dt = 0.01

        initial_energy = sho_energy(np.array([pos[0], vel[0]]))

        # Integrate for many periods
        for _ in range(1000):
            pos, vel = verlet(t, pos, vel, simple_harmonic_oscillator_accel, dt)
            t += dt

        final_energy = sho_energy(np.array([pos[0], vel[0]]))

        # Energy should be conserved to high precision
        energy_drift = np.abs(final_energy - initial_energy) / initial_energy
        assert energy_drift < 1e-4  # < 0.01% drift

    def test_verlet_multidimensional(self):
        """Test Verlet with 2D positions"""
        pos = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2 particles in 2D
        vel = np.array([[0.0, 1.0], [-1.0, 0.0]])

        def accel_2d(t, p, v):
            # Simple harmonic oscillator in 2D
            k = 1.0
            return -k * p

        dt = 0.01
        for _ in range(100):
            pos, vel = verlet(0, pos, vel, accel_2d, dt)

        # Just verify shape is preserved
        assert pos.shape == (2, 2)
        assert vel.shape == (2, 2)

    def test_leapfrog_sho(self):
        """Test Leapfrog integrator"""
        pos = np.array([1.0])
        vel = np.array([0.0])
        t = 0.0
        dt = 0.01

        n_steps = int(2 * np.pi / dt)
        for _ in range(n_steps):
            pos, vel = leapfrog(t, pos, vel, simple_harmonic_oscillator_accel, dt)
            t += dt

        # Should return close to initial state
        # Note: int(2π/dt) doesn't give exact period, so some phase error expected
        assert np.abs(pos[0] - 1.0) < 5e-3
        assert np.abs(vel[0]) < 5e-3

    def test_symplectic_order2(self):
        """Test 2nd order symplectic integrator"""
        pos = np.array([1.0])
        vel = np.array([0.0])

        def force(x):
            k = 1.0
            return -k * x

        dt = 0.01
        n_steps = int(2 * np.pi / dt)

        for _ in range(n_steps):
            pos, vel = symplectic(0, pos, vel, force, dt, order=2)

        # Note: int(2π/dt) doesn't give exact period, so some phase error expected
        assert np.abs(pos[0] - 1.0) < 5e-3
        assert np.abs(vel[0]) < 5e-3

    def test_symplectic_order4(self):
        """Test 4th order symplectic integrator"""
        pos = np.array([1.0])
        vel = np.array([0.0])

        def force(x):
            k = 1.0
            return -k * x

        dt = 0.05  # Can use larger timestep with 4th order
        n_steps = int(2 * np.pi / dt)

        for _ in range(n_steps):
            pos, vel = symplectic(0, pos, vel, force, dt, order=4)

        # Should be more accurate than order 2
        # Note: int(2π/dt) doesn't give exact period, so some phase error expected
        # With dt=0.05, error is ~5e-4 due to phase mismatch (125 steps × 0.05 = 6.25 ≠ 2π)
        assert np.abs(pos[0] - 1.0) < 1e-3
        assert np.abs(vel[0]) < 0.04

    def test_symplectic_energy_conservation(self):
        """Test energy conservation in symplectic integrators"""
        pos = np.array([1.0])
        vel = np.array([0.0])

        def force(x):
            return -x

        initial_energy = sho_energy(np.array([pos[0], vel[0]]))

        dt = 0.01
        for _ in range(1000):
            pos, vel = symplectic(0, pos, vel, force, dt, order=2)

        final_energy = sho_energy(np.array([pos[0], vel[0]]))
        energy_drift = np.abs(final_energy - initial_energy) / initial_energy

        assert energy_drift < 1e-4

    def test_symplectic_invalid_order(self):
        """Test that invalid order raises error"""
        pos = np.array([1.0])
        vel = np.array([0.0])

        def force(x):
            return -x

        with pytest.raises(ValueError, match="Unsupported order"):
            symplectic(0, pos, vel, force, 0.01, order=3)


# ============================================================================
# TESTS: ADAPTIVE METHODS
# ============================================================================

class TestAdaptiveMethods:
    """Tests for adaptive timestep integrators"""

    def test_dormand_prince_basic(self):
        """Test Dormand-Prince step"""
        state = np.array([1.0])
        t = 0.0
        dt = 0.1

        result = dormand_prince_step(t, state, exponential_decay, dt)

        assert isinstance(result, IntegratorResult)
        assert result.state is not None
        assert result.error is not None
        assert result.dt_next is not None

    def test_dormand_prince_accuracy(self):
        """Test Dormand-Prince accuracy on exponential decay"""
        state = np.array([1.0])
        t = 0.0
        dt = 0.1

        # Take 10 steps with FIXED timestep to reach predictable endpoint
        for _ in range(10):
            result = dormand_prince_step(t, state, exponential_decay, dt)
            state = result.state
            # Keep dt fixed (don't use adaptive dt) for predictable endpoint
            t += dt

        # Should be very accurate at t=1.0
        expected = analytic_exponential(1.0, y0=1.0, lam=1.0)
        assert np.abs(state[0] - expected) < 1e-6

    def test_dormand_prince_error_control(self):
        """Test that error estimate is reasonable"""
        state = np.array([1.0])
        t = 0.0
        dt = 0.01

        result = dormand_prince_step(t, state, exponential_decay, dt, tol=1e-6)

        # Error should be small
        assert result.error < 1e-4

    def test_dormand_prince_timestep_adaptation(self):
        """Test that timestep adapts to local error"""
        state = np.array([1.0])
        t = 0.0
        dt = 1.0  # Start with large timestep

        result = dormand_prince_step(t, state, exponential_decay, dt, tol=1e-6)

        # Timestep should decrease if error is large
        if result.error > 1e-6:
            assert result.dt_next < dt

    def test_adaptive_integrate_basic(self):
        """Test adaptive_integrate function"""
        state0 = np.array([1.0])
        times, states = adaptive_integrate(
            exponential_decay,
            state0,
            (0.0, 1.0),
            dt_initial=0.1,
            tol=1e-6
        )

        assert len(times) > 0
        assert len(states) == len(times)
        assert times[0] == 0.0
        assert times[-1] == 1.0

        # Check final state accuracy
        expected = analytic_exponential(1.0, y0=1.0, lam=1.0)
        assert np.abs(states[-1][0] - expected) < 1e-6

    def test_adaptive_integrate_sho(self):
        """Test adaptive integration on SHO"""
        state0 = np.array([1.0, 0.0])
        times, states = adaptive_integrate(
            simple_harmonic_oscillator,
            state0,
            (0.0, 2 * np.pi),
            dt_initial=0.1,
            tol=1e-6
        )

        # Should return close to initial state after one period
        final_state = states[-1]
        assert np.linalg.norm(final_state - state0) < 1e-4

    def test_adaptive_integrate_invalid_method(self):
        """Test that invalid method raises error"""
        state0 = np.array([1.0])

        with pytest.raises(ValueError, match="Unsupported method"):
            adaptive_integrate(exponential_decay, state0, (0, 1), method="invalid")


# ============================================================================
# TESTS: UTILITY FUNCTIONS
# ============================================================================

class TestUtilities:
    """Tests for utility functions"""

    def test_integrate_euler(self):
        """Test generic integrate() with Euler method"""
        state = np.array([1.0])
        result = integrate(exponential_decay, state, 0.0, 0.01, method="euler")
        expected = euler(0.0, state, exponential_decay, 0.01)
        assert np.allclose(result, expected)

    def test_integrate_rk2(self):
        """Test generic integrate() with RK2 method"""
        state = np.array([1.0])
        result = integrate(exponential_decay, state, 0.0, 0.01, method="rk2")
        expected = rk2(0.0, state, exponential_decay, 0.01)
        assert np.allclose(result, expected)

    def test_integrate_rk4(self):
        """Test generic integrate() with RK4 method"""
        state = np.array([1.0])
        result = integrate(exponential_decay, state, 0.0, 0.01, method="rk4")
        expected = rk4(0.0, state, exponential_decay, 0.01)
        assert np.allclose(result, expected)

    def test_integrate_invalid_method(self):
        """Test that invalid method raises error"""
        state = np.array([1.0])

        with pytest.raises(ValueError, match="Unsupported method"):
            integrate(exponential_decay, state, 0.0, 0.01, method="invalid")


# ============================================================================
# TESTS: DETERMINISM
# ============================================================================

class TestDeterminism:
    """Verify all integrators are deterministic"""

    def test_euler_determinism(self):
        """Test Euler is deterministic"""
        state1 = np.array([1.0, 2.0])
        state2 = np.array([1.0, 2.0])

        for _ in range(100):
            state1 = euler(0, state1, simple_harmonic_oscillator, 0.01)
            state2 = euler(0, state2, simple_harmonic_oscillator, 0.01)

        assert np.allclose(state1, state2)

    def test_rk4_determinism(self):
        """Test RK4 is deterministic"""
        state1 = np.array([1.0, 2.0])
        state2 = np.array([1.0, 2.0])

        for _ in range(100):
            state1 = rk4(0, state1, simple_harmonic_oscillator, 0.01)
            state2 = rk4(0, state2, simple_harmonic_oscillator, 0.01)

        assert np.allclose(state1, state2)

    def test_verlet_determinism(self):
        """Test Verlet is deterministic"""
        pos1 = np.array([1.0])
        vel1 = np.array([0.0])
        pos2 = np.array([1.0])
        vel2 = np.array([0.0])

        for _ in range(100):
            pos1, vel1 = verlet(0, pos1, vel1, simple_harmonic_oscillator_accel, 0.01)
            pos2, vel2 = verlet(0, pos2, vel2, simple_harmonic_oscillator_accel, 0.01)

        assert np.allclose(pos1, pos2)
        assert np.allclose(vel1, vel2)

    def test_adaptive_determinism(self):
        """Test adaptive integrator is deterministic"""
        state0 = np.array([1.0])

        times1, states1 = adaptive_integrate(
            exponential_decay, state0, (0, 1), dt_initial=0.1, tol=1e-6
        )
        times2, states2 = adaptive_integrate(
            exponential_decay, state0, (0, 1), dt_initial=0.1, tol=1e-6
        )

        assert np.allclose(times1, times2)
        assert np.allclose(states1, states2)


# ============================================================================
# TESTS: EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_zero_timestep(self):
        """Test with zero timestep"""
        state = np.array([1.0])
        result = euler(0, state, exponential_decay, 0.0)
        assert np.allclose(result, state)

    def test_negative_timestep(self):
        """Test with negative timestep (time reversal)"""
        state = np.array([1.0])

        # Forward
        state_forward = rk4(0, state, exponential_decay, 0.01)

        # Backward
        state_backward = rk4(0.01, state_forward, exponential_decay, -0.01)

        # Should approximately return to original state
        assert np.abs(state_backward[0] - state[0]) < 1e-6

    def test_large_timestep(self):
        """Test that large timestep doesn't crash (but may be inaccurate)"""
        state = np.array([1.0])
        result = euler(0, state, exponential_decay, 100.0)

        # Should still produce a result (even if inaccurate)
        assert result is not None
        assert not np.any(np.isnan(result))

    def test_multidimensional_state(self):
        """Test with high-dimensional state vectors"""
        state = np.random.rand(100)

        def linear_system(t, y):
            return -0.1 * y  # Simple decay

        result = rk4(0, state, linear_system, 0.01)

        assert result.shape == state.shape
        assert not np.any(np.isnan(result))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
