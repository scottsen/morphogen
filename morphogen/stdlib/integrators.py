"""Integrators dialect implementation using NumPy backend.

This module provides numerical integration methods for ordinary differential equations (ODEs)
and stochastic differential equations (SDEs). All integrators are designed for deterministic,
bit-exact computation with explicit timestep control.

Supported Methods:
- Euler (1st order explicit)
- RK2 (2nd order Runge-Kutta, midpoint method)
- RK4 (4th order Runge-Kutta, classic method)
- Verlet (Velocity Verlet, symplectic for physics)
- Leapfrog (Symplectic integration)
- Symplectic (Split-operator methods)
- Adaptive (Dormand-Prince 5(4) with error control)
"""

from typing import Callable, Tuple, Optional, Dict, Any
import numpy as np

from morphogen.core.operator import operator, OpCategory


# Type aliases for clarity
StateVector = np.ndarray  # Shape: (n,) or (n, d) where n=num_entities, d=dimensions
DerivativeFunc = Callable[[float, StateVector], StateVector]  # f(t, y) -> dy/dt
DerivativeFuncWithAccel = Callable[[float, StateVector, StateVector], StateVector]  # f(t, pos, vel) -> accel


class IntegratorResult:
    """Result of an integration step.

    Attributes:
        state: Updated state vector
        error: Local truncation error estimate (for adaptive methods)
        dt_next: Suggested next timestep (for adaptive methods)
    """
    def __init__(self, state: StateVector, error: Optional[float] = None, dt_next: Optional[float] = None):
        self.state = state
        self.error = error
        self.dt_next = dt_next


# ============================================================================
# EXPLICIT METHODS (General ODE solvers)
# ============================================================================

@operator(
    domain="integrators",
    category=OpCategory.INTEGRATE,
    signature="(t: float, state: StateVector, derivative: DerivativeFunc, dt: float) -> StateVector",
    deterministic=True,
    doc="Forward Euler method (1st order explicit)"
)
def euler(t: float, state: StateVector, derivative: DerivativeFunc, dt: float) -> StateVector:
    """Forward Euler method (1st order explicit).

    Simplest integration method: y(t+dt) = y(t) + dt * f(t, y)

    Properties:
    - Order: O(dt)
    - Stability: Poor (only for dt << 1/|lambda|)
    - Use case: Fast prototyping, non-stiff problems, teaching

    Args:
        t: Current time
        state: Current state vector
        derivative: Function computing dy/dt = f(t, y)
        dt: Timestep

    Returns:
        Updated state vector

    Example:
        # Simple harmonic oscillator: d²x/dt² = -k*x
        # State = [x, v], derivative = [v, -k*x]
        def deriv(t, state):
            x, v = state[0], state[1]
            return np.array([v, -k * x])

        state = np.array([1.0, 0.0])  # x=1, v=0
        state = euler(0.0, state, deriv, 0.01)
    """
    return state + dt * derivative(t, state)


@operator(
    domain="integrators",
    category=OpCategory.INTEGRATE,
    signature="(t: float, state: StateVector, derivative: DerivativeFunc, dt: float) -> StateVector",
    deterministic=True,
    doc="Runge-Kutta 2nd order (midpoint method)"
)
def rk2(t: float, state: StateVector, derivative: DerivativeFunc, dt: float) -> StateVector:
    """Runge-Kutta 2nd order (midpoint method).

    Two-stage method with O(dt²) accuracy:
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt/2 * k1)
    y(t+dt) = y(t) + dt * k2

    Properties:
    - Order: O(dt²)
    - Stability: Better than Euler
    - Use case: Balance between speed and accuracy

    Args:
        t: Current time
        state: Current state vector
        derivative: Function computing dy/dt = f(t, y)
        dt: Timestep

    Returns:
        Updated state vector
    """
    k1 = derivative(t, state)
    k2 = derivative(t + 0.5 * dt, state + 0.5 * dt * k1)
    return state + dt * k2


@operator(
    domain="integrators",
    category=OpCategory.INTEGRATE,
    signature="(t: float, state: StateVector, derivative: DerivativeFunc, dt: float) -> StateVector",
    deterministic=True,
    doc="Runge-Kutta 4th order (classic method)"
)
def rk4(t: float, state: StateVector, derivative: DerivativeFunc, dt: float) -> StateVector:
    """Runge-Kutta 4th order (classic method).

    Four-stage method with O(dt⁴) accuracy:
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt/2 * k1)
    k3 = f(t + dt/2, y + dt/2 * k2)
    k4 = f(t + dt, y + dt * k3)
    y(t+dt) = y(t) + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    Properties:
    - Order: O(dt⁴)
    - Stability: Excellent for general ODEs
    - Use case: Gold standard for moderate accuracy needs

    Args:
        t: Current time
        state: Current state vector
        derivative: Function computing dy/dt = f(t, y)
        dt: Timestep

    Returns:
        Updated state vector

    Example:
        # Lorenz attractor
        def lorenz(t, state):
            x, y, z = state
            sigma, rho, beta = 10.0, 28.0, 8.0/3.0
            return np.array([
                sigma * (y - x),
                x * (rho - z) - y,
                x * y - beta * z
            ])

        state = np.array([1.0, 1.0, 1.0])
        for _ in range(1000):
            state = rk4(0.0, state, lorenz, 0.01)
    """
    k1 = derivative(t, state)
    k2 = derivative(t + 0.5 * dt, state + 0.5 * dt * k1)
    k3 = derivative(t + 0.5 * dt, state + 0.5 * dt * k2)
    k4 = derivative(t + dt, state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ============================================================================
# SYMPLECTIC METHODS (For Hamiltonian systems and physics)
# ============================================================================

@operator(
    domain="integrators",
    category=OpCategory.INTEGRATE,
    signature="(t: float, position: StateVector, velocity: StateVector, acceleration: DerivativeFuncWithAccel, dt: float) -> Tuple[StateVector, StateVector]",
    deterministic=True,
    doc="Velocity Verlet method (symplectic integrator for physics)"
)
def verlet(
    t: float,
    position: StateVector,
    velocity: StateVector,
    acceleration: DerivativeFuncWithAccel,
    dt: float
) -> Tuple[StateVector, StateVector]:
    """Velocity Verlet method (symplectic integrator for physics).

    Time-reversible, symplectic integrator that conserves energy in Hamiltonian systems.

    Algorithm:
    v(t+dt/2) = v(t) + dt/2 * a(t, x(t), v(t))
    x(t+dt) = x(t) + dt * v(t+dt/2)
    a(t+dt) = acceleration(t+dt, x(t+dt), v(t+dt/2))
    v(t+dt) = v(t+dt/2) + dt/2 * a(t+dt)

    Properties:
    - Order: O(dt²) for position, O(dt) for velocity
    - Symplectic: Conserves phase space volume (energy conserving)
    - Time-reversible: Running backward gives exact original state
    - Use case: Molecular dynamics, celestial mechanics, particle physics

    Args:
        t: Current time
        position: Position vector(s)
        velocity: Velocity vector(s)
        acceleration: Function computing a = f(t, x, v)
        dt: Timestep

    Returns:
        Tuple of (new_position, new_velocity)

    Example:
        # N-body gravity simulation
        def gravity_accel(t, pos, vel):
            # Compute gravitational forces between particles
            n = len(pos)
            accel = np.zeros_like(pos)
            for i in range(n):
                for j in range(i+1, n):
                    r = pos[j] - pos[i]
                    dist = np.linalg.norm(r)
                    force = G * m[i] * m[j] / (dist**3) * r
                    accel[i] += force / m[i]
                    accel[j] -= force / m[j]
            return accel

        pos, vel = verlet(t, pos, vel, gravity_accel, dt)
    """
    # Half-step velocity update
    accel_current = acceleration(t, position, velocity)
    velocity_half = velocity + 0.5 * dt * accel_current

    # Full-step position update
    position_new = position + dt * velocity_half

    # Half-step velocity update (with new acceleration)
    accel_new = acceleration(t + dt, position_new, velocity_half)
    velocity_new = velocity_half + 0.5 * dt * accel_new

    return position_new, velocity_new


@operator(
    domain="integrators",
    category=OpCategory.INTEGRATE,
    signature="(t: float, position: StateVector, velocity: StateVector, acceleration: DerivativeFuncWithAccel, dt: float) -> Tuple[StateVector, StateVector]",
    deterministic=True,
    doc="Leapfrog integration (symplectic, staggered time steps)"
)
def leapfrog(
    t: float,
    position: StateVector,
    velocity: StateVector,
    acceleration: DerivativeFuncWithAccel,
    dt: float
) -> Tuple[StateVector, StateVector]:
    """Leapfrog integration (symplectic, staggered time steps).

    Similar to Verlet but with staggered updates (position and velocity evaluated at offset times).
    Velocity is offset by dt/2 from position.

    Algorithm:
    v(t+dt/2) = v(t-dt/2) + dt * a(t, x(t))
    x(t+dt) = x(t) + dt * v(t+dt/2)

    Properties:
    - Order: O(dt²)
    - Symplectic: Energy conserving
    - Use case: Same as Verlet (molecular dynamics, astrophysics)

    Note: This implementation uses Verlet internally for simplicity.
    True leapfrog requires tracking v(t-dt/2), which needs initialization.

    Args:
        t: Current time
        position: Position vector(s)
        velocity: Velocity vector(s) at time t-dt/2
        acceleration: Function computing a = f(t, x, v)
        dt: Timestep

    Returns:
        Tuple of (new_position, new_velocity at t+dt/2)
    """
    # Leapfrog is equivalent to Verlet with offset velocity
    # For practical purposes, we use Verlet implementation
    return verlet(t, position, velocity, acceleration, dt)


@operator(
    domain="integrators",
    category=OpCategory.INTEGRATE,
    signature="(t: float, position: StateVector, velocity: StateVector, potential_gradient: Callable, dt: float, order: int) -> Tuple[StateVector, StateVector]",
    deterministic=True,
    doc="Symplectic split-operator method for separable Hamiltonians"
)
def symplectic(
    t: float,
    position: StateVector,
    velocity: StateVector,
    potential_gradient: Callable[[StateVector], StateVector],
    dt: float,
    order: int = 2
) -> Tuple[StateVector, StateVector]:
    """Symplectic split-operator method for separable Hamiltonians.

    For Hamiltonians of form H(p,q) = T(p) + V(q) (kinetic + potential),
    uses operator splitting to preserve symplectic structure.

    Algorithm (2nd order):
    v(t+dt/2) = v(t) - dt/2 * ∇V(x(t))
    x(t+dt) = x(t) + dt * v(t+dt/2)
    v(t+dt) = v(t+dt/2) - dt/2 * ∇V(x(t+dt))

    Properties:
    - Order: O(dt²) for order=2, O(dt⁴) for order=4
    - Symplectic: Exactly preserves phase space volume
    - Use case: Molecular dynamics, celestial mechanics with separable potentials

    Args:
        t: Current time (unused, for API compatibility)
        position: Position vector(s)
        velocity: Velocity vector(s)
        potential_gradient: Function computing -∇V(x) (force per unit mass)
        dt: Timestep
        order: Integration order (2 or 4)

    Returns:
        Tuple of (new_position, new_velocity)

    Example:
        # Harmonic oscillator: V(x) = 0.5 * k * x²
        def force(x):
            k = 1.0
            return -k * x  # F = -∇V = -k*x

        pos, vel = symplectic(t, pos, vel, force, dt, order=2)
    """
    if order == 2:
        # 2nd order symplectic (Störmer-Verlet)
        velocity_half = velocity + 0.5 * dt * potential_gradient(position)
        position_new = position + dt * velocity_half
        velocity_new = velocity_half + 0.5 * dt * potential_gradient(position_new)
        return position_new, velocity_new

    elif order == 4:
        # 4th order symplectic (Yoshida/Forest-Ruth)
        # Coefficients for 4th order composition
        # Reference: Yoshida, Physics Letters A 150, 262 (1990)
        theta = 1.0 / (2.0 - 2.0**(1.0/3.0))  # ≈ 1.3512
        w0 = -2.0**(1.0/3.0) / (2.0 - 2.0**(1.0/3.0))  # ≈ -1.7024
        w1 = theta  # ≈ 1.3512

        # Velocity kick coefficients (c_i)
        c1 = c4 = w1 / 2.0
        c2 = c3 = (w0 + w1) / 2.0

        # Position drift coefficients (d_i)
        d1 = d3 = w1
        d2 = w0

        # 4-stage composition: kick-drift-kick-drift-kick-drift-kick
        velocity = velocity + c1 * dt * potential_gradient(position)
        position = position + d1 * dt * velocity
        velocity = velocity + c2 * dt * potential_gradient(position)
        position = position + d2 * dt * velocity
        velocity = velocity + c3 * dt * potential_gradient(position)
        position = position + d3 * dt * velocity
        velocity = velocity + c4 * dt * potential_gradient(position)

        return position, velocity

    else:
        raise ValueError(f"Unsupported order: {order}. Use 2 or 4.")


# ============================================================================
# ADAPTIVE METHODS (Variable timestep with error control)
# ============================================================================

@operator(
    domain="integrators",
    category=OpCategory.INTEGRATE,
    signature="(t: float, state: StateVector, derivative: DerivativeFunc, dt: float, tol: float) -> IntegratorResult",
    deterministic=True,
    doc="Dormand-Prince 5(4) adaptive step with error control"
)
def dormand_prince_step(
    t: float,
    state: StateVector,
    derivative: DerivativeFunc,
    dt: float,
    tol: float = 1e-6
) -> IntegratorResult:
    """Dormand-Prince 5(4) adaptive step with error control.

    Embedded Runge-Kutta method that estimates local error and suggests next timestep.
    Uses 5th order solution with 4th order error estimate (DOPRI5).

    Properties:
    - Order: O(dt⁵) solution, O(dt⁴) error estimate
    - Adaptive: Automatically adjusts timestep to meet tolerance
    - Use case: Stiff/non-stiff ODEs, when accuracy is critical

    Args:
        t: Current time
        state: Current state vector
        derivative: Function computing dy/dt = f(t, y)
        dt: Suggested timestep (will be adjusted)
        tol: Error tolerance (default 1e-6)

    Returns:
        IntegratorResult with updated state, error estimate, and suggested next dt

    Example:
        # Van der Pol oscillator (stiff for large mu)
        def van_der_pol(t, state):
            x, v = state
            mu = 10.0
            return np.array([v, mu * (1 - x**2) * v - x])

        state = np.array([2.0, 0.0])
        dt = 0.1
        for step in range(1000):
            result = dormand_prince_step(t, state, van_der_pol, dt)
            state = result.state
            dt = result.dt_next
            t += dt
    """
    # Dormand-Prince coefficients (DOPRI5)
    # a_ij coefficients (Butcher tableau)
    a21 = 1.0 / 5.0
    a31, a32 = 3.0 / 40.0, 9.0 / 40.0
    a41, a42, a43 = 44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0
    a51, a52, a53, a54 = 19372.0 / 6561.0, -25360.0 / 2187.0, 64448.0 / 6561.0, -212.0 / 729.0
    a61, a62, a63, a64, a65 = 9017.0 / 3168.0, -355.0 / 33.0, 46732.0 / 5247.0, 49.0 / 176.0, -5103.0 / 18656.0

    # b_i coefficients (5th order solution)
    b1, b3, b4, b5, b6 = 35.0 / 384.0, 500.0 / 1113.0, 125.0 / 192.0, -2187.0 / 6784.0, 11.0 / 84.0

    # b*_i coefficients (4th order solution for error estimation)
    bs1, bs3, bs4, bs5, bs6, bs7 = 5179.0 / 57600.0, 7571.0 / 16695.0, 393.0 / 640.0, -92097.0 / 339200.0, 187.0 / 2100.0, 1.0 / 40.0

    # c_i coefficients (time offsets)
    c2, c3, c4, c5 = 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0

    # Compute k stages
    k1 = derivative(t, state)
    k2 = derivative(t + c2 * dt, state + dt * a21 * k1)
    k3 = derivative(t + c3 * dt, state + dt * (a31 * k1 + a32 * k2))
    k4 = derivative(t + c4 * dt, state + dt * (a41 * k1 + a42 * k2 + a43 * k3))
    k5 = derivative(t + c5 * dt, state + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
    k6 = derivative(t + dt, state + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))

    # 5th order solution
    state_new = state + dt * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)

    # 4th order solution (for error estimation)
    k7 = derivative(t + dt, state_new)  # Reuse final evaluation
    state_err = state + dt * (bs1 * k1 + bs3 * k3 + bs4 * k4 + bs5 * k5 + bs6 * k6 + bs7 * k7)

    # Error estimate (L2 norm of difference)
    error = np.linalg.norm(state_new - state_err)

    # Adaptive timestep selection (safety factor 0.9)
    if error > 0:
        dt_next = 0.9 * dt * (tol / error) ** 0.2
    else:
        dt_next = dt * 1.5  # Increase timestep if error is very small

    # Clamp timestep to reasonable range
    dt_next = np.clip(dt_next, dt * 0.1, dt * 5.0)

    return IntegratorResult(state_new, error, dt_next)


@operator(
    domain="integrators",
    category=OpCategory.INTEGRATE,
    signature="(derivative: DerivativeFunc, state0: StateVector, t_span: Tuple[float, float], dt_initial: float, tol: float, method: str) -> Tuple[ndarray, ndarray]",
    deterministic=True,
    doc="Adaptive integration with error control over time interval"
)
def adaptive_integrate(
    derivative: DerivativeFunc,
    state0: StateVector,
    t_span: Tuple[float, float],
    dt_initial: float = 0.01,
    tol: float = 1e-6,
    method: str = "dopri5"
) -> Tuple[np.ndarray, np.ndarray]:
    """Adaptive integration with error control over time interval.

    Integrates ODE from t_span[0] to t_span[1] with automatic timestep adjustment.

    Args:
        derivative: Function computing dy/dt = f(t, y)
        state0: Initial state vector
        t_span: Tuple (t_start, t_end)
        dt_initial: Initial timestep guess
        tol: Error tolerance
        method: Integration method ("dopri5" only for now)

    Returns:
        Tuple of (time_array, state_array) where:
        - time_array: 1D array of time points
        - state_array: 2D array of states at each time point

    Example:
        # Integrate simple harmonic oscillator from t=0 to t=10
        def deriv(t, state):
            x, v = state
            return np.array([v, -x])

        times, states = adaptive_integrate(deriv, np.array([1.0, 0.0]), (0, 10))
    """
    if method != "dopri5":
        raise ValueError(f"Unsupported method: {method}")

    t_start, t_end = t_span
    t = t_start
    state = state0.copy()
    dt = dt_initial

    # Storage for solution
    times = [t]
    states = [state.copy()]

    while t < t_end:
        # Don't overshoot final time
        if t + dt > t_end:
            dt = t_end - t

        # Take adaptive step
        result = dormand_prince_step(t, state, derivative, dt, tol)

        # Accept step if error is within tolerance
        if result.error is None or result.error <= tol:
            state = result.state
            t += dt
            times.append(t)
            states.append(state.copy())

            # Update timestep for next iteration
            if result.dt_next is not None:
                dt = result.dt_next
        else:
            # Reject step and retry with smaller timestep
            dt = result.dt_next

    return np.array(times), np.array(states)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@operator(
    domain="integrators",
    category=OpCategory.INTEGRATE,
    signature="(derivative: DerivativeFunc, state: StateVector, t: float, dt: float, method: str) -> StateVector",
    deterministic=True,
    doc="Generic integration interface with method selection"
)
def integrate(
    derivative: DerivativeFunc,
    state: StateVector,
    t: float,
    dt: float,
    method: str = "rk4"
) -> StateVector:
    """Generic integration interface with method selection.

    Convenience function that dispatches to appropriate integrator.

    Args:
        derivative: Function computing dy/dt = f(t, y)
        state: Current state vector
        t: Current time
        dt: Timestep
        method: Integration method ("euler", "rk2", "rk4")

    Returns:
        Updated state vector

    Example:
        state = integrate(deriv, state, t, dt, method="rk4")
    """
    if method == "euler":
        return euler(t, state, derivative, dt)
    elif method == "rk2":
        return rk2(t, state, derivative, dt)
    elif method == "rk4":
        return rk4(t, state, derivative, dt)
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'euler', 'rk2', or 'rk4'.")


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Explicit methods
    'euler',
    'rk2',
    'rk4',

    # Symplectic methods
    'verlet',
    'leapfrog',
    'symplectic',

    # Adaptive methods
    'dormand_prince_step',
    'adaptive_integrate',

    # Utilities
    'integrate',
    'IntegratorResult',
]
