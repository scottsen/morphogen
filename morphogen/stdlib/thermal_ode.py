"""ThermalODEDomain - 1D thermal modeling (heat transfer in pipes, rods, walls).

This module implements 1D thermal analysis using ODE integration. Essential for
fire pits, heat exchangers, hotends, battery thermal management, and heat pipes.

Specification: docs/specifications/physics-domains.md
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
from enum import Enum

from morphogen.core.operator import operator, OpCategory


# ============================================================================
# Core Types
# ============================================================================

@dataclass
class ThermalSegment:
    """1D thermal segment (tube, rod, wall)."""
    length: float  # m
    diameter: float  # m
    wall_thickness: float  # m
    conductivity: float  # W/(m·K)
    emissivity: float  # 0-1
    fluid: Optional['Fluid'] = None  # If fluid flows through


@dataclass
class WallTempModel:
    """Wall temperature profile (boundary condition)."""
    profile: Callable[[float], float]  # x (m) -> T (K)

    @staticmethod
    def constant(T: float) -> 'WallTempModel':
        """Constant wall temperature."""
        return WallTempModel(lambda x: T)

    @staticmethod
    def linear(T_hot: float, T_amb: float, length: float) -> 'WallTempModel':
        """Linear temperature decay."""
        return WallTempModel(lambda x: T_hot - (T_hot - T_amb) * (x / length))

    @staticmethod
    def exponential(T_hot: float, T_amb: float, L_char: float) -> 'WallTempModel':
        """Exponential temperature decay."""
        return WallTempModel(lambda x: T_amb + (T_hot - T_amb) * np.exp(-x / L_char))


@dataclass
class ThermalProfile:
    """Temperature distribution along segment."""
    temperatures: np.ndarray  # K
    positions: np.ndarray  # m


# ============================================================================
# Operators
# ============================================================================

@operator(
    domain="thermal_ode",
    category=OpCategory.CONSTRUCT,
    signature="(T_hot: float, T_amb: float, length: float, model: str) -> WallTempModel",
    deterministic=True,
    doc="Create wall temperature profile model"
)
def wall_temp_model(
    T_hot: float,
    T_amb: float,
    length: float,
    model: str = "linear"
) -> WallTempModel:
    """Estimate wall temperature profile along tube.

    Args:
        T_hot: Hot end temperature (K)
        T_amb: Ambient temperature (K)
        length: Tube length (m)
        model: "constant", "linear", or "exponential"

    Returns:
        WallTempModel callable

    Determinism: repro
    """
    if model == "constant":
        return WallTempModel.constant(T_hot)
    elif model == "linear":
        return WallTempModel.linear(T_hot, T_amb, length)
    elif model == "exponential":
        L_char = length / 3.0  # Characteristic length
        return WallTempModel.exponential(T_hot, T_amb, L_char)
    else:
        raise ValueError(f"Unknown model: {model}")


@operator(
    domain="thermal_ode",
    category=OpCategory.INTEGRATE,
    signature="(segment: ThermalSegment, m_dot: float, T_in: float, wall_temp_model: WallTempModel, integrator: str, steps: int) -> float",
    deterministic=True,
    doc="Solve 1D heat transfer ODE"
)
def heat_transfer_1D(
    segment: ThermalSegment,
    m_dot: float,
    T_in: float,
    wall_temp_model: WallTempModel,
    integrator: str = "rk4",
    steps: int = 100
) -> float:
    """Solve 1D heat transfer ODE for fluid heating in tube.

    Governing equation:
        m_dot·c_p·dT/dx = h·A_s·(T_wall(x) - T_air)

    Where:
        h = heat transfer coefficient (W/(m²·K))
        A_s = surface area per unit length (π·D)

    Args:
        segment: Thermal segment geometry
        m_dot: Mass flow rate (kg/s)
        T_in: Inlet temperature (K)
        wall_temp_model: Wall temperature profile
        integrator: "euler", "rk2", "rk4"
        steps: Number of integration steps

    Returns:
        T_out: Outlet temperature (K)

    Determinism: repro
    """
    if segment.fluid is None:
        raise ValueError("Segment must have fluid properties")

    # Fluid properties
    c_p = segment.fluid.specific_heat  # J/(kg·K)
    rho = segment.fluid.density  # kg/m³
    mu = segment.fluid.viscosity  # Pa·s

    # Geometric properties
    D = segment.diameter  # m
    A = np.pi * (D / 2.0)**2  # Cross-sectional area
    L = segment.length  # m
    dx = L / steps

    # Velocity and Reynolds number
    velocity = m_dot / (rho * A) if A > 0 else 0.0
    Re = rho * velocity * D / mu if mu > 0 else 0.0

    # Heat transfer coefficient (Nusselt correlation for turbulent flow)
    if Re > 2300:
        # Dittus-Boelter correlation: Nu = 0.023·Re^0.8·Pr^0.4
        Pr = mu * c_p / segment.conductivity  # Prandtl number
        Nu = 0.023 * Re**0.8 * Pr**0.4
    else:
        # Laminar flow: Nu = 3.66 (constant heat flux)
        Nu = 3.66

    h = Nu * segment.conductivity / D  # Heat transfer coefficient (W/(m²·K))
    A_s = np.pi * D  # Surface area per unit length (m²/m)

    # ODE: dT/dx = (h·A_s / (m_dot·c_p)) · (T_wall(x) - T)
    def dT_dx(x, T):
        T_wall = wall_temp_model.profile(x)
        return (h * A_s / (m_dot * c_p)) * (T_wall - T)

    # Integration
    T = T_in
    x = 0.0

    if integrator == "euler":
        for i in range(steps):
            T = T + dT_dx(x, T) * dx
            x += dx

    elif integrator == "rk2":
        for i in range(steps):
            k1 = dT_dx(x, T)
            k2 = dT_dx(x + dx/2, T + k1*dx/2)
            T = T + k2 * dx
            x += dx

    elif integrator == "rk4":
        for i in range(steps):
            k1 = dT_dx(x, T)
            k2 = dT_dx(x + dx/2, T + k1*dx/2)
            k3 = dT_dx(x + dx/2, T + k2*dx/2)
            k4 = dT_dx(x + dx, T + k3*dx)
            T = T + (k1 + 2*k2 + 2*k3 + k4) * dx / 6.0
            x += dx

    else:
        raise ValueError(f"Unknown integrator: {integrator}")

    return T


@operator(
    domain="thermal_ode",
    category=OpCategory.QUERY,
    signature="(segment: ThermalSegment, m_dot: float, T_in: float, wall_temp_model: WallTempModel, integrator: str, steps: int) -> ThermalProfile",
    deterministic=True,
    doc="Solve 1D heat transfer and return full profile"
)
def heat_transfer_1D_profile(
    segment: ThermalSegment,
    m_dot: float,
    T_in: float,
    wall_temp_model: WallTempModel,
    integrator: str = "rk4",
    steps: int = 100
) -> ThermalProfile:
    """Solve 1D heat transfer ODE and return full temperature profile.

    Args:
        segment: Thermal segment geometry
        m_dot: Mass flow rate (kg/s)
        T_in: Inlet temperature (K)
        wall_temp_model: Wall temperature profile
        integrator: "euler", "rk2", "rk4"
        steps: Number of integration steps

    Returns:
        ThermalProfile with temperatures and positions

    Determinism: repro
    """
    if segment.fluid is None:
        raise ValueError("Segment must have fluid properties")

    # Initialize arrays
    positions = np.linspace(0, segment.length, steps + 1)
    temperatures = np.zeros(steps + 1)
    temperatures[0] = T_in

    # Fluid properties
    c_p = segment.fluid.specific_heat
    rho = segment.fluid.density
    mu = segment.fluid.viscosity

    # Geometric properties
    D = segment.diameter
    A = np.pi * (D / 2.0)**2
    dx = segment.length / steps

    # Velocity and Reynolds number
    velocity = m_dot / (rho * A) if A > 0 else 0.0
    Re = rho * velocity * D / mu if mu > 0 else 0.0

    # Heat transfer coefficient
    if Re > 2300:
        Pr = mu * c_p / segment.conductivity
        Nu = 0.023 * Re**0.8 * Pr**0.4
    else:
        Nu = 3.66

    h = Nu * segment.conductivity / D
    A_s = np.pi * D

    # ODE
    def dT_dx(x, T):
        T_wall = wall_temp_model.profile(x)
        return (h * A_s / (m_dot * c_p)) * (T_wall - T)

    # Integration
    T = T_in
    x = 0.0

    for i in range(steps):
        if integrator == "euler":
            T = T + dT_dx(x, T) * dx
        elif integrator == "rk2":
            k1 = dT_dx(x, T)
            k2 = dT_dx(x + dx/2, T + k1*dx/2)
            T = T + k2 * dx
        elif integrator == "rk4":
            k1 = dT_dx(x, T)
            k2 = dT_dx(x + dx/2, T + k1*dx/2)
            k3 = dT_dx(x + dx/2, T + k2*dx/2)
            k4 = dT_dx(x + dx, T + k3*dx)
            T = T + (k1 + 2*k2 + 2*k3 + k4) * dx / 6.0

        x += dx
        temperatures[i + 1] = T

    return ThermalProfile(temperatures, positions)


@operator(
    domain="thermal_ode",
    category=OpCategory.INTEGRATE,
    signature="(mass: float, c_p: float, heat_input: float, h_conv: float, A_surface: float, T_amb: float, T_initial: float, time: float, dt: float) -> np.ndarray",
    deterministic=True,
    doc="Lumped thermal capacity model for transient heating"
)
def lumped_capacity(
    mass: float,
    c_p: float,
    heat_input: float,
    h_conv: float,
    A_surface: float,
    T_amb: float,
    T_initial: float,
    time: float,
    dt: float
) -> np.ndarray:
    """Lumped thermal capacity model for transient heating.

    Governing equation:
        m·c_p·dT/dt = Q_in - h·A·(T - T_amb)

    Args:
        mass: Body mass (kg)
        c_p: Specific heat capacity (J/(kg·K))
        heat_input: Heat input power (W)
        h_conv: Convective heat transfer coefficient (W/(m²·K))
        A_surface: Surface area (m²)
        T_amb: Ambient temperature (K)
        T_initial: Initial temperature (K)
        time: Total simulation time (s)
        dt: Time step (s)

    Returns:
        temperatures: Array of temperatures over time (K)

    Determinism: repro
    """
    n_steps = int(time / dt)
    temperatures = np.zeros(n_steps + 1)
    temperatures[0] = T_initial

    T = T_initial

    for i in range(n_steps):
        # Heat balance
        Q_in = heat_input
        Q_out = h_conv * A_surface * (T - T_amb)

        # Temperature change
        dT_dt = (Q_in - Q_out) / (mass * c_p)
        T = T + dT_dt * dt

        temperatures[i + 1] = T

    return temperatures


# ============================================================================
# Domain Registration
# ============================================================================

class ThermalODEOperations:
    """Thermal ODE domain operations."""

    @staticmethod
    def wall_temp_model(T_hot, T_amb, length, model="linear"):
        return wall_temp_model(T_hot, T_amb, length, model)

    @staticmethod
    def heat_transfer_1D(segment, m_dot, T_in, wall_temp_model, integrator="rk4", steps=100):
        return heat_transfer_1D(segment, m_dot, T_in, wall_temp_model, integrator, steps)

    @staticmethod
    def heat_transfer_1D_profile(segment, m_dot, T_in, wall_temp_model, integrator="rk4", steps=100):
        return heat_transfer_1D_profile(segment, m_dot, T_in, wall_temp_model, integrator, steps)

    @staticmethod
    def lumped_capacity(mass, c_p, heat_input, h_conv, A_surface, T_amb, T_initial, time, dt):
        return lumped_capacity(mass, c_p, heat_input, h_conv, A_surface, T_amb, T_initial, time, dt)


# Create domain instance
thermal_ode = ThermalODEOperations()


__all__ = [
    'ThermalSegment', 'WallTempModel', 'ThermalProfile',
    'wall_temp_model', 'heat_transfer_1D', 'heat_transfer_1D_profile', 'lumped_capacity',
    'thermal_ode', 'ThermalODEOperations'
]
