"""MultiphaseDomain - Vapor-liquid equilibrium and multiphase systems.

This module implements vapor-liquid equilibrium calculations, flash calculations,
gas-liquid reactions, and multiphase transport. Essential for distillation,
absorption, and multiphase reactor design.

Specification: docs/specifications/chemistry.md
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
from enum import Enum


# ============================================================================
# Constants
# ============================================================================

R_GAS = 8.314  # J/(mol·K)


# ============================================================================
# Core Types
# ============================================================================

class ThermoModel(Enum):
    """Thermodynamic models for VLE."""
    IDEAL = "ideal"
    PENG_ROBINSON = "peng_robinson"
    SRK = "srk"  # Soave-Redlich-Kwong
    RAOULT = "raoult"


@dataclass
class Component:
    """Chemical component properties."""
    name: str
    critical_temp: float  # K
    critical_pressure: float  # Pa
    acentric_factor: float
    molecular_weight: float  # kg/mol
    antoine_A: float = 0.0  # Antoine coefficients for vapor pressure
    antoine_B: float = 0.0
    antoine_C: float = 0.0


# ============================================================================
# Vapor Pressure
# ============================================================================

def antoine_equation(
    temp: float,
    A: float,
    B: float,
    C: float
) -> float:
    """Compute vapor pressure using Antoine equation.

    log10(P) = A - B / (T + C)

    Args:
        temp: Temperature (K)
        A, B, C: Antoine coefficients

    Returns:
        P_sat: Saturation pressure (Pa)

    Determinism: strict
    """
    # Convert temperature to Celsius for typical Antoine coefficients
    T_celsius = temp - 273.15
    log_P = A - B / (T_celsius + C)
    P_sat = 10**log_P  # Typically in mmHg or bar, need conversion

    # Assume coefficients give mmHg, convert to Pa
    P_sat_Pa = P_sat * 133.322

    return P_sat_Pa


# ============================================================================
# Vapor-Liquid Equilibrium Operators
# ============================================================================

def vle_flash(
    feed_composition: List[float],
    temp: float,
    pressure: float,
    components: Optional[List[Component]] = None,
    thermo_model: str = "ideal"
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Perform isothermal flash calculation.

    Solves: z_i = x_i · (1 - V) + y_i · V
            y_i = K_i · x_i
            sum(x_i) = 1
            sum(y_i) = 1

    Args:
        feed_composition: Overall mole fractions
        temp: Temperature (K)
        pressure: Pressure (Pa)
        components: List of components (optional)
        thermo_model: Thermodynamic model

    Returns:
        y_vapor: Vapor phase mole fractions
        x_liquid: Liquid phase mole fractions
        V: Vapor fraction

    Determinism: repro (iterative solver)
    """
    z = np.array(feed_composition)
    n_components = len(z)

    # Compute K-values (vapor-liquid equilibrium ratios)
    if thermo_model == "ideal":
        # Ideal: K_i = P_sat_i / P
        # Use simplified vapor pressures
        K = np.ones(n_components)
        for i in range(n_components):
            # Rough estimate: more volatile components have K > 1
            K[i] = 1.0 + 0.5 * (i - n_components/2) / n_components

    elif thermo_model == "raoult":
        # Raoult's law: K_i = P_sat_i / P
        if components:
            K = np.zeros(n_components)
            for i, comp in enumerate(components):
                P_sat = antoine_equation(temp, comp.antoine_A, comp.antoine_B, comp.antoine_C)
                K[i] = P_sat / pressure
        else:
            K = np.ones(n_components)

    else:
        # Peng-Robinson or SRK - more complex
        # Placeholder: use ideal approximation
        K = np.ones(n_components)

    # Rachford-Rice equation to solve for vapor fraction V
    # sum_i [z_i * (K_i - 1) / (1 + V * (K_i - 1))] = 0

    def rachford_rice(V):
        return np.sum(z * (K - 1) / (1 + V * (K - 1)))

    # Bounds for V
    V_min = 1.0 / (1.0 - np.max(K))
    V_max = 1.0 / (1.0 - np.min(K))
    V_min = max(0.0, V_min)
    V_max = min(1.0, V_max)

    # Bisection method
    V = 0.5
    for _ in range(100):
        f_V = rachford_rice(V)

        if abs(f_V) < 1e-8:
            break

        # Update V
        if f_V > 0:
            V_min = V
        else:
            V_max = V

        V = (V_min + V_max) / 2

    # Compute phase compositions
    x_liquid = z / (1 + V * (K - 1))
    y_vapor = K * x_liquid

    # Normalize
    x_liquid /= np.sum(x_liquid)
    y_vapor /= np.sum(y_vapor)

    return y_vapor, x_liquid, V


def bubble_point(
    liquid_composition: List[float],
    pressure: float,
    components: Optional[List[Component]] = None,
    thermo_model: str = "raoult"
) -> float:
    """Compute bubble point temperature.

    At bubble point: sum(K_i · x_i) = 1

    Args:
        liquid_composition: Liquid mole fractions
        pressure: Pressure (Pa)
        components: List of components
        thermo_model: Thermodynamic model

    Returns:
        temp_bubble: Bubble point temperature (K)

    Determinism: repro
    """
    x = np.array(liquid_composition)

    # Initial guess: average of pure component boiling points
    temp = 350.0  # K

    # Iteratively solve for temperature
    for iteration in range(100):
        # Compute K-values at current temperature
        if thermo_model == "raoult" and components:
            K = np.zeros(len(x))
            for i, comp in enumerate(components):
                P_sat = antoine_equation(temp, comp.antoine_A, comp.antoine_B, comp.antoine_C)
                K[i] = P_sat / pressure
        else:
            # Simplified
            K = np.ones(len(x))

        # Bubble point condition: sum(K_i · x_i) = 1
        sum_Kx = np.sum(K * x)

        if abs(sum_Kx - 1.0) < 1e-6:
            break

        # Update temperature (simple)
        if sum_Kx > 1.0:
            temp -= 1.0  # Lower temperature
        else:
            temp += 1.0  # Raise temperature

    return temp


def dew_point(
    vapor_composition: List[float],
    pressure: float,
    components: Optional[List[Component]] = None,
    thermo_model: str = "raoult"
) -> float:
    """Compute dew point temperature.

    At dew point: sum(y_i / K_i) = 1

    Args:
        vapor_composition: Vapor mole fractions
        pressure: Pressure (Pa)
        components: List of components
        thermo_model: Thermodynamic model

    Returns:
        temp_dew: Dew point temperature (K)

    Determinism: repro
    """
    y = np.array(vapor_composition)

    # Initial guess
    temp = 360.0  # K

    # Iteratively solve
    for iteration in range(100):
        # Compute K-values
        if thermo_model == "raoult" and components:
            K = np.zeros(len(y))
            for i, comp in enumerate(components):
                P_sat = antoine_equation(temp, comp.antoine_A, comp.antoine_B, comp.antoine_C)
                K[i] = P_sat / pressure
        else:
            K = np.ones(len(y))

        # Dew point condition: sum(y_i / K_i) = 1
        sum_y_over_K = np.sum(y / (K + 1e-10))

        if abs(sum_y_over_K - 1.0) < 1e-6:
            break

        # Update temperature
        if sum_y_over_K > 1.0:
            temp += 1.0
        else:
            temp -= 1.0

    return temp


# ============================================================================
# Gas-Liquid Reaction Operators
# ============================================================================

def volumetric_mass_transfer(
    bubble_diameter: float,
    gas_holdup: float,
    diffusivity: float,
    density_liquid: float = 1000.0,
    viscosity_liquid: float = 1e-3
) -> float:
    """Compute volumetric mass transfer coefficient k_L·a.

    Args:
        bubble_diameter: Bubble diameter (m)
        gas_holdup: Gas volume fraction
        diffusivity: Molecular diffusivity in liquid (m²/s)
        density_liquid: Liquid density (kg/m³)
        viscosity_liquid: Liquid viscosity (Pa·s)

    Returns:
        k_L_a: Volumetric mass transfer coefficient (1/s)

    Determinism: strict
    """
    # Specific interfacial area
    a = 6.0 * gas_holdup / bubble_diameter  # m²/m³

    # Mass transfer coefficient (Sherwood correlation)
    # Sh = k_L · d / D ≈ 2 + 0.6 · Re^0.5 · Sc^0.33

    # Estimate Re and Sc
    velocity = 0.2  # m/s (typical rise velocity)
    Re = density_liquid * velocity * bubble_diameter / viscosity_liquid
    Sc = viscosity_liquid / (density_liquid * diffusivity)

    Sh = 2.0 + 0.6 * Re**0.5 * Sc**(1/3)
    k_L = Sh * diffusivity / bubble_diameter

    k_L_a = k_L * a

    return k_L_a


def gas_liquid_reaction(
    conc_liquid: float,
    pressure_gas: float,
    k_L_a: float,
    k_rxn: float,
    henry_constant: float
) -> float:
    """Compute gas-liquid reaction rate.

    For reaction: A(g) -> A(l) -> Products

    Args:
        conc_liquid: Liquid phase concentration (mol/L)
        pressure_gas: Gas phase partial pressure (Pa)
        k_L_a: Volumetric mass transfer coefficient (1/s)
        k_rxn: Reaction rate constant (1/s)
        henry_constant: Henry's law constant (mol/(L·Pa))

    Returns:
        rate: Reaction rate (mol/(L·s))

    Determinism: strict
    """
    # Equilibrium concentration at interface (Henry's law)
    c_sat = henry_constant * pressure_gas  # mol/L

    # Two-film theory with reaction
    # Overall rate limited by mass transfer and reaction

    # Enhancement factor (for fast reactions)
    # Ha = sqrt(k_rxn * D) / k_L (Hatta number)
    # For simple case, assume pseudo-first-order

    # Effective rate
    rate = k_L_a * (c_sat - conc_liquid) + k_rxn * conc_liquid

    return rate


def gas_absorption(
    gas_flow: float,
    liquid_flow: float,
    gas_inlet_composition: float,
    pressure: float,
    temp: float,
    height: float,
    k_L_a: float,
    henry_constant: float
) -> Tuple[float, float]:
    """Simulate gas absorption column.

    Args:
        gas_flow: Gas flow rate (mol/s)
        liquid_flow: Liquid flow rate (L/s)
        gas_inlet_composition: Inlet gas mole fraction
        pressure: Pressure (Pa)
        temp: Temperature (K)
        height: Column height (m)
        k_L_a: Volumetric mass transfer coefficient (1/s)
        henry_constant: Henry's law constant (mol/(L·Pa))

    Returns:
        gas_outlet_composition: Outlet gas mole fraction
        liquid_outlet_conc: Outlet liquid concentration (mol/L)

    Determinism: strict
    """
    # Simplified: assume plug flow, constant flows

    # Discretize column
    n_stages = 20
    dz = height / n_stages

    y_gas = gas_inlet_composition  # Mole fraction
    c_liquid = 0.0  # mol/L

    for stage in range(n_stages):
        # Mass transfer driving force
        P_partial = y_gas * pressure
        c_sat = henry_constant * P_partial

        # Transfer rate per volume
        rate = k_L_a * (c_sat - c_liquid)  # mol/(L·s)

        # Update concentrations (simplified)
        delta_c = rate * dz / 1.0  # Assuming residence time
        c_liquid += delta_c
        y_gas -= delta_c * 0.01  # Decrease gas composition

        y_gas = max(0.0, y_gas)

    return y_gas, c_liquid


# ============================================================================
# Multiphase Flow
# ============================================================================

def two_phase_pressure_drop(
    gas_velocity: float,
    liquid_velocity: float,
    density_gas: float,
    density_liquid: float,
    viscosity_gas: float,
    viscosity_liquid: float,
    diameter: float,
    length: float,
    roughness: float = 1e-5
) -> float:
    """Compute pressure drop in two-phase flow.

    Uses Lockhart-Martinelli correlation.

    Args:
        gas_velocity: Superficial gas velocity (m/s)
        liquid_velocity: Superficial liquid velocity (m/s)
        density_gas: Gas density (kg/m³)
        density_liquid: Liquid density (kg/m³)
        viscosity_gas: Gas viscosity (Pa·s)
        viscosity_liquid: Liquid viscosity (Pa·s)
        diameter: Pipe diameter (m)
        length: Pipe length (m)
        roughness: Surface roughness (m)

    Returns:
        delta_P: Pressure drop (Pa)

    Determinism: strict
    """
    # Single-phase pressure drops
    Re_gas = density_gas * gas_velocity * diameter / viscosity_gas
    Re_liquid = density_liquid * liquid_velocity * diameter / viscosity_liquid

    # Friction factors (Darcy-Weisbach)
    if Re_gas < 2300:
        f_gas = 64 / Re_gas
    else:
        eps = roughness / diameter
        f_gas = 0.25 / (np.log10(eps / 3.7 + 5.74 / Re_gas**0.9))**2

    if Re_liquid < 2300:
        f_liquid = 64 / Re_liquid
    else:
        eps = roughness / diameter
        f_liquid = 0.25 / (np.log10(eps / 3.7 + 5.74 / Re_liquid**0.9))**2

    # Single-phase pressure drops
    dP_gas = f_gas * (length / diameter) * 0.5 * density_gas * gas_velocity**2
    dP_liquid = f_liquid * (length / diameter) * 0.5 * density_liquid * liquid_velocity**2

    # Lockhart-Martinelli parameter
    X = np.sqrt(dP_liquid / (dP_gas + 1e-10))

    # Two-phase multiplier
    if X < 1:
        phi_gas = 1 + 20/X + 1/X**2
        delta_P = phi_gas * dP_gas
    else:
        phi_liquid = 1 + 20*X + X**2
        delta_P = phi_liquid * dP_liquid

    return delta_P


# ============================================================================
# Domain Registration
# ============================================================================

class MultiphaseOperations:
    """Multiphase systems domain operations."""

    # Vapor-Liquid Equilibrium
    vle_flash = staticmethod(vle_flash)
    bubble_point = staticmethod(bubble_point)
    dew_point = staticmethod(dew_point)
    antoine_equation = staticmethod(antoine_equation)

    # Gas-Liquid Reactions
    volumetric_mass_transfer = staticmethod(volumetric_mass_transfer)
    gas_liquid_reaction = staticmethod(gas_liquid_reaction)
    gas_absorption = staticmethod(gas_absorption)

    # Multiphase Flow
    two_phase_pressure_drop = staticmethod(two_phase_pressure_drop)


# Create domain instance
multiphase = MultiphaseOperations()


__all__ = [
    'ThermoModel', 'Component',
    'antoine_equation', 'vle_flash', 'bubble_point', 'dew_point',
    'volumetric_mass_transfer', 'gas_liquid_reaction', 'gas_absorption',
    'two_phase_pressure_drop',
    'multiphase', 'MultiphaseOperations'
]
