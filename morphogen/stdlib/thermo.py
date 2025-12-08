"""ThermoDomain - Thermodynamic properties and equations of state.

This module implements equations of state (EOS), activity coefficient models,
and thermodynamic property calculations. Essential for phase equilibrium,
process simulation, and energy calculations.

Specification: docs/specifications/chemistry.md
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum

from morphogen.core.operator import operator, OpCategory


# ============================================================================
# Constants
# ============================================================================

R_GAS = 8.314  # J/(mol·K) - Universal gas constant


# ============================================================================
# Core Types
# ============================================================================

class EOSType(Enum):
    """Equation of state types."""
    IDEAL_GAS = "ideal_gas"
    VAN_DER_WAALS = "van_der_waals"
    PENG_ROBINSON = "peng_robinson"
    SRK = "srk"  # Soave-Redlich-Kwong
    REDLICH_KWONG = "redlich_kwong"


class ActivityModel(Enum):
    """Activity coefficient models."""
    IDEAL = "ideal"
    NRTL = "nrtl"
    UNIQUAC = "uniquac"
    UNIFAC = "unifac"
    WILSON = "wilson"


@dataclass
class ComponentData:
    """Thermodynamic component data."""
    name: str
    critical_temp: float  # K
    critical_pressure: float  # Pa
    critical_volume: float = 0.0  # m³/mol
    acentric_factor: float = 0.0
    molecular_weight: float = 0.0  # kg/mol

    # Heat capacity coefficients (Cp = A + B*T + C*T² + D*T³)
    Cp_A: float = 0.0  # J/(mol·K)
    Cp_B: float = 0.0
    Cp_C: float = 0.0
    Cp_D: float = 0.0

    # Standard thermodynamic properties
    Hf_298: float = 0.0  # Heat of formation at 298K (J/mol)
    Gf_298: float = 0.0  # Gibbs free energy of formation at 298K (J/mol)
    S_298: float = 0.0  # Entropy at 298K (J/(mol·K))


# ============================================================================
# Equations of State
# ============================================================================

@operator(
    domain="thermo",
    category=OpCategory.QUERY,
    signature="(temp: float, pressure: float, n_moles: float) -> float",
    deterministic=True,
    doc="Compute volume from ideal gas law"
)
def ideal_gas(
    temp: float,
    pressure: float,
    n_moles: float = 1.0
) -> float:
    """Compute volume from ideal gas law.

    PV = nRT

    Args:
        temp: Temperature (K)
        pressure: Pressure (Pa)
        n_moles: Number of moles

    Returns:
        volume: Volume (m³)

    Determinism: strict
    """
    volume = n_moles * R_GAS * temp / pressure
    return volume


@operator(
    domain="thermo",
    category=OpCategory.QUERY,
    signature="(temp: float, pressure: float, critical_temp: float, critical_pressure: float, acentric_factor: float) -> float",
    deterministic=True,
    doc="Solve Peng-Robinson equation of state for compressibility factor"
)
def peng_robinson(
    temp: float,
    pressure: float,
    critical_temp: float,
    critical_pressure: float,
    acentric_factor: float
) -> float:
    """Solve Peng-Robinson equation of state for compressibility factor.

    Z³ - (1-B)Z² + (A-3B²-2B)Z - (AB-B²-B³) = 0

    Args:
        temp: Temperature (K)
        pressure: Pressure (Pa)
        critical_temp: Critical temperature (K)
        critical_pressure: Critical pressure (Pa)
        acentric_factor: Acentric factor

    Returns:
        Z: Compressibility factor (PV/RT)

    Determinism: strict
    """
    T_r = temp / critical_temp

    # Parameters
    kappa = 0.37464 + 1.54226 * acentric_factor - 0.26992 * acentric_factor**2
    alpha = (1 + kappa * (1 - np.sqrt(T_r)))**2

    a = 0.45724 * (R_GAS * critical_temp)**2 / critical_pressure * alpha
    b = 0.07780 * R_GAS * critical_temp / critical_pressure

    A = a * pressure / (R_GAS * temp)**2
    B = b * pressure / (R_GAS * temp)

    # Cubic equation: Z³ + p*Z² + q*Z + r = 0
    p = -(1 - B)
    q = A - 3*B**2 - 2*B
    r = -(A*B - B**2 - B**3)

    # Solve cubic equation
    coeffs = [1, p, q, r]
    roots = np.roots(coeffs)

    # Take real root (vapor phase typically has largest Z)
    real_roots = roots[np.isreal(roots)].real
    if len(real_roots) > 0:
        Z = np.max(real_roots)
    else:
        Z = 1.0  # Fallback to ideal gas

    return Z


@operator(
    domain="thermo",
    category=OpCategory.QUERY,
    signature="(temp: float, pressure: float, critical_temp: float, critical_pressure: float, acentric_factor: float) -> float",
    deterministic=True,
    doc="Solve Soave-Redlich-Kwong EOS for compressibility factor"
)
def soave_redlich_kwong(
    temp: float,
    pressure: float,
    critical_temp: float,
    critical_pressure: float,
    acentric_factor: float
) -> float:
    """Solve Soave-Redlich-Kwong EOS for compressibility factor.

    Args:
        temp: Temperature (K)
        pressure: Pressure (Pa)
        critical_temp: Critical temperature (K)
        critical_pressure: Critical pressure (Pa)
        acentric_factor: Acentric factor

    Returns:
        Z: Compressibility factor

    Determinism: strict
    """
    T_r = temp / critical_temp

    # SRK parameters
    m = 0.480 + 1.574 * acentric_factor - 0.176 * acentric_factor**2
    alpha = (1 + m * (1 - np.sqrt(T_r)))**2

    a = 0.42748 * (R_GAS * critical_temp)**2 / critical_pressure * alpha
    b = 0.08664 * R_GAS * critical_temp / critical_pressure

    A = a * pressure / (R_GAS * temp)**2
    B = b * pressure / (R_GAS * temp)

    # Cubic equation
    coeffs = [1, -1, A - B - B**2, -A*B]
    roots = np.roots(coeffs)

    real_roots = roots[np.isreal(roots)].real
    if len(real_roots) > 0:
        Z = np.max(real_roots)
    else:
        Z = 1.0

    return Z


@operator(
    domain="thermo",
    category=OpCategory.QUERY,
    signature="(temp: float, pressure: float, critical_temp: float, critical_pressure: float, acentric_factor: float, eos: str) -> float",
    deterministic=True,
    doc="Compute fugacity coefficient from EOS"
)
def fugacity_coefficient(
    temp: float,
    pressure: float,
    critical_temp: float,
    critical_pressure: float,
    acentric_factor: float,
    eos: str = "peng_robinson"
) -> float:
    """Compute fugacity coefficient from EOS.

    φ = f / P

    Args:
        temp: Temperature (K)
        pressure: Pressure (Pa)
        critical_temp: Critical temperature (K)
        critical_pressure: Critical pressure (Pa)
        acentric_factor: Acentric factor
        eos: Equation of state

    Returns:
        phi: Fugacity coefficient

    Determinism: strict
    """
    if eos == "peng_robinson":
        Z = peng_robinson(temp, pressure, critical_temp, critical_pressure, acentric_factor)

        # Simplified fugacity coefficient (full calculation is more complex)
        phi = np.exp(Z - 1 - np.log(Z))

    else:
        phi = 1.0  # Ideal gas

    return phi


# ============================================================================
# Activity Coefficients
# ============================================================================

@operator(
    domain="thermo",
    category=OpCategory.QUERY,
    signature="(composition: List[float], temp: float, model: str, parameters: Optional[Dict]) -> np.ndarray",
    deterministic=True,
    doc="Compute activity coefficients for liquid mixture"
)
def activity_coefficient(
    composition: List[float],
    temp: float,
    model: str = "nrtl",
    parameters: Optional[Dict] = None
) -> np.ndarray:
    """Compute activity coefficients for liquid mixture.

    Args:
        composition: Mole fractions
        temp: Temperature (K)
        model: Activity model (nrtl, uniquac, unifac, wilson)
        parameters: Model parameters

    Returns:
        gamma: Activity coefficients

    Determinism: strict
    """
    x = np.array(composition)
    n = len(x)

    if model == "ideal":
        return np.ones(n)

    elif model == "nrtl":
        # NRTL (Non-Random Two-Liquid) model
        # Simplified two-component implementation

        if parameters is None or n != 2:
            # Default parameters
            tau_12 = 0.3
            tau_21 = -0.3
            alpha = 0.3
        else:
            tau_12 = parameters.get('tau_12', 0.3)
            tau_21 = parameters.get('tau_21', -0.3)
            alpha = parameters.get('alpha', 0.3)

        G_12 = np.exp(-alpha * tau_12)
        G_21 = np.exp(-alpha * tau_21)

        ln_gamma_1 = x[1]**2 * (tau_21 * (G_21 / (x[0] + x[1]*G_21))**2 +
                                 tau_12 * G_12 / (x[1] + x[0]*G_12)**2)

        ln_gamma_2 = x[0]**2 * (tau_12 * (G_12 / (x[1] + x[0]*G_12))**2 +
                                 tau_21 * G_21 / (x[0] + x[1]*G_21)**2)

        gamma = np.array([np.exp(ln_gamma_1), np.exp(ln_gamma_2)])

    elif model == "wilson":
        # Wilson model
        if parameters is None or n != 2:
            Lambda_12 = 0.5
            Lambda_21 = 0.5
        else:
            Lambda_12 = parameters.get('Lambda_12', 0.5)
            Lambda_21 = parameters.get('Lambda_21', 0.5)

        ln_gamma_1 = -np.log(x[0] + Lambda_12 * x[1]) + x[1] * (
            Lambda_12 / (x[0] + Lambda_12 * x[1]) - Lambda_21 / (x[1] + Lambda_21 * x[0])
        )

        ln_gamma_2 = -np.log(x[1] + Lambda_21 * x[0]) - x[0] * (
            Lambda_12 / (x[0] + Lambda_12 * x[1]) - Lambda_21 / (x[1] + Lambda_21 * x[0])
        )

        gamma = np.array([np.exp(ln_gamma_1), np.exp(ln_gamma_2)])

    else:
        # Default to ideal
        gamma = np.ones(n)

    return gamma


# ============================================================================
# Thermodynamic Properties
# ============================================================================

@operator(
    domain="thermo",
    category=OpCategory.QUERY,
    signature="(species: str, temp: float, component_data: Optional[ComponentData]) -> float",
    deterministic=True,
    doc="Compute heat capacity at constant pressure"
)
def heat_capacity(
    species: str,
    temp: float,
    component_data: Optional[ComponentData] = None
) -> float:
    """Compute heat capacity at constant pressure.

    Cp = A + B*T + C*T² + D*T³

    Args:
        species: Species name
        temp: Temperature (K)
        component_data: Component thermodynamic data

    Returns:
        Cp: Heat capacity (J/(mol·K))

    Determinism: strict
    """
    if component_data is None:
        # Default values for common species
        defaults = {
            'water': (75.327, 0.0, 0.0, 0.0),
            'ethanol': (112.3, 0.0, 0.0, 0.0),
            'air': (29.1, 0.0, 0.0, 0.0),
            'nitrogen': (29.1, 0.0, 0.0, 0.0),
            'oxygen': (29.4, 0.0, 0.0, 0.0),
        }

        if species.lower() in defaults:
            A, B, C, D = defaults[species.lower()]
        else:
            A, B, C, D = 30.0, 0.0, 0.0, 0.0  # Generic
    else:
        A, B, C, D = component_data.Cp_A, component_data.Cp_B, component_data.Cp_C, component_data.Cp_D

    Cp = A + B * temp + C * temp**2 + D * temp**3

    return Cp


@operator(
    domain="thermo",
    category=OpCategory.QUERY,
    signature="(reactants: Dict[str, float], products: Dict[str, float], temp: float, component_data: Optional[Dict[str, ComponentData]]) -> float",
    deterministic=True,
    doc="Compute enthalpy of reaction"
)
def enthalpy_of_reaction(
    reactants: Dict[str, float],
    products: Dict[str, float],
    temp: float = 298.15,
    component_data: Optional[Dict[str, ComponentData]] = None
) -> float:
    """Compute enthalpy of reaction.

    ΔH_rxn = Σ(ν_i · ΔH_f,i) for products - reactants

    Args:
        reactants: Reactant stoichiometry {species: coefficient}
        products: Product stoichiometry
        temp: Temperature (K)
        component_data: Component thermodynamic data

    Returns:
        delta_H: Enthalpy of reaction (J/mol)

    Determinism: strict
    """
    # Standard heats of formation (J/mol) at 298K
    Hf_data = {
        'H2O': -285830.0,
        'CO2': -393509.0,
        'CH4': -74520.0,
        'NH3': -45940.0,
        'H2': 0.0,
        'O2': 0.0,
        'N2': 0.0,
    }

    delta_H = 0.0

    # Products
    for species, coeff in products.items():
        if component_data and species in component_data:
            Hf = component_data[species].Hf_298
        else:
            Hf = Hf_data.get(species, 0.0)
        delta_H += coeff * Hf

    # Reactants
    for species, coeff in reactants.items():
        if component_data and species in component_data:
            Hf = component_data[species].Hf_298
        else:
            Hf = Hf_data.get(species, 0.0)
        delta_H -= coeff * Hf

    # Temperature correction (simplified - assumes constant Cp)
    if temp != 298.15:
        delta_Cp = 0.0  # Would need to compute from Cp data
        delta_H += delta_Cp * (temp - 298.15)

    return delta_H


@operator(
    domain="thermo",
    category=OpCategory.QUERY,
    signature="(temp: float, enthalpy: float, entropy: float) -> float",
    deterministic=True,
    doc="Compute Gibbs free energy"
)
def gibbs_free_energy(
    temp: float,
    enthalpy: float,
    entropy: float
) -> float:
    """Compute Gibbs free energy.

    G = H - T·S

    Args:
        temp: Temperature (K)
        enthalpy: Enthalpy (J/mol)
        entropy: Entropy (J/(mol·K))

    Returns:
        G: Gibbs free energy (J/mol)

    Determinism: strict
    """
    G = enthalpy - temp * entropy
    return G


@operator(
    domain="thermo",
    category=OpCategory.QUERY,
    signature="(delta_G: float, temp: float) -> float",
    deterministic=True,
    doc="Compute equilibrium constant from Gibbs free energy"
)
def equilibrium_constant(
    delta_G: float,
    temp: float
) -> float:
    """Compute equilibrium constant from Gibbs free energy.

    K = exp(-ΔG / (R·T))

    Args:
        delta_G: Gibbs free energy change (J/mol)
        temp: Temperature (K)

    Returns:
        K_eq: Equilibrium constant

    Determinism: strict
    """
    K_eq = np.exp(-delta_G / (R_GAS * temp))
    return K_eq


# ============================================================================
# Mixture Properties
# ============================================================================

@operator(
    domain="thermo",
    category=OpCategory.QUERY,
    signature="(composition: List[float], pure_enthalpies: List[float], excess_enthalpy: float) -> float",
    deterministic=True,
    doc="Compute enthalpy of mixing"
)
def mixing_enthalpy(
    composition: List[float],
    pure_enthalpies: List[float],
    excess_enthalpy: float = 0.0
) -> float:
    """Compute enthalpy of mixing.

    H_mix = Σ(x_i · H_i) + H_excess

    Args:
        composition: Mole fractions
        pure_enthalpies: Pure component enthalpies (J/mol)
        excess_enthalpy: Excess enthalpy (J/mol)

    Returns:
        H_mix: Mixture enthalpy (J/mol)

    Determinism: strict
    """
    x = np.array(composition)
    H = np.array(pure_enthalpies)

    H_mix = np.sum(x * H) + excess_enthalpy

    return H_mix


@operator(
    domain="thermo",
    category=OpCategory.QUERY,
    signature="(composition: List[float], pure_entropies: List[float]) -> float",
    deterministic=True,
    doc="Compute entropy of mixing (ideal solution)"
)
def mixing_entropy(
    composition: List[float],
    pure_entropies: List[float]
) -> float:
    """Compute entropy of mixing (ideal solution).

    S_mix = Σ(x_i · S_i) - R·Σ(x_i · ln(x_i))

    Args:
        composition: Mole fractions
        pure_entropies: Pure component entropies (J/(mol·K))

    Returns:
        S_mix: Mixture entropy (J/(mol·K))

    Determinism: strict
    """
    x = np.array(composition)
    S = np.array(pure_entropies)

    S_ideal = np.sum(x * S)

    # Entropy of mixing (ideal)
    S_mixing = -R_GAS * np.sum(x * np.log(x + 1e-10))

    S_mix = S_ideal + S_mixing

    return S_mix


# ============================================================================
# Phase Equilibrium
# ============================================================================

@operator(
    domain="thermo",
    category=OpCategory.QUERY,
    signature="(composition_liquid: List[float], vapor_pressures: List[float], pressure: float) -> List[float]",
    deterministic=True,
    doc="Compute vapor composition from Raoult's law"
)
def raoult_law(
    composition_liquid: List[float],
    vapor_pressures: List[float],
    pressure: float
) -> List[float]:
    """Compute vapor composition from Raoult's law.

    y_i = (x_i · P_sat_i) / P

    Args:
        composition_liquid: Liquid mole fractions
        vapor_pressures: Pure component vapor pressures (Pa)
        pressure: Total pressure (Pa)

    Returns:
        composition_vapor: Vapor mole fractions

    Determinism: strict
    """
    x = np.array(composition_liquid)
    P_sat = np.array(vapor_pressures)

    y = x * P_sat / pressure

    # Normalize
    y /= np.sum(y)

    return y.tolist()


# ============================================================================
# Domain Registration
# ============================================================================

class ThermoOperations:
    """Thermodynamics domain operations."""

    # Equations of State
    ideal_gas = staticmethod(ideal_gas)
    peng_robinson = staticmethod(peng_robinson)
    soave_redlich_kwong = staticmethod(soave_redlich_kwong)
    fugacity_coefficient = staticmethod(fugacity_coefficient)

    # Activity Coefficients
    activity_coefficient = staticmethod(activity_coefficient)

    # Thermodynamic Properties
    heat_capacity = staticmethod(heat_capacity)
    enthalpy_of_reaction = staticmethod(enthalpy_of_reaction)
    gibbs_free_energy = staticmethod(gibbs_free_energy)
    equilibrium_constant = staticmethod(equilibrium_constant)

    # Mixture Properties
    mixing_enthalpy = staticmethod(mixing_enthalpy)
    mixing_entropy = staticmethod(mixing_entropy)

    # Phase Equilibrium
    raoult_law = staticmethod(raoult_law)


# Create domain instance
thermo = ThermoOperations()


__all__ = [
    'EOSType', 'ActivityModel', 'ComponentData',
    'ideal_gas', 'peng_robinson', 'soave_redlich_kwong', 'fugacity_coefficient',
    'activity_coefficient',
    'heat_capacity', 'enthalpy_of_reaction', 'gibbs_free_energy', 'equilibrium_constant',
    'mixing_enthalpy', 'mixing_entropy',
    'raoult_law',
    'thermo', 'ThermoOperations'
]
