"""CatalysisDomain - Heterogeneous catalysis and surface kinetics.

This module implements surface reaction kinetics, catalyst characterization,
and adsorption isotherms. Essential for catalytic reactor design and catalyst
development.

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

R_GAS = 8.314  # J/(mol·K)
N_A = 6.02214076e23  # Avogadro's number


# ============================================================================
# Core Types
# ============================================================================

class MechanismType(Enum):
    """Surface reaction mechanisms."""
    LANGMUIR_HINSHELWOOD = "langmuir_hinshelwood"
    ELEY_RIDEAL = "eley_rideal"
    MARS_VAN_KREVELEN = "mars_van_krevelen"


@dataclass
class SurfaceSpecies:
    """Surface-adsorbed species."""
    name: str
    coverage: float  # Fractional coverage (0-1)
    binding_energy: float  # J/mol
    sites_occupied: int = 1  # Number of active sites


@dataclass
class AdsorptionIsotherm:
    """Adsorption isotherm data."""
    pressures: np.ndarray  # Pa
    amounts_adsorbed: np.ndarray  # mol/kg or mol/m²
    temp: float  # K


# ============================================================================
# Surface Kinetics Operators
# ============================================================================

@operator(
    domain="catalysis",
    category=OpCategory.QUERY,
    signature="(coverage_A: float, coverage_B: float, k_surface: float, activation_energy: float, temp: float) -> float",
    deterministic=True,
    doc="Compute reaction rate for Langmuir-Hinshelwood mechanism"
)
def langmuir_hinshelwood(
    coverage_A: float,
    coverage_B: float,
    k_surface: float,
    activation_energy: float = 0.0,
    temp: float = 300.0
) -> float:
    """Compute reaction rate for Langmuir-Hinshelwood mechanism.

    In L-H mechanism, both reactants adsorb on surface before reacting:
    A(g) + * → A*
    B(g) + * → B*
    A* + B* → Products + 2*

    Rate = k · θ_A · θ_B

    Args:
        coverage_A: Surface coverage of species A (0-1)
        coverage_B: Surface coverage of species B (0-1)
        k_surface: Surface reaction rate constant (mol/(m²·s))
        activation_energy: Activation energy (J/mol)
        temp: Temperature (K)

    Returns:
        rate: Reaction rate per unit area (mol/(m²·s))

    Determinism: strict
    """
    # Temperature dependence
    if activation_energy > 0:
        k_eff = k_surface * np.exp(-activation_energy / (R_GAS * temp))
    else:
        k_eff = k_surface

    rate = k_eff * coverage_A * coverage_B

    return rate


@operator(
    domain="catalysis",
    category=OpCategory.QUERY,
    signature="(coverage_A: float, pressure_B: float, k_surface: float, activation_energy: float, temp: float) -> float",
    deterministic=True,
    doc="Compute reaction rate for Eley-Rideal mechanism"
)
def eley_rideal(
    coverage_A: float,
    pressure_B: float,
    k_surface: float,
    activation_energy: float = 0.0,
    temp: float = 300.0
) -> float:
    """Compute reaction rate for Eley-Rideal mechanism.

    In E-R mechanism, one reactant adsorbs while other reacts from gas phase:
    A(g) + * → A*
    A* + B(g) → Products + *

    Rate = k · θ_A · P_B

    Args:
        coverage_A: Surface coverage of species A (0-1)
        pressure_B: Partial pressure of gas-phase species B (Pa)
        k_surface: Surface reaction rate constant (mol/(m²·s·Pa))
        activation_energy: Activation energy (J/mol)
        temp: Temperature (K)

    Returns:
        rate: Reaction rate per unit area (mol/(m²·s))

    Determinism: strict
    """
    # Temperature dependence
    if activation_energy > 0:
        k_eff = k_surface * np.exp(-activation_energy / (R_GAS * temp))
    else:
        k_eff = k_surface

    rate = k_eff * coverage_A * pressure_B

    return rate


@operator(
    domain="catalysis",
    category=OpCategory.INTEGRATE,
    signature="(coverage: np.ndarray, r_adsorption: float, r_desorption: float, r_reaction: float, dt: float) -> np.ndarray",
    deterministic=True,
    doc="Update surface coverage with adsorption, desorption, and reaction"
)
def surface_coverage_step(
    coverage: np.ndarray,
    r_adsorption: float,
    r_desorption: float,
    r_reaction: float,
    dt: float
) -> np.ndarray:
    """Update surface coverage with adsorption, desorption, and reaction.

    dθ/dt = r_ads - r_des - r_rxn

    Args:
        coverage: Current surface coverage (0-1)
        r_adsorption: Adsorption rate (1/s)
        r_desorption: Desorption rate (1/s)
        r_reaction: Reaction rate (1/s)
        dt: Time step (s)

    Returns:
        coverage_new: Updated surface coverage

    Determinism: strict
    """
    # Available sites
    vacant_sites = 1.0 - np.sum(coverage)

    # Rate of change
    dcoverage_dt = r_adsorption * vacant_sites - r_desorption * coverage - r_reaction * coverage

    # Update
    coverage_new = coverage + dcoverage_dt * dt

    # Enforce bounds
    coverage_new = np.clip(coverage_new, 0.0, 1.0)

    return coverage_new


@operator(
    domain="catalysis",
    category=OpCategory.QUERY,
    signature="(pressure: float, K_ads: float, n_sites: int) -> float",
    deterministic=True,
    doc="Compute surface coverage from Langmuir isotherm"
)
def langmuir_adsorption(
    pressure: float,
    K_ads: float,
    n_sites: int = 1
) -> float:
    """Compute surface coverage from Langmuir isotherm.

    θ = (K · P) / (1 + K · P)  for single-site adsorption

    Args:
        pressure: Gas pressure (Pa)
        K_ads: Adsorption equilibrium constant (1/Pa)
        n_sites: Number of sites occupied per molecule

    Returns:
        coverage: Fractional surface coverage (0-1)

    Determinism: strict
    """
    if n_sites == 1:
        coverage = (K_ads * pressure) / (1.0 + K_ads * pressure)
    else:
        # Multi-site adsorption (more complex)
        coverage = (K_ads * pressure)**(1/n_sites) / (1.0 + (K_ads * pressure)**(1/n_sites))

    return coverage


@operator(
    domain="catalysis",
    category=OpCategory.QUERY,
    signature="(pressures: List[float], K_ads_values: List[float]) -> np.ndarray",
    deterministic=True,
    doc="Compute competitive adsorption of multiple species (Langmuir)"
)
def competitive_adsorption(
    pressures: List[float],
    K_ads_values: List[float]
) -> np.ndarray:
    """Compute competitive adsorption of multiple species (Langmuir).

    θ_i = (K_i · P_i) / (1 + Σ(K_j · P_j))

    Args:
        pressures: Partial pressures (Pa)
        K_ads_values: Adsorption constants (1/Pa)

    Returns:
        coverages: Fractional coverages for each species

    Determinism: strict
    """
    P = np.array(pressures)
    K = np.array(K_ads_values)

    denominator = 1.0 + np.sum(K * P)
    coverages = (K * P) / denominator

    return coverages


# ============================================================================
# Catalyst Characterization Operators
# ============================================================================

@operator(
    domain="catalysis",
    category=OpCategory.QUERY,
    signature="(adsorption_isotherm: AdsorptionIsotherm, adsorbate: str, cross_sectional_area: Optional[float]) -> float",
    deterministic=True,
    doc="Compute BET surface area from adsorption isotherm"
)
def bet_surface_area(
    adsorption_isotherm: AdsorptionIsotherm,
    adsorbate: str = "N2",
    cross_sectional_area: Optional[float] = None
) -> float:
    """Compute BET surface area from adsorption isotherm.

    BET equation: P / [n(P0-P)] = 1/(n_m·C) + (C-1)/(n_m·C) · (P/P0)

    Args:
        adsorption_isotherm: Adsorption data (P vs n)
        adsorbate: Adsorbate molecule (N2, Ar, etc.)
        cross_sectional_area: Molecular cross-sectional area (m²)

    Returns:
        surface_area: BET surface area (m²/g)

    Determinism: strict
    """
    # Default cross-sectional areas (m²/molecule)
    cross_sections = {
        'N2': 0.162e-18,
        'Ar': 0.142e-18,
        'Kr': 0.152e-18,
    }

    if cross_sectional_area is None:
        cross_sectional_area = cross_sections.get(adsorbate, 0.16e-18)

    P = adsorption_isotherm.pressures
    n = adsorption_isotherm.amounts_adsorbed
    P0 = 101325.0  # Saturation pressure (Pa) - simplified

    # BET plot: y = P / [n(P0-P)] vs x = P/P0
    x = P / P0
    y = P / (n * (P0 - P))

    # Linear region typically 0.05 < P/P0 < 0.3
    mask = (x > 0.05) & (x < 0.3)
    if np.sum(mask) < 2:
        mask = np.ones_like(x, dtype=bool)

    # Linear fit
    coeffs = np.polyfit(x[mask], y[mask], 1)
    slope = coeffs[0]
    intercept = coeffs[1]

    # BET parameters
    n_m = 1.0 / (slope + intercept)  # Monolayer capacity (mol/kg)

    # Surface area
    surface_area = n_m * N_A * cross_sectional_area  # m²/kg

    # Convert to m²/g
    surface_area /= 1000.0

    return surface_area


@operator(
    domain="catalysis",
    category=OpCategory.QUERY,
    signature="(adsorption_isotherm: AdsorptionIsotherm, method: str) -> Tuple[np.ndarray, np.ndarray]",
    deterministic=True,
    doc="Compute pore size distribution from adsorption isotherm"
)
def pore_size_distribution(
    adsorption_isotherm: AdsorptionIsotherm,
    method: str = "bjh"
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute pore size distribution from adsorption isotherm.

    Args:
        adsorption_isotherm: Adsorption data
        method: Analysis method (bjh, dft)

    Returns:
        pore_diameters: Pore diameters (nm)
        pore_volumes: Differential pore volume (cm³/(g·nm))

    Determinism: strict
    """
    P = adsorption_isotherm.pressures
    n = adsorption_isotherm.amounts_adsorbed

    if method == "bjh":
        # Barrett-Joyner-Halenda method
        # Simplified implementation

        # Kelvin equation for pore radius
        # ln(P/P0) = -2·γ·V_m / (r·R·T)

        gamma = 8.85e-3  # Surface tension of N2 (N/m)
        V_m = 34.7e-6  # Molar volume of liquid N2 (m³/mol)
        temp = adsorption_isotherm.temp
        P0 = 101325.0

        # Kelvin radius
        r_kelvin = -2 * gamma * V_m / (R_GAS * temp * np.log(P / P0 + 1e-10))

        # Pore diameter (including adsorbed layer thickness)
        t_layer = 0.354  # nm (typical N2 layer thickness)
        pore_diameters = 2 * (r_kelvin * 1e9 + t_layer)  # nm

        # Differential pore volume
        dn = np.diff(n, prepend=0)
        pore_volumes = np.abs(dn) / np.diff(pore_diameters, prepend=pore_diameters[0] - 0.1)

    elif method == "dft":
        # Density Functional Theory method (simplified)
        # More accurate but requires complex kernels
        pore_diameters = np.linspace(1, 50, 50)  # nm
        pore_volumes = np.exp(-((pore_diameters - 10) / 5)**2)  # Placeholder Gaussian

    else:
        # Default
        pore_diameters = np.linspace(1, 50, 50)
        pore_volumes = np.zeros_like(pore_diameters)

    return pore_diameters, pore_volumes


@operator(
    domain="catalysis",
    category=OpCategory.QUERY,
    signature="(reaction_rate: float, n_active_sites: float) -> float",
    deterministic=True,
    doc="Compute turnover frequency (TOF)"
)
def turnover_frequency(
    reaction_rate: float,
    n_active_sites: float
) -> float:
    """Compute turnover frequency (TOF).

    TOF = rate / number of active sites

    Args:
        reaction_rate: Reaction rate (mol/s)
        n_active_sites: Number of active sites (mol)

    Returns:
        tof: Turnover frequency (1/s)

    Determinism: strict
    """
    tof = reaction_rate / (n_active_sites + 1e-20)
    return tof


@operator(
    domain="catalysis",
    category=OpCategory.QUERY,
    signature="(rate_desired: float, rate_total: float) -> float",
    deterministic=True,
    doc="Compute catalyst selectivity"
)
def catalyst_selectivity(
    rate_desired: float,
    rate_total: float
) -> float:
    """Compute catalyst selectivity.

    Selectivity = rate_desired / rate_total

    Args:
        rate_desired: Rate of desired product formation
        rate_total: Total reaction rate

    Returns:
        selectivity: Selectivity (0-1)

    Determinism: strict
    """
    selectivity = rate_desired / (rate_total + 1e-20)
    return selectivity


@operator(
    domain="catalysis",
    category=OpCategory.QUERY,
    signature="(activity_initial: float, time: float, deactivation_rate: float, order: int) -> float",
    deterministic=True,
    doc="Model catalyst deactivation over time"
)
def catalyst_deactivation(
    activity_initial: float,
    time: float,
    deactivation_rate: float,
    order: int = 1
) -> float:
    """Model catalyst deactivation over time.

    First-order: a(t) = a0 · exp(-k_d · t)
    Second-order: a(t) = a0 / (1 + a0·k_d·t)

    Args:
        activity_initial: Initial activity
        time: Time (s)
        deactivation_rate: Deactivation rate constant
        order: Deactivation order (1 or 2)

    Returns:
        activity: Current activity

    Determinism: strict
    """
    if order == 1:
        activity = activity_initial * np.exp(-deactivation_rate * time)
    elif order == 2:
        activity = activity_initial / (1.0 + activity_initial * deactivation_rate * time)
    else:
        activity = activity_initial

    return activity


# ============================================================================
# Microkinetic Modeling
# ============================================================================

@operator(
    domain="catalysis",
    category=OpCategory.QUERY,
    signature="(rate_constants: Dict[str, float], partial_pressures: Dict[str, float], n_sites_total: float, max_iterations: int, tolerance: float) -> Dict[str, float]",
    deterministic=True,
    doc="Solve microkinetic model for steady-state surface coverages"
)
def microkinetic_steady_state(
    rate_constants: Dict[str, float],
    partial_pressures: Dict[str, float],
    n_sites_total: float = 1.0,
    max_iterations: int = 1000,
    tolerance: float = 1e-8
) -> Dict[str, float]:
    """Solve microkinetic model for steady-state surface coverages.

    Args:
        rate_constants: Forward and reverse rate constants
        partial_pressures: Gas-phase partial pressures
        n_sites_total: Total site density
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance

    Returns:
        coverages: Steady-state surface coverages

    Determinism: repro (iterative)
    """
    # Simplified example: A adsorption/desorption
    # A + * <-> A*

    k_ads = rate_constants.get('k_ads', 1e-5)
    k_des = rate_constants.get('k_des', 1e-3)
    P_A = partial_pressures.get('A', 1000.0)

    # Steady state: k_ads·P_A·θ_vacant = k_des·θ_A
    # θ_vacant + θ_A = 1

    theta_A = (k_ads * P_A) / (k_ads * P_A + k_des)
    theta_vacant = 1.0 - theta_A

    coverages = {
        'A': theta_A,
        'vacant': theta_vacant
    }

    return coverages


# ============================================================================
# Domain Registration
# ============================================================================

class CatalysisOperations:
    """Catalysis domain operations."""

    # Surface Kinetics
    langmuir_hinshelwood = staticmethod(langmuir_hinshelwood)
    eley_rideal = staticmethod(eley_rideal)
    surface_coverage_step = staticmethod(surface_coverage_step)
    langmuir_adsorption = staticmethod(langmuir_adsorption)
    competitive_adsorption = staticmethod(competitive_adsorption)

    # Catalyst Characterization
    bet_surface_area = staticmethod(bet_surface_area)
    pore_size_distribution = staticmethod(pore_size_distribution)
    turnover_frequency = staticmethod(turnover_frequency)
    catalyst_selectivity = staticmethod(catalyst_selectivity)
    catalyst_deactivation = staticmethod(catalyst_deactivation)

    # Microkinetic Modeling
    microkinetic_steady_state = staticmethod(microkinetic_steady_state)


# Create domain instance
catalysis = CatalysisOperations()


__all__ = [
    'MechanismType', 'SurfaceSpecies', 'AdsorptionIsotherm',
    'langmuir_hinshelwood', 'eley_rideal', 'surface_coverage_step',
    'langmuir_adsorption', 'competitive_adsorption',
    'bet_surface_area', 'pore_size_distribution',
    'turnover_frequency', 'catalyst_selectivity', 'catalyst_deactivation',
    'microkinetic_steady_state',
    'catalysis', 'CatalysisOperations'
]
