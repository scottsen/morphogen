"""ElectrochemDomain - Electrochemistry and battery simulation.

This module implements electrochemical kinetics, battery models, fuel cells,
electrolysis, and corrosion. Essential for energy storage, electrochemical
reactors, and corrosion protection design.

Specification: docs/specifications/chemistry.md
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
from morphogen.core.operator import operator, OpCategory


# ============================================================================
# Constants
# ============================================================================

FARADAY = 96485.3329  # C/mol - Faraday constant
R_GAS = 8.314  # J/(mol·K) - Universal gas constant


# ============================================================================
# Core Types
# ============================================================================

class ElectrodeType(Enum):
    """Electrode types."""
    ANODE = "anode"
    CATHODE = "cathode"


class BatteryType(Enum):
    """Battery chemistry types."""
    LITHIUM_ION = "lithium_ion"
    LEAD_ACID = "lead_acid"
    NICKEL_METAL_HYDRIDE = "nimh"
    SODIUM_ION = "sodium_ion"


@dataclass
class ElectrochemicalReaction:
    """Electrochemical half-reaction."""
    oxidized_species: str
    reduced_species: str
    n_electrons: int  # Electrons transferred
    E_standard: float  # Standard potential (V)
    alpha: float = 0.5  # Transfer coefficient


@dataclass
class BatteryCell:
    """Battery cell specification."""
    chemistry: BatteryType
    capacity: float  # mAh
    voltage_nominal: float  # V
    voltage_min: float  # V
    voltage_max: float  # V
    internal_resistance: float = 0.01  # Ohm
    soc: float = 1.0  # State of charge (0-1)


# ============================================================================
# Electrochemical Kinetics Operators
# ============================================================================

@operator(
    domain="electrochem",
    category=OpCategory.QUERY,
    signature="(overpotential: float, i0: float, alpha: float, n: int, temp: float) -> float",
    deterministic=True,
    doc="Compute current density from Butler-Volmer equation"
)
def butler_volmer(
    overpotential: float,
    i0: float,
    alpha: float = 0.5,
    n: int = 1,
    temp: float = 298.15
) -> float:
    """Compute current density from Butler-Volmer equation.

    i = i0 · [exp(α·n·F·η/(R·T)) - exp(-(1-α)·n·F·η/(R·T))]

    Args:
        overpotential: Overpotential (V), η = E - E_eq
        i0: Exchange current density (A/m²)
        alpha: Transfer coefficient (0-1)
        n: Number of electrons transferred
        temp: Temperature (K)

    Returns:
        current_density: Current density (A/m²)

    Determinism: strict
    """
    # Anodic and cathodic terms
    beta_a = alpha * n * FARADAY / (R_GAS * temp)
    beta_c = (1.0 - alpha) * n * FARADAY / (R_GAS * temp)

    i = i0 * (np.exp(beta_a * overpotential) - np.exp(-beta_c * overpotential))

    return i


@operator(
    domain="electrochem",
    category=OpCategory.QUERY,
    signature="(overpotential: float, i0: float, alpha: float, n: int, temp: float, anodic: bool) -> float",
    deterministic=True,
    doc="Compute current density from Tafel equation (high overpotential limit)"
)
def tafel_equation(
    overpotential: float,
    i0: float,
    alpha: float = 0.5,
    n: int = 1,
    temp: float = 298.15,
    anodic: bool = True
) -> float:
    """Compute current density from Tafel equation (high overpotential limit).

    η = a + b·log(i)

    where:
    - Tafel slope b = 2.303·R·T / (α·n·F)  [anodic]
    - Tafel slope b = 2.303·R·T / ((1-α)·n·F)  [cathodic]

    Args:
        overpotential: Overpotential (V)
        i0: Exchange current density (A/m²)
        alpha: Transfer coefficient
        n: Number of electrons
        temp: Temperature (K)
        anodic: True for anodic, False for cathodic

    Returns:
        current_density: Current density (A/m²)

    Determinism: strict
    """
    if anodic:
        beta = alpha * n * FARADAY / (R_GAS * temp)
    else:
        beta = (1.0 - alpha) * n * FARADAY / (R_GAS * temp)

    i = i0 * np.exp(beta * overpotential)

    return i


@operator(
    domain="electrochem",
    category=OpCategory.QUERY,
    signature="(E_standard: float, conc_oxidized: float, conc_reduced: float, n: int, temp: float) -> float",
    deterministic=True,
    doc="Compute electrode potential from Nernst equation"
)
def nernst(
    E_standard: float,
    conc_oxidized: float,
    conc_reduced: float,
    n: int,
    temp: float = 298.15
) -> float:
    """Compute electrode potential from Nernst equation.

    E = E° + (R·T / (n·F)) · ln(c_ox / c_red)

    Args:
        E_standard: Standard electrode potential (V)
        conc_oxidized: Concentration of oxidized species (mol/L)
        conc_reduced: Concentration of reduced species (mol/L)
        n: Number of electrons transferred
        temp: Temperature (K)

    Returns:
        E: Electrode potential (V)

    Determinism: strict
    """
    E = E_standard + (R_GAS * temp / (n * FARADAY)) * np.log(conc_oxidized / (conc_reduced + 1e-10))

    return E


@operator(
    domain="electrochem",
    category=OpCategory.QUERY,
    signature="(n: int, diffusivity: float, concentration: float, boundary_layer_thickness: float, area: float) -> float",
    deterministic=True,
    doc="Compute mass-transfer-limited current"
)
def limiting_current(
    n: int,
    diffusivity: float,
    concentration: float,
    boundary_layer_thickness: float,
    area: float = 1.0
) -> float:
    """Compute mass-transfer-limited current.

    i_lim = n·F·D·c / δ

    Args:
        n: Number of electrons
        diffusivity: Diffusion coefficient (m²/s)
        concentration: Bulk concentration (mol/m³)
        boundary_layer_thickness: Boundary layer thickness (m)
        area: Electrode area (m²)

    Returns:
        i_lim: Limiting current (A)

    Determinism: strict
    """
    i_lim = n * FARADAY * diffusivity * concentration * area / boundary_layer_thickness

    return i_lim


# ============================================================================
# Battery Simulation Operators
# ============================================================================

@operator(
    domain="electrochem",
    category=OpCategory.QUERY,
    signature="(capacity: float, current: float, time: float, voltage_nominal: float, internal_resistance: float, model: str) -> Tuple[float, float]",
    deterministic=True,
    doc="Simulate battery discharge"
)
def battery_discharge(
    capacity: float,
    current: float,
    time: float,
    voltage_nominal: float = 3.7,
    internal_resistance: float = 0.05,
    model: str = "equivalent_circuit"
) -> Tuple[float, float]:
    """Simulate battery discharge.

    Args:
        capacity: Battery capacity (mAh)
        current: Discharge current (A)
        time: Discharge time (s)
        voltage_nominal: Nominal voltage (V)
        internal_resistance: Internal resistance (Ohm)
        model: Battery model type

    Returns:
        voltage: Terminal voltage (V)
        soc: State of charge (0-1)

    Determinism: strict
    """
    # Convert capacity to Ah
    capacity_Ah = capacity / 1000.0

    # Charge removed
    charge_removed = current * time / 3600.0  # Ah

    # State of charge
    soc = 1.0 - charge_removed / capacity_Ah
    soc = np.clip(soc, 0.0, 1.0)

    if model == "equivalent_circuit":
        # Simple model: V = V_nominal - I·R_internal - f(SOC)

        # Voltage drop from internal resistance
        V_drop_resistance = current * internal_resistance

        # Voltage variation with SOC (simplified)
        # Voltage decreases nonlinearly as SOC drops
        if soc > 0.2:
            V_soc = voltage_nominal * (0.95 + 0.05 * soc)
        else:
            # Sharp drop at low SOC
            V_soc = voltage_nominal * (0.5 + 2.25 * soc)

        voltage = V_soc - V_drop_resistance

    else:
        # Default: linear model
        voltage = voltage_nominal * soc - current * internal_resistance

    voltage = max(voltage, 0.0)

    return voltage, soc


@operator(
    domain="electrochem",
    category=OpCategory.QUERY,
    signature="(capacity: float, current: float, time: float, soc_initial: float, voltage_nominal: float, voltage_max: float, internal_resistance: float) -> Tuple[float, float]",
    deterministic=True,
    doc="Simulate battery charging"
)
def battery_charge(
    capacity: float,
    current: float,
    time: float,
    soc_initial: float,
    voltage_nominal: float = 3.7,
    voltage_max: float = 4.2,
    internal_resistance: float = 0.05
) -> Tuple[float, float]:
    """Simulate battery charging.

    Args:
        capacity: Battery capacity (mAh)
        current: Charge current (A)
        time: Charge time (s)
        soc_initial: Initial state of charge (0-1)
        voltage_nominal: Nominal voltage (V)
        voltage_max: Maximum voltage (V)
        internal_resistance: Internal resistance (Ohm)

    Returns:
        voltage: Terminal voltage (V)
        soc: State of charge (0-1)

    Determinism: strict
    """
    # Convert capacity
    capacity_Ah = capacity / 1000.0

    # Charge added
    charge_added = current * time / 3600.0  # Ah

    # State of charge
    soc = soc_initial + charge_added / capacity_Ah
    soc = np.clip(soc, 0.0, 1.0)

    # Voltage during charging
    V_soc = voltage_nominal + (voltage_max - voltage_nominal) * soc
    voltage = V_soc + current * internal_resistance  # Charging increases voltage

    voltage = min(voltage, voltage_max)

    return voltage, soc


@operator(
    domain="electrochem",
    category=OpCategory.QUERY,
    signature="(n_cycles: int, depth_of_discharge: float, temp: float, chemistry: str) -> float",
    deterministic=True,
    doc="Estimate battery capacity fade after cycling"
)
def battery_cycle_life(
    n_cycles: int,
    depth_of_discharge: float,
    temp: float = 298.15,
    chemistry: str = "lithium_ion"
) -> float:
    """Estimate battery capacity fade after cycling.

    Args:
        n_cycles: Number of charge-discharge cycles
        depth_of_discharge: Depth of discharge (0-1)
        temp: Temperature (K)
        chemistry: Battery chemistry

    Returns:
        capacity_retention: Remaining capacity fraction (0-1)

    Determinism: strict
    """
    # Simplified empirical model
    # Capacity fade = a · N^b · DOD^c · exp(E_a/(R·T))

    if chemistry == "lithium_ion":
        a = 0.0001
        b = 0.5
        c = 1.5
        E_a = 20000  # J/mol
    elif chemistry == "lead_acid":
        a = 0.0005
        b = 0.6
        c = 2.0
        E_a = 25000
    else:
        a = 0.0001
        b = 0.5
        c = 1.5
        E_a = 20000

    fade = a * n_cycles**b * depth_of_discharge**c * np.exp(E_a / (R_GAS * temp))

    capacity_retention = 1.0 - fade
    capacity_retention = np.clip(capacity_retention, 0.0, 1.0)

    return capacity_retention


# ============================================================================
# Fuel Cell Operators
# ============================================================================

@operator(
    domain="electrochem",
    category=OpCategory.QUERY,
    signature="(current_density: float, temp: float, P_H2: float, P_O2: float, cell_type: str) -> float",
    deterministic=True,
    doc="Compute fuel cell voltage"
)
def fuel_cell_voltage(
    current_density: float,
    temp: float = 353.15,
    P_H2: float = 101325.0,
    P_O2: float = 21278.0,
    cell_type: str = "pemfc"
) -> float:
    """Compute fuel cell voltage.

    V = E_nernst - η_act - η_ohmic - η_conc

    Args:
        current_density: Current density (A/m²)
        temp: Temperature (K)
        P_H2: Hydrogen partial pressure (Pa)
        P_O2: Oxygen partial pressure (Pa)
        cell_type: Fuel cell type (pemfc, sofc, etc.)

    Returns:
        voltage: Cell voltage (V)

    Determinism: strict
    """
    # Nernst voltage
    E0 = 1.229  # Standard potential at 298K (V)

    # Temperature correction
    E_nernst = E0 - 0.00085 * (temp - 298.15)

    # Pressure correction (simplified Nernst)
    E_nernst += (R_GAS * temp / (2 * FARADAY)) * np.log(P_H2 * np.sqrt(P_O2) / 101325.0)

    # Activation overpotential (Tafel)
    i0 = 1e-3  # Exchange current density (A/m²)
    alpha = 0.5
    eta_act = (R_GAS * temp / (alpha * 2 * FARADAY)) * np.log((current_density + 1e-6) / i0)

    # Ohmic overpotential
    if cell_type == "pemfc":
        resistance = 0.0001  # Ohm·m²
    elif cell_type == "sofc":
        resistance = 0.0002
    else:
        resistance = 0.0001

    eta_ohmic = current_density * resistance

    # Concentration overpotential (simplified)
    i_lim = 20000.0  # Limiting current density (A/m²)
    if current_density < i_lim * 0.9:
        eta_conc = (R_GAS * temp / (2 * FARADAY)) * np.log(1 - current_density / i_lim)
    else:
        eta_conc = 0.5  # Large overpotential near limit

    # Total voltage
    voltage = E_nernst - eta_act - eta_ohmic + eta_conc  # Note: eta_conc is negative

    voltage = max(voltage, 0.0)

    return voltage


@operator(
    domain="electrochem",
    category=OpCategory.QUERY,
    signature="(voltage: float, temp: float) -> float",
    deterministic=True,
    doc="Compute fuel cell efficiency"
)
def fuel_cell_efficiency(
    voltage: float,
    temp: float = 353.15
) -> float:
    """Compute fuel cell efficiency.

    η = V / ΔG° (electrical)

    Args:
        voltage: Operating voltage (V)
        temp: Temperature (K)

    Returns:
        efficiency: Efficiency (0-1)

    Determinism: strict
    """
    # Thermodynamic limit (Gibbs free energy)
    # ΔG = ΔH - T·ΔS for H2 + 0.5·O2 -> H2O

    delta_H = -285830  # J/mol (enthalpy of formation, liquid water)
    delta_S = -163.2  # J/(mol·K) (entropy change)

    delta_G = delta_H - temp * delta_S

    # Voltage equivalent of Gibbs free energy
    V_thermo = -delta_G / (2 * FARADAY)  # 2 electrons

    efficiency = voltage / V_thermo

    return efficiency


# ============================================================================
# Electrolysis Operators
# ============================================================================

@operator(
    domain="electrochem",
    category=OpCategory.QUERY,
    signature="(current_density: float, temp: float, pressure: float) -> float",
    deterministic=True,
    doc="Compute voltage required for water electrolysis"
)
def water_electrolysis_voltage(
    current_density: float,
    temp: float = 298.15,
    pressure: float = 101325.0
) -> float:
    """Compute voltage required for water electrolysis.

    Args:
        current_density: Current density (A/m²)
        temp: Temperature (K)
        pressure: Pressure (Pa)

    Returns:
        voltage: Cell voltage (V)

    Determinism: strict
    """
    # Thermodynamic voltage
    V_thermo = 1.229  # V at 298K

    # Temperature correction
    V_thermo -= 0.00085 * (temp - 298.15)

    # Overpotentials
    # Anode (O2 evolution)
    i0_anode = 1e-6  # A/m²
    alpha_anode = 0.5
    eta_anode = (R_GAS * temp / (alpha_anode * 4 * FARADAY)) * np.log(current_density / i0_anode)

    # Cathode (H2 evolution)
    i0_cathode = 1e-3  # A/m²
    alpha_cathode = 0.5
    eta_cathode = (R_GAS * temp / (alpha_cathode * 2 * FARADAY)) * np.log(current_density / i0_cathode)

    # Ohmic resistance
    resistance = 0.0002  # Ohm·m²
    eta_ohmic = current_density * resistance

    # Total voltage
    voltage = V_thermo + eta_anode + eta_cathode + eta_ohmic

    return voltage


@operator(
    domain="electrochem",
    category=OpCategory.QUERY,
    signature="(charge_passed: float, moles_produced: float, n_electrons: int) -> float",
    deterministic=True,
    doc="Compute Faradaic efficiency"
)
def faraday_efficiency(
    charge_passed: float,
    moles_produced: float,
    n_electrons: int
) -> float:
    """Compute Faradaic efficiency.

    η_F = (n·F·moles_actual) / charge_passed

    Args:
        charge_passed: Total charge (C)
        moles_produced: Moles of product formed (mol)
        n_electrons: Electrons per molecule

    Returns:
        efficiency: Faradaic efficiency (0-1)

    Determinism: strict
    """
    charge_theoretical = n_electrons * FARADAY * moles_produced

    efficiency = charge_theoretical / (charge_passed + 1e-10)

    return efficiency


# ============================================================================
# Corrosion Operators
# ============================================================================

@operator(
    domain="electrochem",
    category=OpCategory.QUERY,
    signature="(E_corr: float, E_standard_anode: float, E_standard_cathode: float, i0_anode: float, i0_cathode: float, alpha: float) -> float",
    deterministic=True,
    doc="Compute corrosion current from mixed potential theory"
)
def corrosion_current(
    E_corr: float,
    E_standard_anode: float,
    E_standard_cathode: float,
    i0_anode: float,
    i0_cathode: float,
    alpha: float = 0.5
) -> float:
    """Compute corrosion current from mixed potential theory.

    Args:
        E_corr: Corrosion potential (V)
        E_standard_anode: Standard potential of anodic reaction (V)
        E_standard_cathode: Standard potential of cathodic reaction (V)
        i0_anode: Exchange current density for anode (A/m²)
        i0_cathode: Exchange current density for cathode (A/m²)
        alpha: Transfer coefficient

    Returns:
        i_corr: Corrosion current density (A/m²)

    Determinism: strict
    """
    # Anodic overpotential
    eta_anode = E_corr - E_standard_anode
    i_anode = i0_anode * np.exp(alpha * FARADAY * eta_anode / (R_GAS * 298.15))

    # Cathodic overpotential
    eta_cathode = E_corr - E_standard_cathode
    i_cathode = i0_cathode * np.exp(-(1 - alpha) * FARADAY * eta_cathode / (R_GAS * 298.15))

    # At corrosion potential: i_anode = i_cathode
    # Simplified: use geometric mean
    i_corr = np.sqrt(i_anode * i_cathode)

    return i_corr


@operator(
    domain="electrochem",
    category=OpCategory.QUERY,
    signature="(i_corr: float, equivalent_weight: float, density: float, n: int) -> float",
    deterministic=True,
    doc="Convert corrosion current to corrosion rate"
)
def corrosion_rate(
    i_corr: float,
    equivalent_weight: float,
    density: float,
    n: int = 2
) -> float:
    """Convert corrosion current to corrosion rate.

    Rate (mm/year) = i_corr · EW / (n·F·ρ) · 3.16e7 · 10

    Args:
        i_corr: Corrosion current density (A/m²)
        equivalent_weight: Equivalent weight (g/mol)
        density: Material density (g/cm³)
        n: Electrons transferred

    Returns:
        rate: Corrosion rate (mm/year)

    Determinism: strict
    """
    # Convert to g/(m²·s)
    mass_loss_rate = i_corr * equivalent_weight / (n * FARADAY)

    # Convert to mm/year
    # density in g/cm³ = 1e6 g/m³
    # 1 year = 3.156e7 s
    rate = mass_loss_rate / (density * 1e6) * 1e3 * 3.156e7

    return rate


# ============================================================================
# Domain Registration
# ============================================================================

class ElectrochemOperations:
    """Electrochemistry domain operations."""

    # Electrochemical Kinetics
    butler_volmer = staticmethod(butler_volmer)
    tafel_equation = staticmethod(tafel_equation)
    nernst = staticmethod(nernst)
    limiting_current = staticmethod(limiting_current)

    # Battery Simulation
    battery_discharge = staticmethod(battery_discharge)
    battery_charge = staticmethod(battery_charge)
    battery_cycle_life = staticmethod(battery_cycle_life)

    # Fuel Cells
    fuel_cell_voltage = staticmethod(fuel_cell_voltage)
    fuel_cell_efficiency = staticmethod(fuel_cell_efficiency)

    # Electrolysis
    water_electrolysis_voltage = staticmethod(water_electrolysis_voltage)
    faraday_efficiency = staticmethod(faraday_efficiency)

    # Corrosion
    corrosion_current = staticmethod(corrosion_current)
    corrosion_rate = staticmethod(corrosion_rate)


# Create domain instance
electrochem = ElectrochemOperations()


__all__ = [
    'ElectrodeType', 'BatteryType', 'ElectrochemicalReaction', 'BatteryCell',
    'butler_volmer', 'tafel_equation', 'nernst', 'limiting_current',
    'battery_discharge', 'battery_charge', 'battery_cycle_life',
    'fuel_cell_voltage', 'fuel_cell_efficiency',
    'water_electrolysis_voltage', 'faraday_efficiency',
    'corrosion_current', 'corrosion_rate',
    'electrochem', 'ElectrochemOperations'
]
