"""CombustionLightDomain - Simplified combustion metrics (no detailed chemistry).

This module implements approximate combustion quality metrics using equivalence ratio,
temperature, and mixing. Essential for fire pits, burners, mufflers, engine exhaust.

Specification: docs/specifications/physics-domains.md
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from morphogen.core.operator import operator, OpCategory


# ============================================================================
# Core Types
# ============================================================================

@dataclass
class MixtureState:
    """Mixture composition."""
    fuel_rate: float  # kg/s
    air_rate: float  # kg/s

    @property
    def equivalence_ratio(self) -> float:
        """Equivalence ratio φ = (F/A) / (F/A)_stoich."""
        return equivalence_ratio(self.fuel_rate, self.air_rate)


@dataclass
class CombustionZone:
    """Combustion zone properties."""
    temperature: float  # K
    pressure: float  # Pa
    residence_time: float  # s


@dataclass
class SmokeIndex:
    """Smoke/emissions index."""
    value: float  # 0 = very smoky, 1 = clean
    reduction_factor: float  # Relative to baseline


# ============================================================================
# Operators
# ============================================================================

@operator(
    domain="combustion",
    category=OpCategory.QUERY,
    signature="(fuel_rate: float, air_rate: float, stoichiometric_ratio: float) -> float",
    deterministic=True,
    doc="Compute equivalence ratio (φ = actual/stoichiometric)"
)
def equivalence_ratio(
    fuel_rate: float,
    air_rate: float,
    stoichiometric_ratio: float = 0.0676
) -> float:
    """Compute equivalence ratio (φ = actual/stoichiometric).

    φ < 1: lean (excess air)
    φ = 1: stoichiometric
    φ > 1: rich (excess fuel)

    Args:
        fuel_rate: Fuel mass flow rate (kg/s)
        air_rate: Air mass flow rate (kg/s)
        stoichiometric_ratio: (F/A)_stoich, default 0.0676 for gasoline

    Returns:
        phi: Equivalence ratio (dimensionless)

    Determinism: strict
    """
    if air_rate <= 0:
        return float('inf')

    FA_actual = fuel_rate / air_rate
    phi = FA_actual / stoichiometric_ratio

    return phi


@operator(
    domain="combustion",
    category=OpCategory.QUERY,
    signature="(phi: float, T_reactants: float, fuel_type: str) -> float",
    deterministic=True,
    doc="Estimate adiabatic flame temperature"
)
def adiabatic_flame_temperature(
    phi: float,
    T_reactants: float,
    fuel_type: str = "wood"
) -> float:
    """Estimate adiabatic flame temperature.

    Args:
        phi: Equivalence ratio
        T_reactants: Reactant temperature (K)
        fuel_type: "wood", "propane", "methane", "gasoline"

    Returns:
        T_flame: Adiabatic flame temperature (K)

    Determinism: repro
    """
    # Peak temperatures at φ ≈ 1.05 (slightly rich)
    peak_temps = {
        "wood": 1500.0,
        "propane": 2200.0,
        "methane": 2200.0,
        "gasoline": 2300.0
    }

    T_peak = peak_temps.get(fuel_type, 1500.0)

    # Temperature decreases for lean (φ < 1) or rich (φ > 1.1) mixtures
    if phi < 1.0:
        # Lean: temperature drops due to excess air
        T_flame = T_reactants + (T_peak - T_reactants) * phi
    elif phi < 1.1:
        # Near stoichiometric: maximum temperature
        T_flame = T_peak
    else:
        # Rich: temperature drops due to incomplete combustion
        T_flame = T_peak - (phi - 1.1) * 500.0

    # Clamp to reasonable range
    T_flame = max(T_reactants, min(T_flame, T_peak))

    return T_flame


@operator(
    domain="combustion",
    category=OpCategory.QUERY,
    signature="(T_flame: float, secondary_air_flow: float, secondary_air_temp: float, primary_flow: float, mixing_factor: float, model: str) -> float",
    deterministic=True,
    doc="Estimate combustion zone temperature with secondary air"
)
def zone_temperature(
    T_flame: float,
    secondary_air_flow: float,
    secondary_air_temp: float,
    primary_flow: float,
    mixing_factor: float,
    model: str = "energy_balance"
) -> float:
    """Estimate combustion zone temperature with secondary air.

    Args:
        T_flame: Adiabatic flame temperature (K)
        secondary_air_flow: Secondary air mass flow (kg/s)
        secondary_air_temp: Secondary air temperature (K)
        primary_flow: Primary combustion flow (kg/s)
        mixing_factor: Jet mixing effectiveness (0-1)
        model: "energy_balance" or "empirical"

    Returns:
        T_zone: Combustion zone temperature (K)

    Determinism: repro
    """
    if model == "energy_balance":
        # Energy balance: mix hot gases with secondary air
        if primary_flow + secondary_air_flow * mixing_factor > 0:
            T_zone = (
                T_flame * primary_flow +
                secondary_air_temp * secondary_air_flow * mixing_factor
            ) / (primary_flow + secondary_air_flow * mixing_factor)
        else:
            T_zone = T_flame

    elif model == "empirical":
        # Empirical: temperature drops with secondary air injection
        # ΔT proportional to mixing effectiveness
        delta_T = (T_flame - secondary_air_temp) * mixing_factor * 0.3
        T_zone = T_flame - delta_T

    else:
        raise ValueError(f"Unknown model: {model}")

    return T_zone


@operator(
    domain="combustion",
    category=OpCategory.QUERY,
    signature="(phi: float, T_zone: float, mixing_factor: float, residence_time: float, model: str) -> SmokeIndex",
    deterministic=True,
    doc="Estimate smoke reduction effectiveness from secondary air"
)
def smoke_reduction(
    phi: float,
    T_zone: float,
    mixing_factor: float,
    residence_time: float,
    model: str = "empirical"
) -> SmokeIndex:
    """Estimate smoke reduction effectiveness from secondary air.

    Key factors:
    - φ → 1 (better): Approaching stoichiometric
    - T_zone ↑ (better): Higher temperature burns more soot
    - mixing_factor ↑ (better): Better air-fuel mixing
    - residence_time ↑ (better): More time to burn

    Args:
        phi: Equivalence ratio
        T_zone: Combustion zone temperature (K)
        mixing_factor: Jet mixing effectiveness (0-1)
        residence_time: Time in hot zone (s)
        model: "empirical" or "kinetic"

    Returns:
        SmokeIndex with value (0-1) and reduction factor

    Determinism: repro
    """
    if model == "empirical":
        # Empirical correlation based on literature

        # Equivalence ratio factor (best at φ = 1.0)
        phi_factor = np.exp(-2.0 * (phi - 1.0)**2)

        # Temperature factor (normalized to 1200K)
        temp_factor = min(1.0, (T_zone / 1200.0)**2)

        # Mixing factor (already 0-1)
        mix_factor = mixing_factor

        # Residence time factor (saturates at ~1s)
        residence_factor = 1.0 - np.exp(-residence_time / 0.5)

        # Combined smoke index
        smoke_index = phi_factor * temp_factor * mix_factor * residence_factor

        # Reduction factor (compared to baseline without secondary air)
        baseline_index = 0.2  # Typical for primary combustion only
        reduction_factor = smoke_index / baseline_index

    elif model == "kinetic":
        # Approximate kinetic model (simplified Arrhenius)
        # Soot oxidation rate ~ exp(-Ea / (R·T))

        R = 8.314  # J/(mol·K)
        Ea = 100000.0  # Activation energy (J/mol) - typical for soot oxidation

        # Oxidation rate constant
        k = np.exp(-Ea / (R * T_zone)) * mixing_factor

        # Soot burnout (exponential decay with residence time)
        burnout = 1.0 - np.exp(-k * residence_time)

        # Equivalence ratio correction
        phi_correction = min(1.0, 2.0 / phi) if phi > 1.0 else 1.0

        smoke_index = burnout * phi_correction

        baseline_index = 0.2
        reduction_factor = smoke_index / baseline_index

    else:
        raise ValueError(f"Unknown model: {model}")

    return SmokeIndex(value=smoke_index, reduction_factor=reduction_factor)


@operator(
    domain="combustion",
    category=OpCategory.QUERY,
    signature="(phi: float, T_zone: float) -> float",
    deterministic=True,
    doc="Estimate combustion efficiency"
)
def combustion_efficiency(
    phi: float,
    T_zone: float
) -> float:
    """Estimate combustion efficiency.

    Args:
        phi: Equivalence ratio
        T_zone: Combustion zone temperature (K)

    Returns:
        efficiency: Combustion efficiency (0-1)

    Determinism: repro
    """
    # Maximum efficiency near stoichiometric
    phi_eff = 1.0 - abs(phi - 1.0) * 0.5
    phi_eff = max(0.0, min(1.0, phi_eff))

    # Temperature effect (efficiency increases with temperature)
    temp_eff = min(1.0, T_zone / 1500.0)

    efficiency = phi_eff * temp_eff

    return efficiency


@operator(
    domain="combustion",
    category=OpCategory.QUERY,
    signature="(phi: float, T_zone: float, species: str) -> float",
    deterministic=True,
    doc="Estimate emissions index for specific species"
)
def emissions_index(
    phi: float,
    T_zone: float,
    species: str = "CO"
) -> float:
    """Estimate emissions index for specific species.

    Args:
        phi: Equivalence ratio
        T_zone: Combustion zone temperature (K)
        species: "CO", "NOx", "PM" (particulate matter)

    Returns:
        EI: Emissions index (g pollutant / kg fuel)

    Determinism: repro
    """
    if species == "CO":
        # CO increases for rich mixtures and low temperatures
        if phi < 1.0:
            EI_CO = 10.0 * (1.0 - phi)  # Lean quenching
        else:
            EI_CO = 50.0 * (phi - 1.0)  # Rich combustion

        # Temperature correction
        EI_CO *= max(0.1, 1.0 - (T_zone - 1000.0) / 1000.0)

    elif species == "NOx":
        # NOx increases with temperature (thermal NOx)
        if T_zone > 1500.0:
            EI_NOx = 0.01 * (T_zone - 1500.0)**2 / 1000.0
        else:
            EI_NOx = 0.0

    elif species == "PM":
        # Particulate matter increases for rich mixtures
        if phi > 1.0:
            EI_PM = 20.0 * (phi - 1.0)
        else:
            EI_PM = 1.0

        # Temperature correction (higher temp reduces PM)
        EI_PM *= max(0.1, 1.0 - (T_zone - 800.0) / 1000.0)

    else:
        raise ValueError(f"Unknown species: {species}")

    return max(0.0, globals()[f"EI_{species}"])


# ============================================================================
# Convenience Functions
# ============================================================================

@operator(
    domain="combustion",
    category=OpCategory.QUERY,
    signature="(fuel_rate: float, primary_air_rate: float, secondary_air_rate: float, secondary_air_temp: float, mixing_factor: float, residence_time: float) -> dict",
    deterministic=True,
    doc="Analyze fire pit combustion with secondary air injection"
)
def analyze_fire_pit_combustion(
    fuel_rate: float,
    primary_air_rate: float,
    secondary_air_rate: float,
    secondary_air_temp: float,
    mixing_factor: float,
    residence_time: float
) -> dict:
    """Analyze fire pit combustion with secondary air injection.

    Args:
        fuel_rate: Fuel (wood) burn rate (kg/s)
        primary_air_rate: Primary air from below (kg/s)
        secondary_air_rate: Secondary air from J-tubes (kg/s)
        secondary_air_temp: Secondary air temperature (K)
        mixing_factor: Jet mixing effectiveness (0-1)
        residence_time: Residence time in hot zone (s)

    Returns:
        dict with combustion metrics
    """
    # Primary combustion
    phi_primary = equivalence_ratio(fuel_rate, primary_air_rate)
    T_flame = adiabatic_flame_temperature(phi_primary, 293.0, "wood")

    # Zone temperature with secondary air
    primary_flow = fuel_rate + primary_air_rate
    T_zone = zone_temperature(
        T_flame, secondary_air_rate, secondary_air_temp,
        primary_flow, mixing_factor
    )

    # Overall equivalence ratio (with secondary air)
    total_air = primary_air_rate + secondary_air_rate * mixing_factor
    phi_overall = equivalence_ratio(fuel_rate, total_air)

    # Smoke reduction
    smoke = smoke_reduction(phi_overall, T_zone, mixing_factor, residence_time)

    # Combustion efficiency
    efficiency = combustion_efficiency(phi_overall, T_zone)

    return {
        "phi_primary": phi_primary,
        "phi_overall": phi_overall,
        "T_flame": T_flame,
        "T_zone": T_zone,
        "smoke_index": smoke.value,
        "smoke_reduction_factor": smoke.reduction_factor,
        "efficiency": efficiency
    }


# ============================================================================
# Domain Registration
# ============================================================================

class CombustionLightOperations:
    """Combustion light domain operations."""

    @staticmethod
    def equivalence_ratio(fuel_rate, air_rate, stoichiometric_ratio=0.0676):
        return equivalence_ratio(fuel_rate, air_rate, stoichiometric_ratio)

    @staticmethod
    def adiabatic_flame_temperature(phi, T_reactants, fuel_type="wood"):
        return adiabatic_flame_temperature(phi, T_reactants, fuel_type)

    @staticmethod
    def zone_temperature(T_flame, secondary_air_flow, secondary_air_temp, primary_flow, mixing_factor, model="energy_balance"):
        return zone_temperature(T_flame, secondary_air_flow, secondary_air_temp, primary_flow, mixing_factor, model)

    @staticmethod
    def smoke_reduction(phi, T_zone, mixing_factor, residence_time, model="empirical"):
        return smoke_reduction(phi, T_zone, mixing_factor, residence_time, model)

    @staticmethod
    def combustion_efficiency(phi, T_zone):
        return combustion_efficiency(phi, T_zone)

    @staticmethod
    def emissions_index(phi, T_zone, species="CO"):
        return emissions_index(phi, T_zone, species)

    @staticmethod
    def analyze_fire_pit_combustion(fuel_rate, primary_air_rate, secondary_air_rate, secondary_air_temp, mixing_factor, residence_time):
        return analyze_fire_pit_combustion(fuel_rate, primary_air_rate, secondary_air_rate, secondary_air_temp, mixing_factor, residence_time)


# Create domain instance
combustion_light = CombustionLightOperations()


__all__ = [
    'MixtureState', 'CombustionZone', 'SmokeIndex',
    'equivalence_ratio', 'adiabatic_flame_temperature', 'zone_temperature',
    'smoke_reduction', 'combustion_efficiency', 'emissions_index',
    'analyze_fire_pit_combustion', 'combustion_light', 'CombustionLightOperations'
]
