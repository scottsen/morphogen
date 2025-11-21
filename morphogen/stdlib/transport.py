"""TransportDomain - Heat and mass transport phenomena.

This module implements heat transfer (conduction, convection, radiation),
mass transfer (diffusion, convection), and porous media transport. Essential
for reactor design, heat exchangers, and separation processes.

Specification: docs/specifications/chemistry.md
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
from enum import Enum


# ============================================================================
# Constants
# ============================================================================

STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²·K⁴)
BOLTZMANN_K = 1.380649e-23  # J/K
AVOGADRO = 6.02214076e23  # 1/mol


# ============================================================================
# Core Types
# ============================================================================

class GeometryType(Enum):
    """Geometry types for correlations."""
    PIPE = "pipe"
    FLAT_PLATE = "flat_plate"
    SPHERE = "sphere"
    CYLINDER = "cylinder"


@dataclass
class FluidProperties:
    """Fluid transport properties."""
    density: float  # kg/m³
    viscosity: float  # Pa·s
    thermal_conductivity: float  # W/(m·K)
    specific_heat: float  # J/(kg·K)
    diffusivity: float = 1e-9  # m²/s (molecular diffusion)

    def prandtl_number(self) -> float:
        """Compute Prandtl number: Pr = μ·Cp / k."""
        return self.viscosity * self.specific_heat / self.thermal_conductivity

    def schmidt_number(self) -> float:
        """Compute Schmidt number: Sc = μ / (ρ·D)."""
        return self.viscosity / (self.density * self.diffusivity)

    def reynolds_number(self, velocity: float, length: float) -> float:
        """Compute Reynolds number: Re = ρ·v·L / μ."""
        return self.density * velocity * length / self.viscosity


# ============================================================================
# Heat Transfer Operators
# ============================================================================

def conduction(
    temp_gradient: float,
    thermal_conductivity: float,
    area: float
) -> float:
    """Compute heat transfer by conduction (Fourier's law).

    q = -k · A · dT/dx

    Args:
        temp_gradient: Temperature gradient (K/m)
        thermal_conductivity: Thermal conductivity (W/(m·K))
        area: Cross-sectional area (m²)

    Returns:
        q: Heat transfer rate (W)

    Determinism: strict
    """
    q = -thermal_conductivity * area * temp_gradient
    return q


def convection(
    temp_surface: float,
    temp_bulk: float,
    h: float,
    area: float
) -> float:
    """Compute heat transfer by convection (Newton's law of cooling).

    q = h · A · (T_surface - T_bulk)

    Args:
        temp_surface: Surface temperature (K)
        temp_bulk: Bulk fluid temperature (K)
        h: Heat transfer coefficient (W/(m²·K))
        area: Surface area (m²)

    Returns:
        q: Heat transfer rate (W)

    Determinism: strict
    """
    q = h * area * (temp_surface - temp_bulk)
    return q


def radiation(
    temp_surface: float,
    temp_ambient: float,
    emissivity: float,
    area: float
) -> float:
    """Compute heat transfer by radiation (Stefan-Boltzmann law).

    q = ε · σ · A · (T_surface⁴ - T_ambient⁴)

    Args:
        temp_surface: Surface temperature (K)
        temp_ambient: Ambient temperature (K)
        emissivity: Surface emissivity (0-1)
        area: Surface area (m²)

    Returns:
        q: Heat transfer rate (W)

    Determinism: strict
    """
    q = emissivity * STEFAN_BOLTZMANN * area * (temp_surface**4 - temp_ambient**4)
    return q


def nusselt_correlation(
    Re: float,
    Pr: float,
    geometry: str = "pipe",
    L_over_D: Optional[float] = None
) -> float:
    """Compute Nusselt number from empirical correlations.

    The Nusselt number relates to heat transfer coefficient: h = Nu · k / L

    Args:
        Re: Reynolds number
        Pr: Prandtl number
        geometry: Geometry type (pipe, flat_plate, sphere, cylinder)
        L_over_D: Length to diameter ratio (for pipes)

    Returns:
        Nu: Nusselt number

    Determinism: strict
    """
    if geometry == "pipe":
        if Re < 2300:
            # Laminar flow in pipe
            Nu = 3.66
        elif Re < 10000:
            # Transition region (Gnielinski correlation)
            f = (0.79 * np.log(Re) - 1.64)**(-2)
            Nu = ((f/8) * (Re - 1000) * Pr) / (1 + 12.7 * np.sqrt(f/8) * (Pr**(2/3) - 1))
        else:
            # Turbulent flow (Dittus-Boelter)
            Nu = 0.023 * Re**0.8 * Pr**0.4

    elif geometry == "flat_plate":
        if Re < 5e5:
            # Laminar boundary layer
            Nu = 0.664 * Re**0.5 * Pr**(1/3)
        else:
            # Turbulent boundary layer
            Nu = 0.037 * Re**0.8 * Pr**(1/3)

    elif geometry == "sphere":
        # Ranz-Marshall correlation
        Nu = 2.0 + 0.6 * Re**0.5 * Pr**(1/3)

    elif geometry == "cylinder":
        # Churchill-Bernstein correlation
        Nu = 0.3 + (0.62 * Re**0.5 * Pr**(1/3)) / (1 + (0.4/Pr)**(2/3))**0.25

    else:
        # Default
        Nu = 2.0

    return Nu


def heat_transfer_coefficient(
    Re: float,
    Pr: float,
    thermal_conductivity: float,
    length: float,
    geometry: str = "pipe"
) -> float:
    """Compute heat transfer coefficient from correlations.

    h = Nu · k / L

    Args:
        Re: Reynolds number
        Pr: Prandtl number
        thermal_conductivity: Fluid thermal conductivity (W/(m·K))
        length: Characteristic length (m)
        geometry: Geometry type

    Returns:
        h: Heat transfer coefficient (W/(m²·K))

    Determinism: strict
    """
    Nu = nusselt_correlation(Re, Pr, geometry)
    h = Nu * thermal_conductivity / length
    return h


# ============================================================================
# Mass Transfer Operators
# ============================================================================

def fickian_diffusion(
    conc_gradient: float,
    diffusivity: float,
    area: float
) -> float:
    """Compute mass flux by Fickian diffusion.

    J = -D · A · dc/dx

    Args:
        conc_gradient: Concentration gradient (mol/(m³·m))
        diffusivity: Diffusion coefficient (m²/s)
        area: Cross-sectional area (m²)

    Returns:
        flux: Mass flux (mol/s)

    Determinism: strict
    """
    flux = -diffusivity * area * conc_gradient
    return flux


def knudsen_diffusion(
    pore_diameter: float,
    temp: float,
    molecular_weight: float
) -> float:
    """Compute Knudsen diffusion coefficient in pores.

    D_K = (d_pore / 3) · sqrt(8·R·T / (π·M))

    Args:
        pore_diameter: Pore diameter (m)
        temp: Temperature (K)
        molecular_weight: Molecular weight (kg/mol)

    Returns:
        D_K: Knudsen diffusion coefficient (m²/s)

    Determinism: strict
    """
    R = 8.314  # J/(mol·K)
    D_K = (pore_diameter / 3.0) * np.sqrt(8 * R * temp / (np.pi * molecular_weight))
    return D_K


def convective_mass_transfer(
    conc_bulk: float,
    conc_surface: float,
    k_mass: float,
    area: float
) -> float:
    """Compute mass transfer by convection.

    J = k_L · A · (c_bulk - c_surface)

    Args:
        conc_bulk: Bulk concentration (mol/m³)
        conc_surface: Surface concentration (mol/m³)
        k_mass: Mass transfer coefficient (m/s)
        area: Surface area (m²)

    Returns:
        flux: Mass flux (mol/s)

    Determinism: strict
    """
    flux = k_mass * area * (conc_bulk - conc_surface)
    return flux


def sherwood_correlation(
    Re: float,
    Sc: float,
    geometry: str = "pipe"
) -> float:
    """Compute Sherwood number from empirical correlations.

    The Sherwood number relates to mass transfer coefficient: k_L = Sh · D / L

    Args:
        Re: Reynolds number
        Sc: Schmidt number
        geometry: Geometry type

    Returns:
        Sh: Sherwood number

    Determinism: strict
    """
    # Mass transfer correlations are analogous to heat transfer
    # Sh ~ Nu with Sc replacing Pr

    if geometry == "pipe":
        if Re < 2300:
            # Laminar flow
            Sh = 3.66
        else:
            # Turbulent flow (similar to Dittus-Boelter)
            Sh = 0.023 * Re**0.8 * Sc**0.33

    elif geometry == "sphere":
        # Ranz-Marshall correlation
        Sh = 2.0 + 0.6 * Re**0.5 * Sc**(1/3)

    elif geometry == "flat_plate":
        if Re < 5e5:
            # Laminar
            Sh = 0.664 * Re**0.5 * Sc**(1/3)
        else:
            # Turbulent
            Sh = 0.037 * Re**0.8 * Sc**(1/3)

    else:
        Sh = 2.0

    return Sh


def mass_transfer_coefficient(
    Re: float,
    Sc: float,
    diffusivity: float,
    length: float,
    geometry: str = "sphere"
) -> float:
    """Compute mass transfer coefficient from correlations.

    k_L = Sh · D / L

    Args:
        Re: Reynolds number
        Sc: Schmidt number
        diffusivity: Molecular diffusivity (m²/s)
        length: Characteristic length (m)
        geometry: Geometry type

    Returns:
        k_L: Mass transfer coefficient (m/s)

    Determinism: strict
    """
    Sh = sherwood_correlation(Re, Sc, geometry)
    k_L = Sh * diffusivity / length
    return k_L


# ============================================================================
# Porous Media Operators
# ============================================================================

def effective_diffusivity(
    D_molecular: float,
    porosity: float,
    tortuosity: float
) -> float:
    """Compute effective diffusivity in porous media.

    D_eff = D_molecular · (ε / τ)

    Args:
        D_molecular: Molecular diffusion coefficient (m²/s)
        porosity: Porosity (void fraction, 0-1)
        tortuosity: Tortuosity factor (>1)

    Returns:
        D_eff: Effective diffusivity (m²/s)

    Determinism: strict
    """
    D_eff = D_molecular * porosity / tortuosity
    return D_eff


def darcy_flow(
    pressure_gradient: float,
    permeability: float,
    viscosity: float
) -> float:
    """Compute flow velocity through porous media (Darcy's law).

    v = -(κ / μ) · dP/dx

    Args:
        pressure_gradient: Pressure gradient (Pa/m)
        permeability: Permeability (m²)
        viscosity: Fluid viscosity (Pa·s)

    Returns:
        velocity: Darcy velocity (m/s)

    Determinism: strict
    """
    velocity = -(permeability / viscosity) * pressure_gradient
    return velocity


def carman_kozeny(
    porosity: float,
    particle_diameter: float
) -> float:
    """Compute permeability from Carman-Kozeny equation.

    κ = (d_p² · ε³) / (180 · (1 - ε)²)

    Args:
        porosity: Porosity (0-1)
        particle_diameter: Particle diameter (m)

    Returns:
        permeability: Permeability (m²)

    Determinism: strict
    """
    kappa = (particle_diameter**2 * porosity**3) / (180.0 * (1.0 - porosity)**2)
    return kappa


# ============================================================================
# Dimensionless Numbers
# ============================================================================

def reynolds_number(
    density: float,
    velocity: float,
    length: float,
    viscosity: float
) -> float:
    """Compute Reynolds number.

    Re = ρ·v·L / μ

    Args:
        density: Fluid density (kg/m³)
        velocity: Characteristic velocity (m/s)
        length: Characteristic length (m)
        viscosity: Dynamic viscosity (Pa·s)

    Returns:
        Re: Reynolds number

    Determinism: strict
    """
    Re = density * velocity * length / viscosity
    return Re


def prandtl_number(
    viscosity: float,
    specific_heat: float,
    thermal_conductivity: float
) -> float:
    """Compute Prandtl number.

    Pr = μ·Cp / k

    Args:
        viscosity: Dynamic viscosity (Pa·s)
        specific_heat: Specific heat (J/(kg·K))
        thermal_conductivity: Thermal conductivity (W/(m·K))

    Returns:
        Pr: Prandtl number

    Determinism: strict
    """
    Pr = viscosity * specific_heat / thermal_conductivity
    return Pr


def schmidt_number(
    viscosity: float,
    density: float,
    diffusivity: float
) -> float:
    """Compute Schmidt number.

    Sc = μ / (ρ·D)

    Args:
        viscosity: Dynamic viscosity (Pa·s)
        density: Density (kg/m³)
        diffusivity: Molecular diffusivity (m²/s)

    Returns:
        Sc: Schmidt number

    Determinism: strict
    """
    Sc = viscosity / (density * diffusivity)
    return Sc


def peclet_number(
    velocity: float,
    length: float,
    diffusivity: float
) -> float:
    """Compute Peclet number (ratio of convection to diffusion).

    Pe = v·L / D

    Args:
        velocity: Velocity (m/s)
        length: Length (m)
        diffusivity: Diffusivity (m²/s)

    Returns:
        Pe: Peclet number

    Determinism: strict
    """
    Pe = velocity * length / diffusivity
    return Pe


# ============================================================================
# Domain Registration
# ============================================================================

class TransportOperations:
    """Transport phenomena domain operations."""

    # Heat Transfer
    conduction = staticmethod(conduction)
    convection = staticmethod(convection)
    radiation = staticmethod(radiation)
    nusselt_correlation = staticmethod(nusselt_correlation)
    heat_transfer_coefficient = staticmethod(heat_transfer_coefficient)

    # Mass Transfer
    fickian_diffusion = staticmethod(fickian_diffusion)
    knudsen_diffusion = staticmethod(knudsen_diffusion)
    convective_mass_transfer = staticmethod(convective_mass_transfer)
    sherwood_correlation = staticmethod(sherwood_correlation)
    mass_transfer_coefficient = staticmethod(mass_transfer_coefficient)

    # Porous Media
    effective_diffusivity = staticmethod(effective_diffusivity)
    darcy_flow = staticmethod(darcy_flow)
    carman_kozeny = staticmethod(carman_kozeny)

    # Dimensionless Numbers
    reynolds_number = staticmethod(reynolds_number)
    prandtl_number = staticmethod(prandtl_number)
    schmidt_number = staticmethod(schmidt_number)
    peclet_number = staticmethod(peclet_number)


# Create domain instance
transport = TransportOperations()


__all__ = [
    'GeometryType', 'FluidProperties',
    'conduction', 'convection', 'radiation',
    'nusselt_correlation', 'heat_transfer_coefficient',
    'fickian_diffusion', 'knudsen_diffusion', 'convective_mass_transfer',
    'sherwood_correlation', 'mass_transfer_coefficient',
    'effective_diffusivity', 'darcy_flow', 'carman_kozeny',
    'reynolds_number', 'prandtl_number', 'schmidt_number', 'peclet_number',
    'transport', 'TransportOperations'
]
