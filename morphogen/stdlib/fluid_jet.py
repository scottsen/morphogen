"""FluidJetDomain - Model jet exits and their interaction with surrounding flow.

This module implements simplified jet modeling (momentum, entrainment, mixing)
without full CFD. Essential for fire pits, burners, rocket nozzles, spray systems.

Specification: docs/specifications/physics-domains.md
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ============================================================================
# Core Types
# ============================================================================

@dataclass
class Jet:
    """Single jet."""
    flow: float  # Mass flow rate (kg/s)
    velocity: float  # Exit velocity (m/s)
    temperature: float  # Exit temperature (K)
    direction: Tuple[float, float, float]  # Jet direction (unit vector)
    area: float  # Nozzle area (m²)
    position: Tuple[float, float, float]  # Jet origin (m)

    @property
    def momentum_flux(self) -> float:
        """Momentum flux (N)."""
        return self.flow * self.velocity

    @property
    def diameter(self) -> float:
        """Equivalent diameter (m)."""
        return np.sqrt(4.0 * self.area / np.pi)


@dataclass
class JetArray:
    """Collection of jets."""
    jets: List[Jet]

    @property
    def count(self) -> int:
        return len(self.jets)

    @property
    def total_flow(self) -> float:
        return sum(jet.flow for jet in self.jets)


# ============================================================================
# Operators
# ============================================================================

def jet_from_tube(
    tube_diameter: float,
    tube_position: Tuple[float, float, float],
    tube_direction: Tuple[float, float, float],
    m_dot: float,
    T_out: float,
    rho: Optional[float] = None
) -> Jet:
    """Create jet from tube exit flow conditions.

    Args:
        tube_diameter: Tube diameter (m)
        tube_position: Tube exit position (m)
        tube_direction: Tube direction (unit vector)
        m_dot: Mass flow rate (kg/s)
        T_out: Exit temperature (K)
        rho: Fluid density (kg/m³), if None uses ideal gas

    Returns:
        Jet object

    Determinism: strict
    """
    # Calculate density from temperature if not provided (ideal gas)
    if rho is None:
        # Air: P/(R·T) with P=101325 Pa, R=287 J/(kg·K)
        rho = 101325.0 / (287.0 * T_out)

    # Calculate area and velocity
    area = np.pi * (tube_diameter / 2.0)**2
    velocity = m_dot / (rho * area) if area > 0 else 0.0

    # Normalize direction vector
    dir_array = np.array(tube_direction)
    dir_norm = np.linalg.norm(dir_array)
    if dir_norm > 0:
        dir_array = dir_array / dir_norm
    direction = tuple(dir_array)

    return Jet(
        flow=m_dot,
        velocity=velocity,
        temperature=T_out,
        direction=direction,
        area=area,
        position=tube_position
    )


def jet_reynolds(
    jet: Jet,
    mu: float = 1.8e-5
) -> float:
    """Compute jet Reynolds number.

    Formula: Re = ρ·v·D/μ

    Args:
        jet: Jet object
        mu: Dynamic viscosity (Pa·s), default for air

    Returns:
        Re: Reynolds number (dimensionless)

    Determinism: strict
    """
    # Calculate density from temperature (ideal gas)
    rho = 101325.0 / (287.0 * jet.temperature)

    Re = rho * jet.velocity * jet.diameter / mu
    return Re


def jet_entrainment(
    jet: Jet,
    plume_velocity: float,
    plume_density: float,
    model: str = "empirical"
) -> float:
    """Estimate jet entrainment and mixing with ambient flow.

    Args:
        jet: Jet object
        plume_velocity: Ambient plume velocity (m/s)
        plume_density: Ambient plume density (kg/m³)
        model: "empirical" or "momentum_ratio"

    Returns:
        mixing_factor: Mixing effectiveness (0-1)

    Determinism: repro
    """
    # Calculate jet density
    jet_density = 101325.0 / (287.0 * jet.temperature)

    if model == "empirical":
        # Empirical correlation: mixing increases with velocity ratio
        velocity_ratio = jet.velocity / max(plume_velocity, 0.1)
        mixing_factor = min(1.0, 0.1 * np.sqrt(velocity_ratio))

    elif model == "momentum_ratio":
        # Momentum ratio model
        jet_momentum = jet_density * jet.velocity**2
        plume_momentum = plume_density * plume_velocity**2
        momentum_ratio = jet_momentum / max(plume_momentum, 1e-10)
        mixing_factor = min(1.0, 0.5 * np.sqrt(momentum_ratio))

    else:
        raise ValueError(f"Unknown model: {model}")

    return mixing_factor


def jet_field_2d(
    jet_array: JetArray,
    grid_size: Tuple[int, int],
    grid_bounds: Tuple[float, float, float, float],  # (x_min, x_max, y_min, y_max)
    decay: float = 0.1
) -> np.ndarray:
    """Generate 2D vector field visualization of jets (CFD-lite).

    Uses simple Gaussian jet profiles for quick visualization.

    Args:
        jet_array: Array of jets
        grid_size: (nx, ny) grid resolution
        grid_bounds: (x_min, x_max, y_min, y_max) in meters
        decay: Jet decay rate (1/m)

    Returns:
        field: (nx, ny, 2) array of velocity vectors (m/s)

    Determinism: repro
    """
    nx, ny = grid_size
    x_min, x_max, y_min, y_max = grid_bounds

    # Create grid
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)

    # Initialize field
    field = np.zeros((ny, nx, 2))

    # Superpose jet contributions
    for jet in jet_array.jets:
        # Project jet onto 2D plane (assume z is vertical)
        jet_x, jet_y, jet_z = jet.position
        jet_vx, jet_vy, jet_vz = jet.direction

        # Distance from each grid point to jet origin
        dx = X - jet_x
        dy = Y - jet_y
        r = np.sqrt(dx**2 + dy**2)

        # Gaussian decay
        strength = jet.velocity * np.exp(-decay * r)

        # Add velocity components
        field[:, :, 0] += strength * jet_vx
        field[:, :, 1] += strength * jet_vy

    return field


def jet_centerline_velocity(
    jet: Jet,
    distance: float
) -> float:
    """Compute jet centerline velocity at distance from nozzle.

    Uses classical jet theory:
        v(x) = v0 · (D / (x + x0))

    Args:
        jet: Jet object
        distance: Distance from nozzle (m)

    Returns:
        velocity: Centerline velocity (m/s)

    Determinism: strict
    """
    # Virtual origin distance
    x0 = 0.5 * jet.diameter

    # Centerline velocity decay
    if distance + x0 > 0:
        velocity = jet.velocity * jet.diameter / (distance + x0)
    else:
        velocity = jet.velocity

    return velocity


def jet_spreading_width(
    jet: Jet,
    distance: float,
    spreading_rate: float = 0.1
) -> float:
    """Compute jet spreading width at distance from nozzle.

    Args:
        jet: Jet object
        distance: Distance from nozzle (m)
        spreading_rate: Spreading rate (dimensionless), typical ~0.1

    Returns:
        width: Jet half-width (m)

    Determinism: strict
    """
    width = jet.diameter / 2.0 + spreading_rate * distance
    return width


# ============================================================================
# Convenience Functions
# ============================================================================

def create_jet_array_radial(
    n_jets: int,
    radius: float,
    jet_diameter: float,
    m_dot_per_jet: float,
    temperature: float,
    height: float = 0.0,
    angle_inward: float = 0.0
) -> JetArray:
    """Create radial array of jets (e.g., fire pit secondary air).

    Args:
        n_jets: Number of jets
        radius: Radial distance from center (m)
        jet_diameter: Diameter of each jet (m)
        m_dot_per_jet: Mass flow per jet (kg/s)
        temperature: Jet temperature (K)
        height: Height above ground (m)
        angle_inward: Angle toward center (radians)

    Returns:
        JetArray
    """
    jets = []

    for i in range(n_jets):
        angle = 2 * np.pi * i / n_jets

        # Position
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height
        position = (x, y, z)

        # Direction (point inward and upward)
        dir_x = -np.cos(angle) * np.cos(angle_inward)
        dir_y = -np.sin(angle) * np.cos(angle_inward)
        dir_z = np.sin(angle_inward)
        direction = (dir_x, dir_y, dir_z)

        # Create jet
        jet = jet_from_tube(
            tube_diameter=jet_diameter,
            tube_position=position,
            tube_direction=direction,
            m_dot=m_dot_per_jet,
            T_out=temperature
        )

        jets.append(jet)

    return JetArray(jets)


# ============================================================================
# Domain Registration
# ============================================================================

class FluidJetOperations:
    """Fluid jet domain operations."""

    @staticmethod
    def jet_from_tube(tube_diameter, tube_position, tube_direction, m_dot, T_out, rho=None):
        return jet_from_tube(tube_diameter, tube_position, tube_direction, m_dot, T_out, rho)

    @staticmethod
    def jet_reynolds(jet, mu=1.8e-5):
        return jet_reynolds(jet, mu)

    @staticmethod
    def jet_entrainment(jet, plume_velocity, plume_density, model="empirical"):
        return jet_entrainment(jet, plume_velocity, plume_density, model)

    @staticmethod
    def jet_field_2d(jet_array, grid_size, grid_bounds, decay=0.1):
        return jet_field_2d(jet_array, grid_size, grid_bounds, decay)

    @staticmethod
    def jet_centerline_velocity(jet, distance):
        return jet_centerline_velocity(jet, distance)

    @staticmethod
    def jet_spreading_width(jet, distance, spreading_rate=0.1):
        return jet_spreading_width(jet, distance, spreading_rate)

    @staticmethod
    def create_jet_array_radial(n_jets, radius, jet_diameter, m_dot_per_jet, temperature, height=0.0, angle_inward=0.0):
        return create_jet_array_radial(n_jets, radius, jet_diameter, m_dot_per_jet, temperature, height, angle_inward)


# Create domain instance
fluid_jet = FluidJetOperations()


__all__ = [
    'Jet', 'JetArray',
    'jet_from_tube', 'jet_reynolds', 'jet_entrainment', 'jet_field_2d',
    'jet_centerline_velocity', 'jet_spreading_width', 'create_jet_array_radial',
    'fluid_jet', 'FluidJetOperations'
]
