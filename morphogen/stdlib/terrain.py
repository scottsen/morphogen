"""
Terrain Generation Domain

Provides procedural terrain generation and manipulation:
- Heightmap generation using noise
- Erosion simulation (hydraulic, thermal)
- Terrain analysis (slope, aspect, curvature)
- Biome placement
- Level of detail (LOD) generation
- Export to common formats

Extends the noise domain with terrain-specific operations.

Version: v0.10.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List
from enum import Enum

from morphogen.core.operator import operator, OpCategory


class ErosionType(Enum):
    """Erosion simulation types"""
    HYDRAULIC = "hydraulic"  # Water erosion
    THERMAL = "thermal"      # Temperature-based erosion


class BiomeType(Enum):
    """Terrain biome types"""
    OCEAN = "ocean"
    BEACH = "beach"
    GRASSLAND = "grassland"
    FOREST = "forest"
    MOUNTAIN = "mountain"
    SNOW = "snow"
    DESERT = "desert"


@dataclass
class Heightmap:
    """2D heightmap data structure

    Attributes:
        data: 2D array of height values
        scale: Real-world scale (meters per pixel)
        min_height: Minimum height value
        max_height: Maximum height value
    """
    data: np.ndarray
    scale: float = 1.0
    min_height: float = 0.0
    max_height: float = 1.0

    def copy(self) -> 'Heightmap':
        """Create a deep copy"""
        return Heightmap(
            data=self.data.copy(),
            scale=self.scale,
            min_height=self.min_height,
            max_height=self.max_height
        )

    @property
    def shape(self) -> Tuple[int, int]:
        """Heightmap shape (rows, cols)"""
        return self.data.shape


@dataclass
class BiomeMap:
    """2D biome classification map

    Attributes:
        data: 2D array of biome type indices
        biome_types: List of biome types
    """
    data: np.ndarray
    biome_types: List[BiomeType]

    def copy(self) -> 'BiomeMap':
        """Create a deep copy"""
        return BiomeMap(
            data=self.data.copy(),
            biome_types=list(self.biome_types)
        )


class TerrainOperations:
    """Terrain generation and manipulation operations"""

    @staticmethod
    @operator(
        domain="terrain",
        category=OpCategory.CONSTRUCT,
        signature="(width: int, height: int, scale: float) -> Heightmap",
        deterministic=True,
        doc="Create an empty heightmap"
    )
    def create_heightmap(width: int, height: int, scale: float = 1.0) -> Heightmap:
        """Create an empty heightmap

        Args:
            width: Width in pixels
            height: Height in pixels
            scale: Real-world scale (meters per pixel)

        Returns:
            Empty heightmap (all zeros)
        """
        return Heightmap(
            data=np.zeros((height, width), dtype=np.float64),
            scale=scale
        )

    @staticmethod
    @operator(
        domain="terrain",
        category=OpCategory.CONSTRUCT,
        signature="(width: int, height: int, scale: float, octaves: int, persistence: float, lacunarity: float, seed: Optional[int]) -> Heightmap",
        deterministic=False,  # Uses random seed
        doc="Generate heightmap using Perlin noise"
    )
    def from_noise_perlin(width: int, height: int, scale: float = 50.0,
                         octaves: int = 6, persistence: float = 0.5,
                         lacunarity: float = 2.0, seed: Optional[int] = None) -> Heightmap:
        """Generate heightmap using Perlin noise

        Args:
            width: Width in pixels
            height: Height in pixels
            scale: Noise scale (larger = smoother)
            octaves: Number of noise octaves
            persistence: Amplitude decrease per octave
            lacunarity: Frequency increase per octave
            seed: Random seed

        Returns:
            Generated heightmap
        """
        if seed is not None:
            np.random.seed(seed)

        heightmap = np.zeros((height, width))

        for octave in range(octaves):
            freq = lacunarity ** octave
            amp = persistence ** octave

            # Simple Perlin-like noise using random gradients
            noise_scale = scale / freq
            noise = TerrainOperations._generate_noise_layer(
                width, height, noise_scale, seed + octave if seed else None
            )

            heightmap += noise * amp

        # Normalize to [0, 1]
        heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min() + 1e-10)

        return Heightmap(data=heightmap, min_height=0.0, max_height=1.0)

    @staticmethod
    def _generate_noise_layer(width: int, height: int, scale: float,
                             seed: Optional[int] = None) -> np.ndarray:
        """Generate a single noise layer"""
        if seed is not None:
            np.random.seed(seed)

        # Create grid
        x = np.linspace(0, width / scale, width)
        y = np.linspace(0, height / scale, height)
        xx, yy = np.meshgrid(x, y)

        # Simple value noise (can be replaced with real Perlin)
        grid_size = max(1, int(max(width, height) / scale))
        grad_x = np.random.randn(grid_size + 2, grid_size + 2)
        grad_y = np.random.randn(grid_size + 2, grid_size + 2)

        # Interpolate
        xx_norm = xx / width * grid_size
        yy_norm = yy / height * grid_size

        x0 = np.floor(xx_norm).astype(int)
        y0 = np.floor(yy_norm).astype(int)
        x1 = x0 + 1
        y1 = y0 + 1

        sx = xx_norm - x0
        sy = yy_norm - y0

        # Bilinear interpolation
        n00 = grad_x[y0, x0]
        n10 = grad_x[y0, x1]
        n01 = grad_x[y1, x0]
        n11 = grad_x[y1, x1]

        nx0 = n00 * (1 - sx) + n10 * sx
        nx1 = n01 * (1 - sx) + n11 * sx

        return nx0 * (1 - sy) + nx1 * sy

    @staticmethod
    @operator(
        domain="terrain",
        category=OpCategory.TRANSFORM,
        signature="(terrain: Heightmap, iterations: int, rain_amount: float, evaporation: float, capacity: float) -> Heightmap",
        deterministic=True,
        doc="Simulate hydraulic (water) erosion"
    )
    def hydraulic_erosion(terrain: Heightmap, iterations: int = 100,
                         rain_amount: float = 0.01,
                         evaporation: float = 0.5,
                         capacity: float = 0.1) -> Heightmap:
        """Simulate hydraulic (water) erosion

        Args:
            terrain: Input heightmap
            iterations: Number of erosion iterations
            rain_amount: Amount of water added per iteration
            evaporation: Water evaporation rate (0-1)
            capacity: Sediment carrying capacity

        Returns:
            Eroded heightmap
        """
        result = terrain.copy()
        h, w = result.data.shape

        # Water and sediment maps
        water = np.zeros((h, w))
        sediment = np.zeros((h, w))

        for _ in range(iterations):
            # Add rain
            water += rain_amount

            # Calculate water flow (simple 4-neighbor flow)
            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    if water[i, j] > 0:
                        # Get neighbors
                        neighbors = [
                            (i-1, j), (i+1, j), (i, j-1), (i, j+1)
                        ]

                        # Flow to lowest neighbor
                        min_height = result.data[i, j]
                        flow_target = None

                        for ni, nj in neighbors:
                            if result.data[ni, nj] < min_height:
                                min_height = result.data[ni, nj]
                                flow_target = (ni, nj)

                        if flow_target:
                            # Erode current cell
                            height_diff = result.data[i, j] - min_height
                            erosion_amount = min(height_diff * 0.1, water[i, j] * capacity)

                            result.data[i, j] -= erosion_amount
                            sediment[i, j] += erosion_amount

                            # Flow water and sediment
                            ti, tj = flow_target
                            flow_amount = water[i, j] * 0.5

                            water[ti, tj] += flow_amount
                            water[i, j] -= flow_amount

                            sediment[ti, tj] += sediment[i, j] * 0.5
                            sediment[i, j] *= 0.5

                            # Deposit sediment
                            if water[ti, tj] < capacity:
                                deposit = min(sediment[ti, tj], capacity - water[ti, tj])
                                result.data[ti, tj] += deposit
                                sediment[ti, tj] -= deposit

            # Evaporate water
            water *= (1 - evaporation)

        return result

    @staticmethod
    @operator(
        domain="terrain",
        category=OpCategory.TRANSFORM,
        signature="(terrain: Heightmap, iterations: int, talus_angle: float) -> Heightmap",
        deterministic=True,
        doc="Simulate thermal erosion (smoothing based on slope)"
    )
    def thermal_erosion(terrain: Heightmap, iterations: int = 50,
                       talus_angle: float = 0.5) -> Heightmap:
        """Simulate thermal erosion (smoothing based on slope)

        Args:
            terrain: Input heightmap
            iterations: Number of erosion iterations
            talus_angle: Critical slope angle (material slides if exceeded)

        Returns:
            Eroded heightmap
        """
        result = terrain.copy()
        h, w = result.data.shape

        for _ in range(iterations):
            new_data = result.data.copy()

            for i in range(1, h - 1):
                for j in range(1, w - 1):
                    # Check all neighbors
                    neighbors = [
                        (i-1, j), (i+1, j), (i, j-1), (i, j+1)
                    ]

                    for ni, nj in neighbors:
                        height_diff = result.data[i, j] - result.data[ni, nj]

                        # If slope exceeds talus angle, move material
                        if height_diff > talus_angle:
                            transfer = (height_diff - talus_angle) * 0.5
                            new_data[i, j] -= transfer
                            new_data[ni, nj] += transfer

            result.data = new_data

        return result

    @staticmethod
    @operator(
        domain="terrain",
        category=OpCategory.QUERY,
        signature="(terrain: Heightmap) -> np.ndarray",
        deterministic=True,
        doc="Calculate terrain slope at each point"
    )
    def calculate_slope(terrain: Heightmap) -> np.ndarray:
        """Calculate terrain slope at each point

        Args:
            terrain: Input heightmap

        Returns:
            2D array of slope values (radians)
        """
        dy, dx = np.gradient(terrain.data)
        slope = np.arctan(np.sqrt(dx**2 + dy**2))
        return slope

    @staticmethod
    @operator(
        domain="terrain",
        category=OpCategory.QUERY,
        signature="(terrain: Heightmap) -> np.ndarray",
        deterministic=True,
        doc="Calculate terrain aspect (direction of slope)"
    )
    def calculate_aspect(terrain: Heightmap) -> np.ndarray:
        """Calculate terrain aspect (direction of slope)

        Args:
            terrain: Input heightmap

        Returns:
            2D array of aspect values (radians, 0=North, clockwise)
        """
        dy, dx = np.gradient(terrain.data)
        aspect = np.arctan2(dy, dx)
        return aspect

    @staticmethod
    @operator(
        domain="terrain",
        category=OpCategory.QUERY,
        signature="(terrain: Heightmap, moisture: Optional[np.ndarray], temperature: Optional[np.ndarray]) -> BiomeMap",
        deterministic=True,
        doc="Classify terrain into biomes based on height, moisture, temperature"
    )
    def classify_biomes(terrain: Heightmap, moisture: Optional[np.ndarray] = None,
                       temperature: Optional[np.ndarray] = None) -> BiomeMap:
        """Classify terrain into biomes based on height, moisture, temperature

        Args:
            terrain: Input heightmap
            moisture: Optional moisture map (0-1, default based on height)
            temperature: Optional temperature map (0-1, default based on height)

        Returns:
            Biome classification map
        """
        h, w = terrain.shape

        # Default moisture: higher in valleys
        if moisture is None:
            moisture = 1.0 - terrain.data

        # Default temperature: lower at higher elevations
        if temperature is None:
            temperature = 1.0 - terrain.data * 0.8

        biome_data = np.zeros((h, w), dtype=int)
        biome_types = [BiomeType.OCEAN, BiomeType.BEACH, BiomeType.GRASSLAND,
                      BiomeType.FOREST, BiomeType.MOUNTAIN, BiomeType.SNOW, BiomeType.DESERT]

        for i in range(h):
            for j in range(w):
                height = terrain.data[i, j]
                temp = temperature[i, j]
                moist = moisture[i, j]

                # Classification rules
                if height < 0.3:
                    biome_data[i, j] = 0  # Ocean
                elif height < 0.35:
                    biome_data[i, j] = 1  # Beach
                elif height > 0.8:
                    if temp < 0.3:
                        biome_data[i, j] = 5  # Snow
                    else:
                        biome_data[i, j] = 4  # Mountain
                elif temp < 0.4 and moist < 0.3:
                    biome_data[i, j] = 6  # Desert
                elif moist > 0.6:
                    biome_data[i, j] = 3  # Forest
                else:
                    biome_data[i, j] = 2  # Grassland

        return BiomeMap(data=biome_data, biome_types=biome_types)

    @staticmethod
    @operator(
        domain="terrain",
        category=OpCategory.TRANSFORM,
        signature="(terrain: Heightmap, num_levels: int) -> Heightmap",
        deterministic=True,
        doc="Create terraced terrain (stepped levels)"
    )
    def terrace(terrain: Heightmap, num_levels: int = 5) -> Heightmap:
        """Create terraced terrain (stepped levels)

        Args:
            terrain: Input heightmap
            num_levels: Number of terrace levels

        Returns:
            Terraced heightmap
        """
        result = terrain.copy()

        # Quantize heights to discrete levels
        level_height = 1.0 / num_levels
        result.data = np.floor(result.data / level_height) * level_height

        return result

    @staticmethod
    @operator(
        domain="terrain",
        category=OpCategory.TRANSFORM,
        signature="(terrain: Heightmap, iterations: int, strength: float) -> Heightmap",
        deterministic=True,
        doc="Smooth terrain using averaging filter"
    )
    def smooth(terrain: Heightmap, iterations: int = 1, strength: float = 0.5) -> Heightmap:
        """Smooth terrain using averaging filter

        Args:
            terrain: Input heightmap
            iterations: Number of smoothing iterations
            strength: Smoothing strength (0-1)

        Returns:
            Smoothed heightmap
        """
        from scipy.ndimage import uniform_filter

        result = terrain.copy()

        for _ in range(iterations):
            smoothed = uniform_filter(result.data, size=3, mode='reflect')
            result.data = result.data * (1 - strength) + smoothed * strength

        return result

    @staticmethod
    @operator(
        domain="terrain",
        category=OpCategory.TRANSFORM,
        signature="(terrain: Heightmap, min_val: float, max_val: float) -> Heightmap",
        deterministic=True,
        doc="Normalize heightmap to specified range"
    )
    def normalize(terrain: Heightmap, min_val: float = 0.0, max_val: float = 1.0) -> Heightmap:
        """Normalize heightmap to specified range

        Args:
            terrain: Input heightmap
            min_val: Minimum value
            max_val: Maximum value

        Returns:
            Normalized heightmap
        """
        result = terrain.copy()

        data_min = result.data.min()
        data_max = result.data.max()

        if data_max > data_min:
            result.data = (result.data - data_min) / (data_max - data_min)
            result.data = result.data * (max_val - min_val) + min_val

        result.min_height = min_val
        result.max_height = max_val

        return result

    @staticmethod
    @operator(
        domain="terrain",
        category=OpCategory.TRANSFORM,
        signature="(terrain: Heightmap, falloff: float) -> Heightmap",
        deterministic=True,
        doc="Apply radial falloff to create island"
    )
    def island_mask(terrain: Heightmap, falloff: float = 0.5) -> Heightmap:
        """Apply radial falloff to create island

        Args:
            terrain: Input heightmap
            falloff: Falloff strength (higher = steeper falloff)

        Returns:
            Heightmap with island shape
        """
        result = terrain.copy()
        h, w = result.shape

        # Create distance map from center
        cy, cx = h / 2, w / 2
        y, x = np.ogrid[:h, :w]

        distance = np.sqrt((x - cx)**2 + (y - cy)**2)
        max_distance = np.sqrt(cx**2 + cy**2)
        distance_norm = distance / max_distance

        # Apply falloff
        mask = 1.0 - np.power(distance_norm, falloff)
        mask = np.clip(mask, 0, 1)

        result.data *= mask

        return result


# Export singleton instance for DSL access
terrain = TerrainOperations()

# Export operators for domain registry discovery
create_heightmap = TerrainOperations.create_heightmap
from_noise_perlin = TerrainOperations.from_noise_perlin
hydraulic_erosion = TerrainOperations.hydraulic_erosion
thermal_erosion = TerrainOperations.thermal_erosion
calculate_slope = TerrainOperations.calculate_slope
calculate_aspect = TerrainOperations.calculate_aspect
classify_biomes = TerrainOperations.classify_biomes
terrace = TerrainOperations.terrace
smooth = TerrainOperations.smooth
normalize = TerrainOperations.normalize
island_mask = TerrainOperations.island_mask
