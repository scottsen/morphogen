"""Palette operations for scalar-to-color mapping.

This module provides comprehensive palette generation and manipulation for:
- Fractal coloring (Mandelbrot, Julia sets)
- Plasma effects
- Audio visualizers
- Spectrogram rendering
- Heatmaps and scientific visualization
- Procedural art and effects
- Simulation debugging

Supports multiple palette types:
- Gradient-based palettes
- Scientific colormaps (Inferno, Viridis, Plasma)
- Procedural palettes (Cosine gradients, HSV wheels)
- Custom color stops
"""

from typing import List, Tuple, Optional, Union
import numpy as np

from morphogen.core.operator import operator, OpCategory


class Palette:
    """Color palette for mapping scalar values to RGB colors.

    A palette is a 1D array of RGB colors that can be sampled
    using scalar values in [0, 1].
    """

    def __init__(self, colors: np.ndarray, name: str = "custom"):
        """Initialize palette.

        Args:
            colors: RGB color array (shape: (num_colors, 3), values in [0, 1])
            name: Palette name for identification
        """
        if colors.shape[1] != 3:
            raise ValueError(f"Colors must be Nx3 array, got shape {colors.shape}")

        self.colors = colors.astype(np.float32)
        self.name = name
        self.num_colors = len(colors)

    def sample(self, t: Union[float, np.ndarray]) -> np.ndarray:
        """Sample palette at normalized position(s).

        Args:
            t: Normalized position(s) in [0, 1] (scalar or array)

        Returns:
            RGB color(s) (shape: (3,) for scalar or (..., 3) for array)
        """
        t = np.asarray(t, dtype=np.float32)
        is_scalar = t.ndim == 0

        if is_scalar:
            t = np.array([t])

        # Clamp to [0, 1]
        t = np.clip(t, 0.0, 1.0)

        # Map to color index range
        indices = t * (self.num_colors - 1)

        # Integer and fractional parts
        idx0 = np.floor(indices).astype(np.int32)
        idx1 = np.minimum(idx0 + 1, self.num_colors - 1)
        frac = indices - idx0

        # Interpolate colors
        original_shape = t.shape
        idx0_flat = idx0.flatten()
        idx1_flat = idx1.flatten()
        frac_flat = frac.flatten().reshape(-1, 1)

        colors = self.colors[idx0_flat] * (1 - frac_flat) + self.colors[idx1_flat] * frac_flat

        # Reshape to match input
        if is_scalar:
            return colors[0]
        else:
            return colors.reshape(*original_shape, 3)

    def __repr__(self) -> str:
        return f"Palette(name='{self.name}', num_colors={self.num_colors})"


class PaletteOperations:
    """Namespace for palette operations (accessed as 'palette' in DSL)."""

    # ============================================================================
    # LAYER 1: PALETTE CREATION
    # ============================================================================

    @staticmethod
    @operator(
        domain="palette",
        category=OpCategory.CONSTRUCT,
        signature="(colors: List[Tuple[float, float, float]], name: str) -> Palette",
        deterministic=True,
        doc="Create palette from list of RGB colors"
    )
    def from_colors(colors: List[Tuple[float, float, float]], name: str = "custom") -> Palette:
        """Create palette from list of RGB colors.

        Args:
            colors: List of (r, g, b) tuples, values in [0, 1]
            name: Palette name

        Returns:
            Palette object

        Example:
            >>> pal = palette.from_colors([(0, 0, 0), (1, 0, 0), (1, 1, 1)])
        """
        colors_array = np.array(colors, dtype=np.float32)
        return Palette(colors_array, name)

    @staticmethod
    @operator(
        domain="palette",
        category=OpCategory.CONSTRUCT,
        signature="(stops: List[Tuple[float, Tuple[float, float, float]]], resolution: int, name: str) -> Palette",
        deterministic=True,
        doc="Create palette from color gradient stops"
    )
    def from_gradient(stops: List[Tuple[float, Tuple[float, float, float]]],
                     resolution: int = 256,
                     name: str = "gradient") -> Palette:
        """Create palette from color gradient stops.

        Args:
            stops: List of (position, (r, g, b)) tuples
                  position in [0, 1], RGB in [0, 1]
            resolution: Number of colors in final palette
            name: Palette name

        Returns:
            Palette object

        Example:
            >>> pal = palette.from_gradient([
            ...     (0.0, (0, 0, 0)),      # Black at start
            ...     (0.5, (1, 0, 0)),      # Red at middle
            ...     (1.0, (1, 1, 1))       # White at end
            ... ])
        """
        if len(stops) < 2:
            raise ValueError("Need at least 2 gradient stops")

        # Sort stops by position
        stops = sorted(stops, key=lambda x: x[0])

        # Generate colors
        colors = np.zeros((resolution, 3), dtype=np.float32)

        for i in range(resolution):
            t = i / (resolution - 1)

            # Find surrounding stops
            for j in range(len(stops) - 1):
                pos0, color0 = stops[j]
                pos1, color1 = stops[j + 1]

                if pos0 <= t <= pos1:
                    # Interpolate between stops
                    if pos1 > pos0:
                        local_t = (t - pos0) / (pos1 - pos0)
                    else:
                        local_t = 0.0

                    color0 = np.array(color0, dtype=np.float32)
                    color1 = np.array(color1, dtype=np.float32)
                    colors[i] = color0 * (1 - local_t) + color1 * local_t
                    break

        return Palette(colors, name)

    @staticmethod
    @operator(
        domain="palette",
        category=OpCategory.CONSTRUCT,
        signature="(resolution: int) -> Palette",
        deterministic=True,
        doc="Create greyscale palette"
    )
    def greyscale(resolution: int = 256) -> Palette:
        """Create greyscale palette.

        Args:
            resolution: Number of colors

        Returns:
            Palette from black to white
        """
        t = np.linspace(0, 1, resolution, dtype=np.float32)
        colors = np.stack([t, t, t], axis=1)
        return Palette(colors, "greyscale")

    @staticmethod
    @operator(
        domain="palette",
        category=OpCategory.CONSTRUCT,
        signature="(resolution: int) -> Palette",
        deterministic=True,
        doc="Create rainbow palette (HSV hue sweep)"
    )
    def rainbow(resolution: int = 256) -> Palette:
        """Create rainbow palette (HSV hue sweep).

        Args:
            resolution: Number of colors

        Returns:
            Palette with full hue range
        """
        colors = np.zeros((resolution, 3), dtype=np.float32)

        for i in range(resolution):
            hue = i / resolution

            # HSV to RGB (S=1, V=1)
            h = hue * 6.0
            c = 1.0
            x = 1.0 - abs((h % 2) - 1.0)

            if h < 1:
                r, g, b = c, x, 0
            elif h < 2:
                r, g, b = x, c, 0
            elif h < 3:
                r, g, b = 0, c, x
            elif h < 4:
                r, g, b = 0, x, c
            elif h < 5:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x

            colors[i] = [r, g, b]

        return Palette(colors, "rainbow")

    @staticmethod
    @operator(
        domain="palette",
        category=OpCategory.CONSTRUCT,
        signature="(resolution: int) -> Palette",
        deterministic=True,
        doc="Create HSV color wheel palette"
    )
    def hsv_wheel(resolution: int = 256) -> Palette:
        """Create HSV color wheel palette.

        Alias for rainbow() - provides intuitive name for HSV hue sweep.

        Args:
            resolution: Number of colors

        Returns:
            Palette with full HSV hue range
        """
        return PaletteOperations.rainbow(resolution)

    # ============================================================================
    # SCIENTIFIC COLORMAPS (Matplotlib-inspired)
    # ============================================================================

    @staticmethod
    @operator(
        domain="palette",
        category=OpCategory.CONSTRUCT,
        signature="(resolution: int) -> Palette",
        deterministic=True,
        doc="Create Inferno colormap"
    )
    def inferno(resolution: int = 256) -> Palette:
        """Create Inferno colormap.

        Perceptually uniform, colorblind-friendly colormap.
        Black → Purple → Orange → Yellow

        Args:
            resolution: Number of colors

        Returns:
            Inferno palette
        """
        # Inferno gradient stops (approximate)
        stops = [
            (0.0, (0.0, 0.0, 0.0)),
            (0.2, (0.2, 0.0, 0.3)),
            (0.4, (0.5, 0.1, 0.4)),
            (0.6, (0.8, 0.3, 0.2)),
            (0.8, (1.0, 0.6, 0.2)),
            (1.0, (1.0, 1.0, 0.6))
        ]
        return PaletteOperations.from_gradient(stops, resolution, "inferno")

    @staticmethod
    @operator(
        domain="palette",
        category=OpCategory.CONSTRUCT,
        signature="(resolution: int) -> Palette",
        deterministic=True,
        doc="Create Viridis colormap"
    )
    def viridis(resolution: int = 256) -> Palette:
        """Create Viridis colormap.

        Perceptually uniform, colorblind-friendly colormap.
        Purple → Teal → Green → Yellow

        Args:
            resolution: Number of colors

        Returns:
            Viridis palette
        """
        # Viridis gradient stops (approximate)
        stops = [
            (0.0, (0.27, 0.00, 0.33)),
            (0.25, (0.28, 0.27, 0.52)),
            (0.5, (0.13, 0.57, 0.55)),
            (0.75, (0.37, 0.79, 0.38)),
            (1.0, (0.99, 0.91, 0.15))
        ]
        return PaletteOperations.from_gradient(stops, resolution, "viridis")

    @staticmethod
    @operator(
        domain="palette",
        category=OpCategory.CONSTRUCT,
        signature="(resolution: int) -> Palette",
        deterministic=True,
        doc="Create Plasma colormap"
    )
    def plasma(resolution: int = 256) -> Palette:
        """Create Plasma colormap.

        Perceptually uniform, colorblind-friendly colormap.
        Purple → Pink → Orange → Yellow

        Args:
            resolution: Number of colors

        Returns:
            Plasma palette
        """
        # Plasma gradient stops (approximate)
        stops = [
            (0.0, (0.05, 0.03, 0.53)),
            (0.25, (0.54, 0.07, 0.56)),
            (0.5, (0.87, 0.22, 0.40)),
            (0.75, (0.99, 0.55, 0.24)),
            (1.0, (0.94, 0.98, 0.13))
        ]
        return PaletteOperations.from_gradient(stops, resolution, "plasma")

    @staticmethod
    @operator(
        domain="palette",
        category=OpCategory.CONSTRUCT,
        signature="(resolution: int) -> Palette",
        deterministic=True,
        doc="Create Magma colormap"
    )
    def magma(resolution: int = 256) -> Palette:
        """Create Magma colormap.

        Perceptually uniform, colorblind-friendly colormap.
        Black → Purple → Red → White

        Args:
            resolution: Number of colors

        Returns:
            Magma palette
        """
        # Magma gradient stops (approximate)
        stops = [
            (0.0, (0.0, 0.0, 0.0)),
            (0.25, (0.28, 0.16, 0.47)),
            (0.5, (0.67, 0.17, 0.53)),
            (0.75, (0.99, 0.50, 0.38)),
            (1.0, (1.0, 0.99, 0.75))
        ]
        return PaletteOperations.from_gradient(stops, resolution, "magma")

    # ============================================================================
    # PROCEDURAL PALETTES
    # ============================================================================

    @staticmethod
    @operator(
        domain="palette",
        category=OpCategory.CONSTRUCT,
        signature="(a: Tuple[float, float, float], b: Tuple[float, float, float], c: Tuple[float, float, float], d: Tuple[float, float, float], resolution: int) -> Palette",
        deterministic=True,
        doc="Create palette using Inigo Quilez's cosine gradient formula"
    )
    def cosine(a: Tuple[float, float, float] = (0.5, 0.5, 0.5),
               b: Tuple[float, float, float] = (0.5, 0.5, 0.5),
               c: Tuple[float, float, float] = (1.0, 1.0, 1.0),
               d: Tuple[float, float, float] = (0.0, 0.33, 0.67),
               resolution: int = 256) -> Palette:
        """Create palette using Inigo Quilez's cosine gradient formula.

        Generates smooth, procedural color gradients using:
        color(t) = a + b * cos(2π * (c * t + d))

        This technique is widely used in shader programming.

        Args:
            a: DC offset for RGB channels
            b: Amplitude for RGB channels
            c: Frequency for RGB channels
            d: Phase for RGB channels
            resolution: Number of colors

        Returns:
            Procedurally generated palette

        Example:
            >>> # Warm sunset palette
            >>> pal = palette.cosine(
            ...     a=(0.5, 0.5, 0.5),
            ...     b=(0.5, 0.5, 0.5),
            ...     c=(1.0, 1.0, 1.0),
            ...     d=(0.0, 0.10, 0.20)
            ... )
        """
        a = np.array(a, dtype=np.float32)
        b = np.array(b, dtype=np.float32)
        c = np.array(c, dtype=np.float32)
        d = np.array(d, dtype=np.float32)

        t = np.linspace(0, 1, resolution, dtype=np.float32).reshape(-1, 1)

        # Cosine gradient formula
        colors = a + b * np.cos(2.0 * np.pi * (c * t + d))

        # Clamp to [0, 1]
        colors = np.clip(colors, 0.0, 1.0)

        return Palette(colors, "cosine")

    @staticmethod
    @operator(
        domain="palette",
        category=OpCategory.CONSTRUCT,
        signature="(resolution: int) -> Palette",
        deterministic=True,
        doc="Create fire palette"
    )
    def fire(resolution: int = 256) -> Palette:
        """Create fire palette (black → red → orange → yellow → white).

        Args:
            resolution: Number of colors

        Returns:
            Fire palette
        """
        stops = [
            (0.0, (0.0, 0.0, 0.0)),
            (0.25, (0.5, 0.0, 0.0)),
            (0.5, (1.0, 0.3, 0.0)),
            (0.75, (1.0, 0.8, 0.0)),
            (1.0, (1.0, 1.0, 1.0))
        ]
        return PaletteOperations.from_gradient(stops, resolution, "fire")

    @staticmethod
    @operator(
        domain="palette",
        category=OpCategory.CONSTRUCT,
        signature="(resolution: int) -> Palette",
        deterministic=True,
        doc="Create ice palette"
    )
    def ice(resolution: int = 256) -> Palette:
        """Create ice palette (black → blue → cyan → white).

        Args:
            resolution: Number of colors

        Returns:
            Ice palette
        """
        stops = [
            (0.0, (0.0, 0.0, 0.0)),
            (0.33, (0.0, 0.0, 0.5)),
            (0.67, (0.0, 0.5, 1.0)),
            (1.0, (1.0, 1.0, 1.0))
        ]
        return PaletteOperations.from_gradient(stops, resolution, "ice")

    # ============================================================================
    # LAYER 2: PALETTE TRANSFORMATIONS
    # ============================================================================

    @staticmethod
    @operator(
        domain="palette",
        category=OpCategory.TRANSFORM,
        signature="(pal: Palette, amount: float) -> Palette",
        deterministic=True,
        doc="Shift palette colors cyclically"
    )
    def shift(pal: Palette, amount: float) -> Palette:
        """Shift palette colors cyclically.

        Args:
            pal: Input palette
            amount: Shift amount (0 to 1, wraps around)

        Returns:
            Shifted palette

        Example:
            >>> shifted = palette.shift(pal, 0.25)  # Shift by 25%
        """
        shift_idx = int(amount * pal.num_colors) % pal.num_colors
        colors = np.roll(pal.colors, shift_idx, axis=0)
        return Palette(colors, f"{pal.name}_shifted")

    @staticmethod
    @operator(
        domain="palette",
        category=OpCategory.TRANSFORM,
        signature="(pal: Palette, speed: float, time: float) -> Palette",
        deterministic=True,
        doc="Cycle palette colors over time"
    )
    def cycle(pal: Palette, speed: float, time: float) -> Palette:
        """Cycle palette colors over time.

        Convenience wrapper for animated palette shifting.

        Args:
            pal: Input palette
            speed: Cycles per time unit
            time: Current time

        Returns:
            Cycled palette
        """
        amount = (speed * time) % 1.0
        return PaletteOperations.shift(pal, amount)

    @staticmethod
    @operator(
        domain="palette",
        category=OpCategory.TRANSFORM,
        signature="(pal: Palette) -> Palette",
        deterministic=True,
        doc="Flip palette (reverse color order)"
    )
    def flip(pal: Palette) -> Palette:
        """Flip palette (reverse color order).

        Args:
            pal: Input palette

        Returns:
            Flipped palette
        """
        colors = pal.colors[::-1].copy()
        return Palette(colors, f"{pal.name}_flipped")

    @staticmethod
    @operator(
        domain="palette",
        category=OpCategory.TRANSFORM,
        signature="(pal: Palette) -> Palette",
        deterministic=True,
        doc="Reverse palette (alias for flip)"
    )
    def reverse(pal: Palette) -> Palette:
        """Reverse palette (alias for flip).

        Args:
            pal: Input palette

        Returns:
            Reversed palette
        """
        return PaletteOperations.flip(pal)

    @staticmethod
    @operator(
        domain="palette",
        category=OpCategory.TRANSFORM,
        signature="(pal_a: Palette, pal_b: Palette, t: float) -> Palette",
        deterministic=True,
        doc="Interpolate between two palettes"
    )
    def lerp(pal_a: Palette, pal_b: Palette, t: float) -> Palette:
        """Interpolate between two palettes.

        Args:
            pal_a: First palette
            pal_b: Second palette
            t: Interpolation factor (0 = pal_a, 1 = pal_b)

        Returns:
            Interpolated palette

        Note:
            Both palettes must have the same number of colors.
        """
        if pal_a.num_colors != pal_b.num_colors:
            raise ValueError(f"Palettes must have same size: {pal_a.num_colors} vs {pal_b.num_colors}")

        t = np.clip(t, 0.0, 1.0)
        colors = pal_a.colors * (1 - t) + pal_b.colors * t

        return Palette(colors, f"lerp_{pal_a.name}_{pal_b.name}")

    @staticmethod
    @operator(
        domain="palette",
        category=OpCategory.TRANSFORM,
        signature="(pal: Palette, factor: float) -> Palette",
        deterministic=True,
        doc="Adjust palette saturation"
    )
    def saturate(pal: Palette, factor: float) -> Palette:
        """Adjust palette saturation.

        Args:
            pal: Input palette
            factor: Saturation multiplier (0 = greyscale, 1 = original, >1 = boosted)

        Returns:
            Saturated palette
        """
        # Convert to HSV, scale saturation, convert back
        colors = pal.colors.copy()

        for i in range(len(colors)):
            r, g, b = colors[i]

            # RGB to HSV
            max_c = max(r, g, b)
            min_c = min(r, g, b)
            delta = max_c - min_c

            # Value
            v = max_c

            # Saturation
            s = delta / max_c if max_c > 0 else 0

            # Hue
            if delta == 0:
                h = 0
            elif max_c == r:
                h = ((g - b) / delta) % 6
            elif max_c == g:
                h = (b - r) / delta + 2
            else:
                h = (r - g) / delta + 4
            h /= 6.0

            # Scale saturation
            s = np.clip(s * factor, 0, 1)

            # HSV to RGB
            c = v * s
            x = c * (1 - abs((h * 6) % 2 - 1))
            m = v - c

            h6 = h * 6
            if h6 < 1:
                r, g, b = c, x, 0
            elif h6 < 2:
                r, g, b = x, c, 0
            elif h6 < 3:
                r, g, b = 0, c, x
            elif h6 < 4:
                r, g, b = 0, x, c
            elif h6 < 5:
                r, g, b = x, 0, c
            else:
                r, g, b = c, 0, x

            colors[i] = [r + m, g + m, b + m]

        return Palette(colors, f"{pal.name}_saturated")

    @staticmethod
    @operator(
        domain="palette",
        category=OpCategory.TRANSFORM,
        signature="(pal: Palette, factor: float) -> Palette",
        deterministic=True,
        doc="Adjust palette brightness"
    )
    def brightness(pal: Palette, factor: float) -> Palette:
        """Adjust palette brightness.

        Args:
            pal: Input palette
            factor: Brightness multiplier

        Returns:
            Adjusted palette
        """
        colors = np.clip(pal.colors * factor, 0, 1)
        return Palette(colors, f"{pal.name}_brightness")

    # ============================================================================
    # LAYER 3: PALETTE APPLICATION
    # ============================================================================

    @staticmethod
    @operator(
        domain="palette",
        category=OpCategory.TRANSFORM,
        signature="(pal: Palette, field: ndarray, vmin: Optional[float], vmax: Optional[float]) -> ndarray",
        deterministic=True,
        doc="Map scalar field to RGB using palette"
    )
    def map(pal: Palette, field: np.ndarray, vmin: Optional[float] = None,
            vmax: Optional[float] = None) -> np.ndarray:
        """Map scalar field to RGB using palette.

        Args:
            pal: Palette to use for mapping
            field: Scalar field (any shape)
            vmin: Minimum value (defaults to field min)
            vmax: Maximum value (defaults to field max)

        Returns:
            RGB image with shape (*field.shape, 3)

        Example:
            >>> noise_field = noise.perlin2d((256, 256))
            >>> rgb = palette.map(pal, noise_field.data)
        """
        if vmin is None:
            vmin = np.min(field)
        if vmax is None:
            vmax = np.max(field)

        # Normalize to [0, 1]
        if vmax > vmin:
            normalized = (field - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(field)

        # Sample palette
        return pal.sample(normalized)

    @staticmethod
    @operator(
        domain="palette",
        category=OpCategory.TRANSFORM,
        signature="(pal: Palette, field: ndarray, frequency: float) -> ndarray",
        deterministic=True,
        doc="Map scalar field to palette with cyclic wrapping"
    )
    def map_cyclic(pal: Palette, field: np.ndarray, frequency: float = 1.0) -> np.ndarray:
        """Map scalar field to palette with cyclic wrapping.

        Useful for phase visualization, angle fields, etc.

        Args:
            pal: Palette to use
            field: Scalar field
            frequency: Number of palette cycles across field range

        Returns:
            RGB image
        """
        # Wrap to [0, 1] with frequency
        normalized = (field * frequency) % 1.0
        return pal.sample(normalized)


# Create singleton instance for use as 'palette' namespace
palette = PaletteOperations()

# Export operators for domain registry discovery
from_colors = PaletteOperations.from_colors
from_gradient = PaletteOperations.from_gradient
lerp = PaletteOperations.lerp
map = PaletteOperations.map
map_cyclic = PaletteOperations.map_cyclic
reverse = PaletteOperations.reverse
flip = PaletteOperations.flip
shift = PaletteOperations.shift
cycle = PaletteOperations.cycle
brightness = PaletteOperations.brightness
saturate = PaletteOperations.saturate
cosine = PaletteOperations.cosine
greyscale = PaletteOperations.greyscale
rainbow = PaletteOperations.rainbow
viridis = PaletteOperations.viridis
plasma = PaletteOperations.plasma
inferno = PaletteOperations.inferno
magma = PaletteOperations.magma
fire = PaletteOperations.fire
ice = PaletteOperations.ice
hsv_wheel = PaletteOperations.hsv_wheel
