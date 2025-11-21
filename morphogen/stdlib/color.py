"""Color operations for manipulation and conversion.

This module provides comprehensive color handling for:
- Color space conversion (RGB, HSV, HSL, etc.)
- Color manipulation (blend, mix, brighten, saturate)
- Blend modes (multiply, screen, overlay, etc.)
- Temperature/blackbody radiation
- Gamma correction
- Procedural coloring for fractals, visualizers, simulations

Supports multiple color representations:
- RGB (Red, Green, Blue)
- HSV (Hue, Saturation, Value)
- HSL (Hue, Saturation, Lightness)
- Hex (#RRGGBB)
- Temperature (Kelvin)
"""

from typing import Tuple, Union, List
import numpy as np

from morphogen.core.operator import operator, OpCategory


class ColorOperations:
    """Namespace for color operations (accessed as 'color' in DSL)."""

    # ============================================================================
    # LAYER 1: COLOR SPACE CONVERSIONS
    # ============================================================================

    @staticmethod
    @operator(
        domain="color",
        category=OpCategory.QUERY,
        signature="(rgb: Union[Tuple[float, float, float], ndarray]) -> Union[Tuple[float, float, float], ndarray]",
        deterministic=True,
        doc="Convert RGB to HSV color space"
    )
    def rgb_to_hsv(rgb: Union[Tuple[float, float, float], np.ndarray]) -> Union[Tuple[float, float, float], np.ndarray]:
        """Convert RGB to HSV color space.

        Args:
            rgb: RGB color(s) - tuple (r,g,b) or array (..., 3), values in [0, 1]

        Returns:
            HSV color(s) - (h,s,v) or (..., 3)
            h in [0, 1], s in [0, 1], v in [0, 1]

        Example:
            >>> hsv = color.rgb_to_hsv((1.0, 0.0, 0.0))  # Red
            >>> # hsv = (0.0, 1.0, 1.0)
        """
        rgb = np.asarray(rgb, dtype=np.float32)
        is_scalar = rgb.ndim == 1

        if is_scalar:
            rgb = rgb.reshape(1, 3)

        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        delta = max_c - min_c

        # Value
        v = max_c

        # Saturation
        s = np.where(max_c > 0, delta / max_c, 0)

        # Hue
        h = np.zeros_like(max_c)

        # Red is max
        mask = (delta > 0) & (max_c == r)
        h = np.where(mask, ((g - b) / delta) % 6, h)

        # Green is max
        mask = (delta > 0) & (max_c == g)
        h = np.where(mask, (b - r) / delta + 2, h)

        # Blue is max
        mask = (delta > 0) & (max_c == b)
        h = np.where(mask, (r - g) / delta + 4, h)

        h = h / 6.0  # Normalize to [0, 1]

        hsv = np.stack([h, s, v], axis=-1)

        if is_scalar:
            return tuple(hsv[0])
        return hsv

    @staticmethod
    @operator(
        domain="color",
        category=OpCategory.QUERY,
        signature="(hsv: Union[Tuple[float, float, float], ndarray]) -> Union[Tuple[float, float, float], ndarray]",
        deterministic=True,
        doc="Convert HSV to RGB color space"
    )
    def hsv_to_rgb(hsv: Union[Tuple[float, float, float], np.ndarray]) -> Union[Tuple[float, float, float], np.ndarray]:
        """Convert HSV to RGB color space.

        Args:
            hsv: HSV color(s) - tuple (h,s,v) or array (..., 3)
                h in [0, 1], s in [0, 1], v in [0, 1]

        Returns:
            RGB color(s) - (r,g,b) or (..., 3), values in [0, 1]

        Example:
            >>> rgb = color.hsv_to_rgb((0.0, 1.0, 1.0))  # Red
            >>> # rgb = (1.0, 0.0, 0.0)
        """
        hsv = np.asarray(hsv, dtype=np.float32)
        is_scalar = hsv.ndim == 1

        if is_scalar:
            hsv = hsv.reshape(1, 3)

        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

        h = h * 6.0  # Scale to [0, 6]
        c = v * s
        x = c * (1 - np.abs((h % 2) - 1))
        m = v - c

        # Initialize RGB
        rgb = np.zeros_like(hsv)

        # Determine RGB based on hue sector
        mask0 = (h >= 0) & (h < 1)
        mask1 = (h >= 1) & (h < 2)
        mask2 = (h >= 2) & (h < 3)
        mask3 = (h >= 3) & (h < 4)
        mask4 = (h >= 4) & (h < 5)
        mask5 = (h >= 5) & (h < 6)

        rgb[..., 0] = np.where(mask0, c, rgb[..., 0])
        rgb[..., 1] = np.where(mask0, x, rgb[..., 1])

        rgb[..., 0] = np.where(mask1, x, rgb[..., 0])
        rgb[..., 1] = np.where(mask1, c, rgb[..., 1])

        rgb[..., 1] = np.where(mask2, c, rgb[..., 1])
        rgb[..., 2] = np.where(mask2, x, rgb[..., 2])

        rgb[..., 1] = np.where(mask3, x, rgb[..., 1])
        rgb[..., 2] = np.where(mask3, c, rgb[..., 2])

        rgb[..., 0] = np.where(mask4, x, rgb[..., 0])
        rgb[..., 2] = np.where(mask4, c, rgb[..., 2])

        rgb[..., 0] = np.where(mask5, c, rgb[..., 0])
        rgb[..., 2] = np.where(mask5, x, rgb[..., 2])

        rgb = rgb + m[..., np.newaxis]

        if is_scalar:
            return tuple(rgb[0])
        return rgb

    @staticmethod
    @operator(
        domain="color",
        category=OpCategory.QUERY,
        signature="(rgb: Union[Tuple[float, float, float], ndarray]) -> Union[Tuple[float, float, float], ndarray]",
        deterministic=True,
        doc="Convert RGB to HSL color space"
    )
    def rgb_to_hsl(rgb: Union[Tuple[float, float, float], np.ndarray]) -> Union[Tuple[float, float, float], np.ndarray]:
        """Convert RGB to HSL color space.

        Args:
            rgb: RGB color(s) - (r,g,b) or (..., 3), values in [0, 1]

        Returns:
            HSL color(s) - (h,s,l) or (..., 3)
            h in [0, 1], s in [0, 1], l in [0, 1]
        """
        rgb = np.asarray(rgb, dtype=np.float32)
        is_scalar = rgb.ndim == 1

        if is_scalar:
            rgb = rgb.reshape(1, 3)

        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        delta = max_c - min_c

        # Lightness
        l = (max_c + min_c) / 2.0

        # Saturation
        s = np.where(delta == 0, 0,
                    np.where(l < 0.5, delta / (max_c + min_c),
                            delta / (2.0 - max_c - min_c)))

        # Hue (same as HSV)
        h = np.zeros_like(max_c)

        mask = (delta > 0) & (max_c == r)
        h = np.where(mask, ((g - b) / delta) % 6, h)

        mask = (delta > 0) & (max_c == g)
        h = np.where(mask, (b - r) / delta + 2, h)

        mask = (delta > 0) & (max_c == b)
        h = np.where(mask, (r - g) / delta + 4, h)

        h = h / 6.0

        hsl = np.stack([h, s, l], axis=-1)

        if is_scalar:
            return tuple(hsl[0])
        return hsl

    @staticmethod
    @operator(
        domain="color",
        category=OpCategory.QUERY,
        signature="(hsl: Union[Tuple[float, float, float], ndarray]) -> Union[Tuple[float, float, float], ndarray]",
        deterministic=True,
        doc="Convert HSL to RGB color space"
    )
    def hsl_to_rgb(hsl: Union[Tuple[float, float, float], np.ndarray]) -> Union[Tuple[float, float, float], np.ndarray]:
        """Convert HSL to RGB color space.

        Args:
            hsl: HSL color(s) - (h,s,l) or (..., 3)
                h in [0, 1], s in [0, 1], l in [0, 1]

        Returns:
            RGB color(s) - (r,g,b) or (..., 3), values in [0, 1]
        """
        hsl = np.asarray(hsl, dtype=np.float32)
        is_scalar = hsl.ndim == 1

        if is_scalar:
            hsl = hsl.reshape(1, 3)

        h, s, l = hsl[..., 0], hsl[..., 1], hsl[..., 2]

        c = (1 - np.abs(2 * l - 1)) * s
        h6 = h * 6.0
        x = c * (1 - np.abs((h6 % 2) - 1))
        m = l - c / 2.0

        # Initialize RGB
        rgb = np.zeros_like(hsl)

        # Determine RGB based on hue sector
        mask0 = (h6 >= 0) & (h6 < 1)
        mask1 = (h6 >= 1) & (h6 < 2)
        mask2 = (h6 >= 2) & (h6 < 3)
        mask3 = (h6 >= 3) & (h6 < 4)
        mask4 = (h6 >= 4) & (h6 < 5)
        mask5 = (h6 >= 5) & (h6 < 6)

        rgb[..., 0] = np.where(mask0, c, rgb[..., 0])
        rgb[..., 1] = np.where(mask0, x, rgb[..., 1])

        rgb[..., 0] = np.where(mask1, x, rgb[..., 0])
        rgb[..., 1] = np.where(mask1, c, rgb[..., 1])

        rgb[..., 1] = np.where(mask2, c, rgb[..., 1])
        rgb[..., 2] = np.where(mask2, x, rgb[..., 2])

        rgb[..., 1] = np.where(mask3, x, rgb[..., 1])
        rgb[..., 2] = np.where(mask3, c, rgb[..., 2])

        rgb[..., 0] = np.where(mask4, x, rgb[..., 0])
        rgb[..., 2] = np.where(mask4, c, rgb[..., 2])

        rgb[..., 0] = np.where(mask5, c, rgb[..., 0])
        rgb[..., 2] = np.where(mask5, x, rgb[..., 2])

        rgb = rgb + m[..., np.newaxis]

        if is_scalar:
            return tuple(rgb[0])
        return rgb

    @staticmethod
    @operator(
        domain="color",
        category=OpCategory.QUERY,
        signature="(hex_color: str) -> Tuple[float, float, float]",
        deterministic=True,
        doc="Convert hex color string to RGB"
    )
    def hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
        """Convert hex color string to RGB.

        Args:
            hex_color: Hex color string ("#RRGGBB" or "RRGGBB")

        Returns:
            RGB tuple (r, g, b) with values in [0, 1]

        Example:
            >>> rgb = color.hex_to_rgb("#FF0000")  # Red
            >>> # rgb = (1.0, 0.0, 0.0)
        """
        hex_color = hex_color.lstrip('#')

        if len(hex_color) != 6:
            raise ValueError(f"Hex color must be 6 characters, got: {hex_color}")

        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0

        return (r, g, b)

    @staticmethod
    @operator(
        domain="color",
        category=OpCategory.QUERY,
        signature="(rgb: Tuple[float, float, float]) -> str",
        deterministic=True,
        doc="Convert RGB to hex color string"
    )
    def rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
        """Convert RGB to hex color string.

        Args:
            rgb: RGB tuple (r, g, b) with values in [0, 1]

        Returns:
            Hex color string "#RRGGBB"

        Example:
            >>> hex_str = color.rgb_to_hex((1.0, 0.0, 0.0))
            >>> # hex_str = "#FF0000"
        """
        r, g, b = rgb
        r_int = int(np.clip(r, 0, 1) * 255)
        g_int = int(np.clip(g, 0, 1) * 255)
        b_int = int(np.clip(b, 0, 1) * 255)

        return f"#{r_int:02X}{g_int:02X}{b_int:02X}"

    @staticmethod
    @operator(
        domain="color",
        category=OpCategory.QUERY,
        signature="(kelvin: Union[float, ndarray]) -> Union[Tuple[float, float, float], ndarray]",
        deterministic=True,
        doc="Convert color temperature (Kelvin) to RGB"
    )
    def temperature_to_rgb(kelvin: Union[float, np.ndarray]) -> Union[Tuple[float, float, float], np.ndarray]:
        """Convert color temperature (Kelvin) to RGB.

        Uses blackbody radiation approximation.
        Useful for fire, stars, lighting effects.

        Args:
            kelvin: Temperature in Kelvin (1000-40000)
                   1850K = candle flame
                   2700K = incandescent bulb
                   5500K = daylight
                   6500K = overcast sky
                   10000K = blue sky

        Returns:
            RGB color(s) in [0, 1]

        Example:
            >>> fire_color = color.temperature_to_rgb(1850)  # Warm orange
            >>> daylight = color.temperature_to_rgb(5500)    # Neutral white
        """
        kelvin = np.asarray(kelvin, dtype=np.float32)
        is_scalar = kelvin.ndim == 0

        if is_scalar:
            kelvin = np.array([kelvin])

        # Temperature in hundreds
        temp = kelvin / 100.0

        # Red
        r = np.where(temp <= 66,
                    255,
                    329.698727446 * ((temp - 60) ** -0.1332047592))

        # Green
        g = np.where(temp <= 66,
                    99.4708025861 * np.log(temp) - 161.1195681661,
                    288.1221695283 * ((temp - 60) ** -0.0755148492))

        # Blue
        b = np.where(temp >= 66, 255,
                    np.where(temp <= 19, 0,
                            138.5177312231 * np.log(temp - 10) - 305.0447927307))

        # Normalize to [0, 1]
        rgb = np.stack([r, g, b], axis=-1) / 255.0
        rgb = np.clip(rgb, 0, 1)

        if is_scalar:
            return tuple(rgb[0])
        return rgb

    # ============================================================================
    # LAYER 2: COLOR MANIPULATION
    # ============================================================================

    @staticmethod
    @operator(
        domain="color",
        category=OpCategory.TRANSFORM,
        signature="(color_a: Union[Tuple[float, float, float], ndarray], color_b: Union[Tuple[float, float, float], ndarray]) -> Union[Tuple[float, float, float], ndarray]",
        deterministic=True,
        doc="Add two colors (clamped to [0, 1])"
    )
    def add(color_a: Union[Tuple[float, float, float], np.ndarray],
            color_b: Union[Tuple[float, float, float], np.ndarray]) -> Union[Tuple[float, float, float], np.ndarray]:
        """Add two colors (clamped to [0, 1]).

        Args:
            color_a: First RGB color(s)
            color_b: Second RGB color(s)

        Returns:
            Sum of colors, clamped to [0, 1]
        """
        a = np.asarray(color_a, dtype=np.float32)
        b = np.asarray(color_b, dtype=np.float32)
        return np.clip(a + b, 0, 1)

    @staticmethod
    @operator(
        domain="color",
        category=OpCategory.TRANSFORM,
        signature="(color_a: Union[Tuple[float, float, float], ndarray], color_b: Union[Tuple[float, float, float], ndarray]) -> Union[Tuple[float, float, float], ndarray]",
        deterministic=True,
        doc="Multiply two colors element-wise"
    )
    def multiply(color_a: Union[Tuple[float, float, float], np.ndarray],
                 color_b: Union[Tuple[float, float, float], np.ndarray]) -> Union[Tuple[float, float, float], np.ndarray]:
        """Multiply two colors element-wise.

        Args:
            color_a: First RGB color(s)
            color_b: Second RGB color(s)

        Returns:
            Product of colors
        """
        a = np.asarray(color_a, dtype=np.float32)
        b = np.asarray(color_b, dtype=np.float32)
        return a * b

    @staticmethod
    @operator(
        domain="color",
        category=OpCategory.TRANSFORM,
        signature="(color_a: Union[Tuple[float, float, float], ndarray], color_b: Union[Tuple[float, float, float], ndarray], t: Union[float, ndarray]) -> Union[Tuple[float, float, float], ndarray]",
        deterministic=True,
        doc="Linear interpolation between two colors"
    )
    def mix(color_a: Union[Tuple[float, float, float], np.ndarray],
            color_b: Union[Tuple[float, float, float], np.ndarray],
            t: Union[float, np.ndarray]) -> Union[Tuple[float, float, float], np.ndarray]:
        """Linear interpolation between two colors.

        Args:
            color_a: First RGB color(s)
            color_b: Second RGB color(s)
            t: Interpolation factor (0 = color_a, 1 = color_b)

        Returns:
            Interpolated color(s)

        Example:
            >>> mixed = color.mix((1, 0, 0), (0, 0, 1), 0.5)  # Purple
        """
        a = np.asarray(color_a, dtype=np.float32)
        b = np.asarray(color_b, dtype=np.float32)
        t = np.asarray(t, dtype=np.float32)

        return a * (1 - t) + b * t

    @staticmethod
    @operator(
        domain="color",
        category=OpCategory.TRANSFORM,
        signature="(rgb: Union[Tuple[float, float, float], ndarray], factor: float) -> Union[Tuple[float, float, float], ndarray]",
        deterministic=True,
        doc="Adjust color brightness"
    )
    def brightness(rgb: Union[Tuple[float, float, float], np.ndarray],
                   factor: float) -> Union[Tuple[float, float, float], np.ndarray]:
        """Adjust color brightness.

        Args:
            rgb: RGB color(s)
            factor: Brightness multiplier (0 = black, 1 = original, >1 = brighter)

        Returns:
            Adjusted color(s), clamped to [0, 1]
        """
        rgb = np.asarray(rgb, dtype=np.float32)
        return np.clip(rgb * factor, 0, 1)

    @staticmethod
    @operator(
        domain="color",
        category=OpCategory.TRANSFORM,
        signature="(rgb: Union[Tuple[float, float, float], ndarray], factor: float) -> Union[Tuple[float, float, float], ndarray]",
        deterministic=True,
        doc="Adjust color saturation"
    )
    def saturate(rgb: Union[Tuple[float, float, float], np.ndarray],
                 factor: float) -> Union[Tuple[float, float, float], np.ndarray]:
        """Adjust color saturation.

        Args:
            rgb: RGB color(s)
            factor: Saturation multiplier (0 = greyscale, 1 = original, >1 = boosted)

        Returns:
            Adjusted color(s)
        """
        # Convert to HSV, scale saturation, convert back
        hsv = ColorOperations.rgb_to_hsv(rgb)
        hsv = np.asarray(hsv, dtype=np.float32)

        if hsv.ndim == 1:
            hsv[1] = np.clip(hsv[1] * factor, 0, 1)
        else:
            hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 1)

        return ColorOperations.hsv_to_rgb(hsv)

    @staticmethod
    @operator(
        domain="color",
        category=OpCategory.TRANSFORM,
        signature="(rgb: Union[Tuple[float, float, float], ndarray], gamma: float) -> Union[Tuple[float, float, float], ndarray]",
        deterministic=True,
        doc="Apply gamma correction to color"
    )
    def gamma_correct(rgb: Union[Tuple[float, float, float], np.ndarray],
                      gamma: float = 2.2) -> Union[Tuple[float, float, float], np.ndarray]:
        """Apply gamma correction to color.

        Args:
            rgb: RGB color(s) in [0, 1]
            gamma: Gamma value (typically 2.2 for sRGB)

        Returns:
            Gamma-corrected color(s)

        Example:
            >>> linear_rgb = color.gamma_correct(srgb, 2.2)  # sRGB to linear
            >>> srgb = color.gamma_correct(linear_rgb, 1/2.2)  # Linear to sRGB
        """
        rgb = np.asarray(rgb, dtype=np.float32)
        return np.power(np.clip(rgb, 0, 1), gamma)

    # ============================================================================
    # LAYER 3: BLEND MODES
    # ============================================================================

    @staticmethod
    @operator(
        domain="color",
        category=OpCategory.TRANSFORM,
        signature="(base: Union[Tuple[float, float, float], ndarray], blend: Union[Tuple[float, float, float], ndarray]) -> Union[Tuple[float, float, float], ndarray]",
        deterministic=True,
        doc="Overlay blend mode"
    )
    def blend_overlay(base: Union[Tuple[float, float, float], np.ndarray],
                     blend: Union[Tuple[float, float, float], np.ndarray]) -> Union[Tuple[float, float, float], np.ndarray]:
        """Overlay blend mode.

        Combines multiply and screen based on base color.

        Args:
            base: Base RGB color(s)
            blend: Blend RGB color(s)

        Returns:
            Overlaid color(s)
        """
        base = np.asarray(base, dtype=np.float32)
        blend = np.asarray(blend, dtype=np.float32)

        result = np.where(base < 0.5,
                         2 * base * blend,
                         1 - 2 * (1 - base) * (1 - blend))

        return np.clip(result, 0, 1)

    @staticmethod
    @operator(
        domain="color",
        category=OpCategory.TRANSFORM,
        signature="(base: Union[Tuple[float, float, float], ndarray], blend: Union[Tuple[float, float, float], ndarray]) -> Union[Tuple[float, float, float], ndarray]",
        deterministic=True,
        doc="Screen blend mode (inverted multiply)"
    )
    def blend_screen(base: Union[Tuple[float, float, float], np.ndarray],
                    blend: Union[Tuple[float, float, float], np.ndarray]) -> Union[Tuple[float, float, float], np.ndarray]:
        """Screen blend mode (inverted multiply).

        Args:
            base: Base RGB color(s)
            blend: Blend RGB color(s)

        Returns:
            Screened color(s)
        """
        base = np.asarray(base, dtype=np.float32)
        blend = np.asarray(blend, dtype=np.float32)

        return 1 - (1 - base) * (1 - blend)

    @staticmethod
    @operator(
        domain="color",
        category=OpCategory.TRANSFORM,
        signature="(base: Union[Tuple[float, float, float], ndarray], blend: Union[Tuple[float, float, float], ndarray]) -> Union[Tuple[float, float, float], ndarray]",
        deterministic=True,
        doc="Multiply blend mode"
    )
    def blend_multiply(base: Union[Tuple[float, float, float], np.ndarray],
                      blend: Union[Tuple[float, float, float], np.ndarray]) -> Union[Tuple[float, float, float], np.ndarray]:
        """Multiply blend mode.

        Args:
            base: Base RGB color(s)
            blend: Blend RGB color(s)

        Returns:
            Multiplied color(s)
        """
        return ColorOperations.multiply(base, blend)

    @staticmethod
    @operator(
        domain="color",
        category=OpCategory.TRANSFORM,
        signature="(base: Union[Tuple[float, float, float], ndarray], blend: Union[Tuple[float, float, float], ndarray]) -> Union[Tuple[float, float, float], ndarray]",
        deterministic=True,
        doc="Difference blend mode"
    )
    def blend_difference(base: Union[Tuple[float, float, float], np.ndarray],
                        blend: Union[Tuple[float, float, float], np.ndarray]) -> Union[Tuple[float, float, float], np.ndarray]:
        """Difference blend mode.

        Args:
            base: Base RGB color(s)
            blend: Blend RGB color(s)

        Returns:
            Difference color(s)
        """
        base = np.asarray(base, dtype=np.float32)
        blend = np.asarray(blend, dtype=np.float32)

        return np.abs(base - blend)

    @staticmethod
    @operator(
        domain="color",
        category=OpCategory.TRANSFORM,
        signature="(base: Union[Tuple[float, float, float], ndarray], blend: Union[Tuple[float, float, float], ndarray]) -> Union[Tuple[float, float, float], ndarray]",
        deterministic=True,
        doc="Soft light blend mode (gentle overlay)"
    )
    def blend_soft_light(base: Union[Tuple[float, float, float], np.ndarray],
                        blend: Union[Tuple[float, float, float], np.ndarray]) -> Union[Tuple[float, float, float], np.ndarray]:
        """Soft light blend mode (gentle overlay).

        Args:
            base: Base RGB color(s)
            blend: Blend RGB color(s)

        Returns:
            Soft lit color(s)
        """
        base = np.asarray(base, dtype=np.float32)
        blend = np.asarray(blend, dtype=np.float32)

        result = np.where(blend < 0.5,
                         2 * base * blend + base ** 2 * (1 - 2 * blend),
                         2 * base * (1 - blend) + np.sqrt(base) * (2 * blend - 1))

        return np.clip(result, 0, 1)

    # ============================================================================
    # LAYER 4: UTILITY OPERATIONS
    # ============================================================================

    @staticmethod
    @operator(
        domain="color",
        category=OpCategory.TRANSFORM,
        signature="(rgb: Union[Tuple[float, float, float], ndarray], levels: int) -> Union[Tuple[float, float, float], ndarray]",
        deterministic=True,
        doc="Posterize color (reduce to N discrete levels)"
    )
    def posterize(rgb: Union[Tuple[float, float, float], np.ndarray],
                  levels: int) -> Union[Tuple[float, float, float], np.ndarray]:
        """Posterize color (reduce to N discrete levels).

        Args:
            rgb: RGB color(s)
            levels: Number of levels per channel (e.g., 4)

        Returns:
            Posterized color(s)

        Example:
            >>> posterized = color.posterize((0.7, 0.3, 0.9), 4)
        """
        rgb = np.asarray(rgb, dtype=np.float32)

        # Quantize to levels
        quantized = np.floor(rgb * levels) / (levels - 1) if levels > 1 else rgb

        return np.clip(quantized, 0, 1)

    @staticmethod
    @operator(
        domain="color",
        category=OpCategory.TRANSFORM,
        signature="(rgb: Union[Tuple[float, float, float], ndarray], threshold_value: float) -> Union[Tuple[float, float, float], ndarray]",
        deterministic=True,
        doc="Threshold color to black or white"
    )
    def threshold(rgb: Union[Tuple[float, float, float], np.ndarray],
                  threshold_value: float = 0.5) -> Union[Tuple[float, float, float], np.ndarray]:
        """Threshold color to black or white.

        Args:
            rgb: RGB color(s)
            threshold_value: Threshold for brightness

        Returns:
            Thresholded color(s) (0 or 1)
        """
        rgb = np.asarray(rgb, dtype=np.float32)

        # Compute luminance
        luminance = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]

        # Threshold
        result = np.where(luminance > threshold_value, 1.0, 0.0)

        if rgb.ndim == 1:
            return (result, result, result)
        else:
            return np.stack([result, result, result], axis=-1)


# Create singleton instance for use as 'color' namespace
color = ColorOperations()

# Export operators for domain registry discovery
rgb_to_hsv = ColorOperations.rgb_to_hsv
hsv_to_rgb = ColorOperations.hsv_to_rgb
rgb_to_hsl = ColorOperations.rgb_to_hsl
hsl_to_rgb = ColorOperations.hsl_to_rgb
rgb_to_hex = ColorOperations.rgb_to_hex
hex_to_rgb = ColorOperations.hex_to_rgb
mix = ColorOperations.mix
add = ColorOperations.add
multiply = ColorOperations.multiply
brightness = ColorOperations.brightness
saturate = ColorOperations.saturate
gamma_correct = ColorOperations.gamma_correct
temperature_to_rgb = ColorOperations.temperature_to_rgb
threshold = ColorOperations.threshold
posterize = ColorOperations.posterize
blend_multiply = ColorOperations.blend_multiply
blend_screen = ColorOperations.blend_screen
blend_overlay = ColorOperations.blend_overlay
blend_soft_light = ColorOperations.blend_soft_light
blend_difference = ColorOperations.blend_difference
