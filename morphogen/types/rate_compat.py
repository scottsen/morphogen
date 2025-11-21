"""Rate compatibility checking for time-varying values.

This module provides validation and conversion utilities for managing
different execution rates (audio, control, visual, simulation, event).

Rate mismatches can lead to:
- Aliasing (downsampling audio to control rate without filtering)
- Performance issues (upsampling control to audio rate unnecessarily)
- Timing errors (mixing event and continuous streams)

This module enforces explicit rate conversions to prevent these issues.
"""

from typing import Optional, Tuple
from enum import Enum
from dataclasses import dataclass


class Rate(Enum):
    """Execution rates for time-varying values.

    Different rates have different update frequencies:
    - AUDIO: 44.1kHz - 192kHz (typical audio sample rates)
    - CONTROL: 1Hz - 1kHz (control parameters, typically ~100Hz)
    - VISUAL: 30Hz - 144Hz (frame rates for visual rendering)
    - SIMULATION: Variable (physics simulation timestep)
    - EVENT: Irregular (discrete events, no fixed rate)
    """
    AUDIO = "audio"
    CONTROL = "control"
    VISUAL = "visual"
    SIMULATION = "sim"
    EVENT = "event"


@dataclass
class RateInfo:
    """Information about a specific rate.

    Attributes:
        rate: The rate category
        frequency: Update frequency in Hz (None for event rate)
        sample_rate: Specific sample rate for audio (e.g., 44100.0)
    """
    rate: Rate
    frequency: Optional[float] = None
    sample_rate: Optional[float] = None

    def __post_init__(self):
        """Validate rate info consistency."""
        if self.rate == Rate.AUDIO and self.sample_rate is None:
            self.sample_rate = 44100.0  # Default audio sample rate
        if self.rate == Rate.EVENT and self.frequency is not None:
            raise ValueError("Event rate cannot have a fixed frequency")


class RateCompatibilityError(TypeError):
    """Raised when incompatible rates are used without conversion."""

    def __init__(self, source_rate: Rate, target_rate: Rate, message: Optional[str] = None):
        self.source_rate = source_rate
        self.target_rate = target_rate
        default_msg = (
            f"Rate mismatch: cannot use {source_rate.value} stream "
            f"where {target_rate.value} stream is expected. "
            f"Use an explicit rate conversion operator."
        )
        super().__init__(message or default_msg)


def check_rate_compatibility(source: Rate, target: Rate) -> bool:
    """Check if two rates are compatible (identical).

    Args:
        source: Source rate
        target: Target rate

    Returns:
        True if rates match exactly, False otherwise

    Note:
        Rates must match exactly - no implicit conversions.
        Use explicit conversion operators for rate changes.
    """
    return source == target


def check_sample_rate_compatibility(
    source_sr: Optional[float],
    target_sr: Optional[float],
    tolerance: float = 0.1
) -> bool:
    """Check if two sample rates are compatible.

    Args:
        source_sr: Source sample rate in Hz
        target_sr: Target sample rate in Hz
        tolerance: Tolerance for floating-point comparison

    Returns:
        True if sample rates match within tolerance

    Note:
        None sample rates are considered compatible with any rate.
    """
    if source_sr is None or target_sr is None:
        return True
    return abs(source_sr - target_sr) < tolerance


def get_conversion_factor(source: RateInfo, target: RateInfo) -> Optional[float]:
    """Calculate the conversion factor between two rates.

    Args:
        source: Source rate information
        target: Target rate information

    Returns:
        Conversion factor (ratio of frequencies), or None if not applicable

    Examples:
        >>> source = RateInfo(Rate.AUDIO, sample_rate=44100.0)
        >>> target = RateInfo(Rate.CONTROL, frequency=100.0)
        >>> get_conversion_factor(source, target)
        441.0  # 44100 / 100 (downsample ratio)
    """
    source_freq = source.sample_rate or source.frequency
    target_freq = target.sample_rate or target.frequency

    if source_freq is None or target_freq is None:
        return None

    return source_freq / target_freq


def requires_resampling(source: RateInfo, target: RateInfo) -> bool:
    """Check if conversion between rates requires resampling.

    Args:
        source: Source rate information
        target: Target rate information

    Returns:
        True if resampling is required

    Note:
        Resampling is needed when moving between continuous rates
        (audio, control, visual, sim) but not for event streams.
    """
    # Event streams don't resample
    if source.rate == Rate.EVENT or target.rate == Rate.EVENT:
        return False

    # Same rate category doesn't need resampling
    if source.rate == target.rate:
        # But audio at different sample rates might
        if source.rate == Rate.AUDIO:
            return not check_sample_rate_compatibility(
                source.sample_rate, target.sample_rate
            )
        return False

    # Different continuous rates require resampling
    return True


def recommend_conversion_operator(source: Rate, target: Rate) -> str:
    """Recommend an explicit conversion operator for rate mismatch.

    Args:
        source: Source rate
        target: Target rate

    Returns:
        Name of recommended conversion operator

    Examples:
        >>> recommend_conversion_operator(Rate.AUDIO, Rate.CONTROL)
        'audio_to_control'

        >>> recommend_conversion_operator(Rate.CONTROL, Rate.AUDIO)
        'control_to_audio'
    """
    conversions = {
        (Rate.AUDIO, Rate.CONTROL): "audio_to_control",
        (Rate.CONTROL, Rate.AUDIO): "control_to_audio",
        (Rate.AUDIO, Rate.VISUAL): "audio_to_visual",
        (Rate.VISUAL, Rate.AUDIO): "visual_to_audio",
        (Rate.CONTROL, Rate.VISUAL): "control_to_visual",
        (Rate.VISUAL, Rate.CONTROL): "visual_to_control",
        (Rate.SIMULATION, Rate.AUDIO): "sim_to_audio",
        (Rate.AUDIO, Rate.SIMULATION): "audio_to_sim",
        (Rate.SIMULATION, Rate.CONTROL): "sim_to_control",
        (Rate.CONTROL, Rate.SIMULATION): "control_to_sim",
        (Rate.EVENT, Rate.AUDIO): "events_to_audio",
        (Rate.EVENT, Rate.CONTROL): "events_to_control",
    }

    key = (source, target)
    if key in conversions:
        return conversions[key]

    # Generic fallback
    return f"{source.value}_to_{target.value}"


def validate_rate_compatibility(
    source: Rate,
    target: Rate,
    source_sample_rate: Optional[float] = None,
    target_sample_rate: Optional[float] = None
) -> Tuple[bool, Optional[str]]:
    """Validate rate compatibility and provide error message if invalid.

    Args:
        source: Source rate
        target: Target rate
        source_sample_rate: Source sample rate (for audio)
        target_sample_rate: Target sample rate (for audio)

    Returns:
        Tuple of (is_compatible, error_message)
        error_message is None if compatible

    Examples:
        >>> validate_rate_compatibility(Rate.AUDIO, Rate.AUDIO, 44100.0, 44100.0)
        (True, None)

        >>> valid, msg = validate_rate_compatibility(Rate.AUDIO, Rate.CONTROL)
        >>> valid
        False
        >>> print(msg)
        Rate mismatch: audio → control requires explicit conversion.
        Use 'audio_to_control()' operator.
    """
    # Compare by value to handle multiple Rate enum definitions
    source_val = source.value if hasattr(source, 'value') else source
    target_val = target.value if hasattr(target, 'value') else target

    # Check basic rate compatibility
    if source_val != target_val:
        operator = recommend_conversion_operator(source, target)
        msg = (
            f"Rate mismatch: {source_val} → {target_val} requires explicit conversion.\n"
            f"Use '{operator}()' operator."
        )
        return (False, msg)

    # For audio, check sample rate compatibility (compare by value to handle different enum instances)
    if source_val == "audio" and source_sample_rate is not None and target_sample_rate is not None:
        if not check_sample_rate_compatibility(source_sample_rate, target_sample_rate):
            msg = (
                f"Audio sample rate mismatch: {source_sample_rate}Hz → {target_sample_rate}Hz.\n"
                f"Use 'resample({target_sample_rate})' operator."
            )
            return (False, msg)

    return (True, None)


# Typical rate configurations
RATE_CONFIGS = {
    "cd_audio": RateInfo(Rate.AUDIO, sample_rate=44100.0),
    "dvd_audio": RateInfo(Rate.AUDIO, sample_rate=48000.0),
    "hd_audio": RateInfo(Rate.AUDIO, sample_rate=96000.0),
    "uhd_audio": RateInfo(Rate.AUDIO, sample_rate=192000.0),
    "control_slow": RateInfo(Rate.CONTROL, frequency=10.0),
    "control_medium": RateInfo(Rate.CONTROL, frequency=100.0),
    "control_fast": RateInfo(Rate.CONTROL, frequency=1000.0),
    "visual_30fps": RateInfo(Rate.VISUAL, frequency=30.0),
    "visual_60fps": RateInfo(Rate.VISUAL, frequency=60.0),
    "visual_144fps": RateInfo(Rate.VISUAL, frequency=144.0),
    "sim_60hz": RateInfo(Rate.SIMULATION, frequency=60.0),
    "sim_120hz": RateInfo(Rate.SIMULATION, frequency=120.0),
    "events": RateInfo(Rate.EVENT),
}
