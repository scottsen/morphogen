"""
Cross-Domain Type Validation

Validates type compatibility and data flow correctness across domain boundaries.

Level 3 Enhancements:
- Physical unit compatibility checking
- Rate compatibility validation
- Dimensional analysis for cross-domain flows
"""

from typing import Any, Dict, List, Optional, TYPE_CHECKING
import numpy as np

# Import type system components
if TYPE_CHECKING:
    from morphogen.ast.types import Type, StreamType, Rate

# Import unit and rate checking
try:
    from morphogen.types.units import check_unit_compatibility, parse_unit, Unit
    from morphogen.types.rate_compat import (
        validate_rate_compatibility,
        Rate,
        RateCompatibilityError,
        recommend_conversion_operator
    )
    _HAS_UNITS = True
except ImportError:
    _HAS_UNITS = False
    Rate = None  # type: ignore
    RateCompatibilityError = TypeError  # type: ignore


class CrossDomainTypeError(TypeError):
    """Raised when types are incompatible across domain boundaries."""

    def __init__(self, source_domain: str, target_domain: str, message: str):
        self.source_domain = source_domain
        self.target_domain = target_domain
        super().__init__(
            f"Cross-domain type error ({source_domain} → {target_domain}): {message}"
        )


class CrossDomainValidationError(ValueError):
    """Raised when cross-domain flow validation fails."""

    def __init__(self, source_domain: str, target_domain: str, message: str):
        self.source_domain = source_domain
        self.target_domain = target_domain
        super().__init__(
            f"Cross-domain validation error ({source_domain} → {target_domain}): {message}"
        )


def validate_cross_domain_flow(
    source_domain: str,
    target_domain: str,
    source_data: Any,
    interface_class: Optional[Any] = None
) -> bool:
    """
    Validate a cross-domain data flow.

    Args:
        source_domain: Source domain name
        target_domain: Target domain name
        source_data: Data from source domain
        interface_class: Optional DomainInterface class to use

    Returns:
        True if flow is valid

    Raises:
        CrossDomainTypeError: If types are incompatible
        CrossDomainValidationError: If validation fails
    """
    # If interface_class not provided, look it up
    if interface_class is None:
        from .registry import CrossDomainRegistry
        interface_class = CrossDomainRegistry.get(source_domain, target_domain)

    # Create interface instance
    interface = interface_class(source_data=source_data)

    # Run validation
    try:
        return interface.validate()
    except TypeError as e:
        raise CrossDomainTypeError(source_domain, target_domain, str(e))
    except ValueError as e:
        raise CrossDomainValidationError(source_domain, target_domain, str(e))


def validate_field_data(data: Any, allow_vector: bool = True) -> bool:
    """
    Validate field domain data.

    Args:
        data: Field data (numpy array or object with .data attribute)
        allow_vector: If True, allow multi-channel fields

    Returns:
        True if valid

    Raises:
        TypeError: If data is not a valid field
    """
    # Extract numpy array
    if hasattr(data, 'data'):
        arr = data.data
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        raise TypeError(f"Field data must be numpy array or have .data attribute, got {type(data)}")

    # Check dimensions
    if arr.ndim < 2:
        raise TypeError(f"Field must be at least 2D, got {arr.ndim}D")

    if not allow_vector and arr.ndim > 2:
        raise TypeError(f"Scalar field required, got {arr.ndim}D array")

    return True


def validate_agent_positions(positions: np.ndarray, ndim: int = 2) -> bool:
    """
    Validate agent positions array.

    Args:
        positions: Nx2 or Nx3 array of agent positions
        ndim: Expected spatial dimensions (2 or 3)

    Returns:
        True if valid

    Raises:
        TypeError: If positions are invalid
    """
    if not isinstance(positions, np.ndarray):
        raise TypeError(f"Agent positions must be numpy array, got {type(positions)}")

    if positions.ndim != 2:
        raise TypeError(f"Agent positions must be 2D (N x {ndim}), got shape {positions.shape}")

    if positions.shape[1] != ndim:
        raise TypeError(
            f"Agent positions must be N x {ndim}, got shape {positions.shape}"
        )

    return True


def validate_audio_params(params: Dict[str, np.ndarray]) -> bool:
    """
    Validate audio parameter dictionary.

    Args:
        params: Dict with keys like 'triggers', 'amplitudes', 'frequencies'

    Returns:
        True if valid

    Raises:
        TypeError: If params are invalid
    """
    required_keys = ['triggers']
    optional_keys = ['amplitudes', 'frequencies', 'positions', 'durations']

    # Check required keys
    for key in required_keys:
        if key not in params:
            raise TypeError(f"Audio params missing required key: {key}")

    # Validate array types
    for key, value in params.items():
        if key not in required_keys + optional_keys:
            raise TypeError(f"Unknown audio parameter: {key}")

        if not isinstance(value, (list, np.ndarray)):
            raise TypeError(f"Audio param '{key}' must be list or array, got {type(value)}")

    return True


def check_dimensional_compatibility(
    field_shape: tuple,
    positions: np.ndarray
) -> bool:
    """
    Check if field shape and agent positions are compatible.

    Args:
        field_shape: Shape of field (H, W) or (H, W, C)
        positions: Agent positions array

    Returns:
        True if compatible

    Raises:
        ValueError: If dimensions don't match
    """
    ndim = positions.shape[1]  # Spatial dimensions from positions

    if len(field_shape) < 2:
        raise ValueError(f"Field shape must be at least 2D, got {field_shape}")

    # Check field spatial dims match position dims
    field_spatial_dims = 2 if len(field_shape) <= 3 else 3

    if ndim != field_spatial_dims:
        raise ValueError(
            f"Field is {field_spatial_dims}D but positions are {ndim}D"
        )

    return True


def validate_mapping(
    mapping: Dict[str, str],
    valid_source_props: List[str],
    valid_target_params: List[str]
) -> bool:
    """
    Validate a property mapping dict.

    Args:
        mapping: Dict mapping source properties to target parameters
        valid_source_props: List of valid source property names
        valid_target_params: List of valid target parameter names

    Returns:
        True if valid

    Raises:
        ValueError: If mapping contains invalid properties
    """
    for source_prop, target_param in mapping.items():
        if source_prop not in valid_source_props:
            raise ValueError(
                f"Invalid source property '{source_prop}'. "
                f"Valid options: {valid_source_props}"
            )

        if target_param not in valid_target_params:
            raise ValueError(
                f"Invalid target parameter '{target_param}'. "
                f"Valid options: {valid_target_params}"
            )

    return True


# Level 3: Unit Compatibility Validation

def validate_unit_compatibility(
    source_unit: Optional[str],
    target_unit: Optional[str],
    source_domain: str,
    target_domain: str
) -> bool:
    """
    Validate physical unit compatibility across domain boundaries.

    Args:
        source_unit: Source domain unit (e.g., "m/s")
        target_unit: Target domain unit (e.g., "m/s")
        source_domain: Source domain name
        target_domain: Target domain name

    Returns:
        True if units are compatible

    Raises:
        CrossDomainTypeError: If units are incompatible

    Examples:
        >>> validate_unit_compatibility("m/s", "m/s", "field", "agents")
        True

        >>> validate_unit_compatibility("m/s", "m", "field", "agents")
        CrossDomainTypeError: Unit mismatch (field → agents): m/s vs m
    """
    if not _HAS_UNITS:
        # Fallback to string comparison if units module not available
        if source_unit != target_unit:
            raise CrossDomainTypeError(
                source_domain,
                target_domain,
                f"Unit mismatch: {source_unit} vs {target_unit}"
            )
        return True

    # Check dimensional compatibility
    if not check_unit_compatibility(source_unit, target_unit):
        # Parse units to get dimensional information
        try:
            source_parsed = parse_unit(source_unit) if source_unit else None
            target_parsed = parse_unit(target_unit) if target_unit else None

            source_dims = source_parsed.dimensions if source_parsed else "dimensionless"
            target_dims = target_parsed.dimensions if target_parsed else "dimensionless"

            raise CrossDomainTypeError(
                source_domain,
                target_domain,
                f"Incompatible units: {source_unit} [{source_dims}] vs {target_unit} [{target_dims}]"
            )
        except ValueError as e:
            raise CrossDomainTypeError(
                source_domain,
                target_domain,
                f"Invalid unit expression: {e}"
            )

    return True


def validate_rate_compatibility_cross_domain(
    source_rate: Optional['Rate'],
    target_rate: Optional['Rate'],
    source_domain: str,
    target_domain: str,
    source_sample_rate: Optional[float] = None,
    target_sample_rate: Optional[float] = None
) -> bool:
    """
    Validate rate compatibility across domain boundaries.

    Args:
        source_rate: Source domain rate
        target_rate: Target domain rate
        source_domain: Source domain name
        target_domain: Target domain name
        source_sample_rate: Source sample rate (for audio)
        target_sample_rate: Target sample rate (for audio)

    Returns:
        True if rates are compatible

    Raises:
        CrossDomainTypeError: If rates are incompatible

    Examples:
        >>> from morphogen.types.rate_compat import Rate
        >>> validate_rate_compatibility_cross_domain(
        ...     Rate.AUDIO, Rate.AUDIO, "audio", "visual"
        ... )
        True

        >>> validate_rate_compatibility_cross_domain(
        ...     Rate.AUDIO, Rate.CONTROL, "audio", "visual"
        ... )
        CrossDomainTypeError: Rate mismatch (audio → visual): audio vs control
    """
    if not _HAS_UNITS or Rate is None:
        return True  # Skip validation if rate module not available

    if source_rate is None or target_rate is None:
        return True  # Allow None rates for backward compatibility

    # Validate using rate compatibility checker
    is_compatible, error_msg = validate_rate_compatibility(
        source_rate,
        target_rate,
        source_sample_rate,
        target_sample_rate
    )

    if not is_compatible:
        raise CrossDomainTypeError(
            source_domain,
            target_domain,
            f"Rate incompatibility: {error_msg}"
        )

    return True


def validate_type_with_units(
    source_type: 'Type',
    target_type: 'Type',
    source_domain: str,
    target_domain: str
) -> bool:
    """
    Validate complete type compatibility including units and rates.

    Args:
        source_type: Source type object
        target_type: Target type object
        source_domain: Source domain name
        target_domain: Target domain name

    Returns:
        True if types are fully compatible

    Raises:
        CrossDomainTypeError: If types are incompatible

    Note:
        This is the comprehensive Level 3 type validator that checks:
        - Basic type compatibility
        - Physical unit compatibility
        - Rate compatibility (for stream types)
    """
    # Check if base type categories match (ScalarType, VectorType, StreamType, etc.)
    if type(source_type).__name__ != type(target_type).__name__:
        raise CrossDomainTypeError(
            source_domain,
            target_domain,
            f"Incompatible types: {type(source_type).__name__} vs {type(target_type).__name__}"
        )

    # Check unit compatibility if types have units
    source_unit = getattr(source_type, 'unit', None)
    target_unit = getattr(target_type, 'unit', None)

    if source_unit is not None or target_unit is not None:
        validate_unit_compatibility(
            source_unit,
            target_unit,
            source_domain,
            target_domain
        )

    # Check rate compatibility for stream types
    from morphogen.ast.types import StreamType
    if isinstance(source_type, StreamType) and isinstance(target_type, StreamType):
        validate_rate_compatibility_cross_domain(
            source_type.rate,
            target_type.rate,
            source_domain,
            target_domain,
            source_type.sample_rate,
            target_type.sample_rate
        )

    # Finally check general compatibility (after units/rates are validated)
    if not source_type.is_compatible_with(target_type):
        raise CrossDomainTypeError(
            source_domain,
            target_domain,
            f"Incompatible types: {type(source_type).__name__} vs {type(target_type).__name__}"
        )

    return True


def check_unit_conversion_needed(
    source_unit: Optional[str],
    target_unit: Optional[str]
) -> Optional[float]:
    """
    Check if unit conversion is needed and return conversion factor.

    Args:
        source_unit: Source unit string
        target_unit: Target unit string

    Returns:
        Conversion factor if conversion needed, None if units are identical
        or conversion is not possible

    Examples:
        >>> check_unit_conversion_needed("m", "cm")
        100.0

        >>> check_unit_conversion_needed("m", "m")
        None
    """
    if not _HAS_UNITS:
        return None

    if source_unit == target_unit:
        return None

    if source_unit is None or target_unit is None:
        return None

    try:
        source_parsed = parse_unit(source_unit)
        target_parsed = parse_unit(target_unit)

        if source_parsed.is_compatible_with(target_parsed):
            # Return conversion factor
            return source_parsed.scale / target_parsed.scale
        else:
            return None
    except ValueError:
        return None
