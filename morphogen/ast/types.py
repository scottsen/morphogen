"""Type system definitions for Creative Computation DSL."""

from dataclasses import dataclass
from typing import Optional, List, Union
from enum import Enum

# Import unit checking functionality
try:
    from morphogen.types.units import check_unit_compatibility, get_unit_dimensions
except ImportError:
    # Fallback if units module not available
    def check_unit_compatibility(u1: Optional[str], u2: Optional[str]) -> bool:
        if u1 is None or u2 is None:
            return True
        return u1 == u2

    def get_unit_dimensions(u: Optional[str]):
        return None


class BaseType(Enum):
    """Core type categories in the DSL."""
    # Scalar types
    F32 = "f32"
    F64 = "f64"
    F16 = "f16"
    I32 = "i32"
    I64 = "i64"
    U32 = "u32"
    U64 = "u64"
    BOOL = "bool"

    # Vector types
    VEC2 = "Vec2"
    VEC3 = "Vec3"

    # Field types
    FIELD2D = "Field2D"
    FIELD3D = "Field3D"

    # Agent types
    AGENTS = "Agents"

    # Signal types
    SIGNAL = "Signal"

    # Visual types
    VISUAL = "Visual"

    # Boundary types
    BOUNDARY_SPEC = "BoundarySpec"

    # Link types
    LINK = "Link"

    # Unit type
    UNIT = "Unit"

    # Stream types
    SIG = "Sig"  # Audio-rate stream
    CTL = "Ctl"  # Control-rate stream
    EVT = "Evt"  # Event stream


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
class Type:
    """Base class for all types in the DSL."""
    def __eq__(self, other) -> bool:
        raise NotImplementedError

    def is_compatible_with(self, other: 'Type') -> bool:
        """Check if this type can be used where 'other' is expected."""
        raise NotImplementedError


@dataclass
class ScalarType(Type):
    """Scalar numeric type."""
    base: BaseType
    unit: Optional[str] = None

    def __eq__(self, other) -> bool:
        if not isinstance(other, ScalarType):
            return False
        return self.base == other.base and self.unit == other.unit

    def is_compatible_with(self, other: Type) -> bool:
        if not isinstance(other, ScalarType):
            return False
        # Base type must match
        if self.base != other.base:
            return False
        # Units must be dimensionally compatible
        return check_unit_compatibility(self.unit, other.unit)


@dataclass
class VectorType(Type):
    """Vector type (Vec2, Vec3)."""
    base: BaseType  # VEC2 or VEC3
    element_type: ScalarType
    unit: Optional[str] = None

    def __eq__(self, other) -> bool:
        if not isinstance(other, VectorType):
            return False
        return (self.base == other.base and
                self.element_type == other.element_type and
                self.unit == other.unit)

    def is_compatible_with(self, other: Type) -> bool:
        if not isinstance(other, VectorType):
            return False
        if self.base != other.base:
            return False
        if not self.element_type.is_compatible_with(other.element_type):
            return False
        # Units must be dimensionally compatible
        return check_unit_compatibility(self.unit, other.unit)


@dataclass
class FieldType(Type):
    """Field type (Field2D, Field3D)."""
    base: BaseType  # FIELD2D or FIELD3D
    element_type: Type
    unit: Optional[str] = None

    def __eq__(self, other) -> bool:
        if not isinstance(other, FieldType):
            return False
        return (self.base == other.base and
                self.element_type == other.element_type and
                self.unit == other.unit)

    def is_compatible_with(self, other: Type) -> bool:
        if not isinstance(other, FieldType):
            return False
        if self.base != other.base:
            return False
        if not self.element_type.is_compatible_with(other.element_type):
            return False
        # Units must be dimensionally compatible
        return check_unit_compatibility(self.unit, other.unit)


@dataclass
class RecordType(Type):
    """Record type for agent data structures."""
    fields: dict[str, Type]

    def __eq__(self, other) -> bool:
        if not isinstance(other, RecordType):
            return False
        if set(self.fields.keys()) != set(other.fields.keys()):
            return False
        return all(self.fields[k] == other.fields[k] for k in self.fields)

    def is_compatible_with(self, other: Type) -> bool:
        if not isinstance(other, RecordType):
            return False
        # Check that all required fields in 'other' are present in 'self'
        for field_name, field_type in other.fields.items():
            if field_name not in self.fields:
                return False
            if not self.fields[field_name].is_compatible_with(field_type):
                return False
        return True


@dataclass
class AgentType(Type):
    """Agent collection type."""
    record_type: RecordType

    def __eq__(self, other) -> bool:
        if not isinstance(other, AgentType):
            return False
        return self.record_type == other.record_type

    def is_compatible_with(self, other: Type) -> bool:
        if not isinstance(other, AgentType):
            return False
        return self.record_type.is_compatible_with(other.record_type)


@dataclass
class SignalType(Type):
    """Time-varying signal type."""
    element_type: Type
    unit: Optional[str] = None

    def __eq__(self, other) -> bool:
        if not isinstance(other, SignalType):
            return False
        return (self.element_type == other.element_type and
                self.unit == other.unit)

    def is_compatible_with(self, other: Type) -> bool:
        if not isinstance(other, SignalType):
            return False
        if not self.element_type.is_compatible_with(other.element_type):
            return False
        # Units must be dimensionally compatible
        return check_unit_compatibility(self.unit, other.unit)


@dataclass
class StreamType(Type):
    """Rate-specific stream type for audio and control signals.

    Stream types enforce rate compatibility:
    - Sig (audio-rate): High-frequency sample streams (44.1kHz+)
    - Ctl (control-rate): Low-frequency control signals (~100Hz)
    - Evt (event-rate): Discrete event streams

    Rate mismatches require explicit conversion operators.
    """
    base: BaseType  # SIG, CTL, or EVT
    element_type: Type
    rate: Rate
    sample_rate: Optional[float] = None  # For AUDIO rate, e.g., 44100.0
    unit: Optional[str] = None

    def __eq__(self, other) -> bool:
        if not isinstance(other, StreamType):
            return False
        return (self.base == other.base and
                self.element_type == other.element_type and
                self.rate == other.rate and
                self.sample_rate == other.sample_rate and
                self.unit == other.unit)

    def is_compatible_with(self, other: Type) -> bool:
        if not isinstance(other, StreamType):
            return False
        # Base type must match
        if self.base != other.base:
            return False
        # Element types must be compatible
        if not self.element_type.is_compatible_with(other.element_type):
            return False
        # CRITICAL: Rates must match (no implicit conversion)
        if self.rate != other.rate:
            return False
        # For audio-rate streams, sample rates should match
        if self.rate == Rate.AUDIO and self.sample_rate is not None and other.sample_rate is not None:
            if abs(self.sample_rate - other.sample_rate) > 0.1:
                return False
        # Units must be dimensionally compatible
        return check_unit_compatibility(self.unit, other.unit)


@dataclass
class VisualType(Type):
    """Visual/renderable type (opaque)."""

    def __eq__(self, other) -> bool:
        return isinstance(other, VisualType)

    def is_compatible_with(self, other: Type) -> bool:
        return isinstance(other, VisualType)


@dataclass
class BoundarySpecType(Type):
    """Boundary condition specification type."""

    def __eq__(self, other) -> bool:
        return isinstance(other, BoundarySpecType)

    def is_compatible_with(self, other: Type) -> bool:
        return isinstance(other, BoundarySpecType)


@dataclass
class LinkType(Type):
    """Metadata dependency link type (no runtime cost)."""

    def __eq__(self, other) -> bool:
        return isinstance(other, LinkType)

    def is_compatible_with(self, other: Type) -> bool:
        return isinstance(other, LinkType)


@dataclass
class FunctionType(Type):
    """Function type."""
    param_types: List[Type]
    return_type: Type

    def __eq__(self, other) -> bool:
        if not isinstance(other, FunctionType):
            return False
        return (self.param_types == other.param_types and
                self.return_type == other.return_type)

    def is_compatible_with(self, other: Type) -> bool:
        if not isinstance(other, FunctionType):
            return False
        if len(self.param_types) != len(other.param_types):
            return False
        for p1, p2 in zip(self.param_types, other.param_types):
            if not p1.is_compatible_with(p2):
                return False
        return self.return_type.is_compatible_with(other.return_type)


# Helper functions for common types

def f32(unit: Optional[str] = None) -> ScalarType:
    """Create an f32 scalar type."""
    return ScalarType(BaseType.F32, unit)


def f64(unit: Optional[str] = None) -> ScalarType:
    """Create an f64 scalar type."""
    return ScalarType(BaseType.F64, unit)


def i32() -> ScalarType:
    """Create an i32 scalar type."""
    return ScalarType(BaseType.I32)


def u64() -> ScalarType:
    """Create a u64 scalar type."""
    return ScalarType(BaseType.U64)


def vec2(unit: Optional[str] = None) -> VectorType:
    """Create a Vec2 type."""
    return VectorType(BaseType.VEC2, f32(unit), unit)


def vec3(unit: Optional[str] = None) -> VectorType:
    """Create a Vec3 type."""
    return VectorType(BaseType.VEC3, f32(unit), unit)


def field2d(element_type: Type, unit: Optional[str] = None) -> FieldType:
    """Create a Field2D type."""
    return FieldType(BaseType.FIELD2D, element_type, unit)


def field3d(element_type: Type, unit: Optional[str] = None) -> FieldType:
    """Create a Field3D type."""
    return FieldType(BaseType.FIELD3D, element_type, unit)


def agents(record_type: RecordType) -> AgentType:
    """Create an Agents type."""
    return AgentType(record_type)


def signal(element_type: Type, unit: Optional[str] = None) -> SignalType:
    """Create a Signal type."""
    return SignalType(element_type, unit)


def sig(element_type: Type, sample_rate: float = 44100.0, unit: Optional[str] = None) -> StreamType:
    """Create an audio-rate stream (Sig) type.

    Args:
        element_type: Element type of the stream (typically f32)
        sample_rate: Sample rate in Hz (default: 44100.0)
        unit: Optional physical unit

    Returns:
        StreamType configured for audio-rate processing
    """
    return StreamType(BaseType.SIG, element_type, Rate.AUDIO, sample_rate, unit)


def ctl(element_type: Type, unit: Optional[str] = None) -> StreamType:
    """Create a control-rate stream (Ctl) type.

    Args:
        element_type: Element type of the stream (typically f32)
        unit: Optional physical unit

    Returns:
        StreamType configured for control-rate processing
    """
    return StreamType(BaseType.CTL, element_type, Rate.CONTROL, None, unit)


def evt(element_type: Type) -> StreamType:
    """Create an event stream (Evt) type.

    Args:
        element_type: Type of events in the stream

    Returns:
        StreamType configured for discrete events
    """
    return StreamType(BaseType.EVT, element_type, Rate.EVENT, None, None)
