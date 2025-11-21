"""Tests for Level 3 cross-domain validators with unit and rate checking."""

import pytest
from morphogen.cross_domain.validators import (
    validate_unit_compatibility,
    validate_rate_compatibility_cross_domain,
    validate_type_with_units,
    check_unit_conversion_needed,
    CrossDomainTypeError,
)
from morphogen.ast.types import (
    f32,
    vec2,
    field2d,
    sig,
    ctl,
    StreamType,
    BaseType,
    Rate,
)


class TestUnitCompatibilityValidation:
    """Test physical unit compatibility across domains."""

    def test_same_units_compatible(self):
        """Same units are compatible."""
        result = validate_unit_compatibility("m/s", "m/s", "field", "agents")
        assert result is True

    def test_none_units_compatible(self):
        """None units are compatible with anything."""
        result = validate_unit_compatibility(None, None, "field", "agents")
        assert result is True

        result = validate_unit_compatibility("m/s", None, "field", "agents")
        assert result is True

        result = validate_unit_compatibility(None, "m/s", "field", "agents")
        assert result is True

    def test_dimensionally_compatible_units(self):
        """Dimensionally compatible units are valid."""
        # m/s and km/s are both velocity
        result = validate_unit_compatibility("m/s", "km/s", "field", "agents")
        assert result is True

    def test_incompatible_units_error(self):
        """Incompatible units raise error."""
        with pytest.raises(CrossDomainTypeError) as exc_info:
            validate_unit_compatibility("m/s", "m", "field", "agents")

        error = exc_info.value
        assert error.source_domain == "field"
        assert error.target_domain == "agents"
        assert "Incompatible units" in str(error)

    def test_invalid_unit_expression_error(self):
        """Invalid unit expressions raise error."""
        with pytest.raises(CrossDomainTypeError) as exc_info:
            validate_unit_compatibility("invalid_unit", "m", "field", "agents")

        assert "Invalid unit expression" in str(exc_info.value) or "Unknown unit" in str(exc_info.value)

    def test_dimensionless_compatible(self):
        """Dimensionless units are compatible."""
        result = validate_unit_compatibility("1", "1", "field", "agents")
        assert result is True


class TestRateCompatibilityValidation:
    """Test rate compatibility across domains."""

    def test_same_rate_compatible(self):
        """Same rates are compatible."""
        result = validate_rate_compatibility_cross_domain(
            Rate.AUDIO, Rate.AUDIO, "audio", "visual"
        )
        assert result is True

    def test_different_rates_incompatible(self):
        """Different rates are incompatible."""
        with pytest.raises(CrossDomainTypeError) as exc_info:
            validate_rate_compatibility_cross_domain(
                Rate.AUDIO, Rate.CONTROL, "audio", "visual"
            )

        error = exc_info.value
        assert error.source_domain == "audio"
        assert error.target_domain == "visual"
        assert "Rate incompatibility" in str(error)

    def test_none_rates_compatible(self):
        """None rates are compatible (backward compatibility)."""
        result = validate_rate_compatibility_cross_domain(
            None, None, "field", "agents"
        )
        assert result is True

        result = validate_rate_compatibility_cross_domain(
            Rate.AUDIO, None, "audio", "field"
        )
        assert result is True

    def test_audio_sample_rate_mismatch(self):
        """Audio sample rate mismatches are detected."""
        with pytest.raises(CrossDomainTypeError) as exc_info:
            validate_rate_compatibility_cross_domain(
                Rate.AUDIO, Rate.AUDIO, "audio", "visual",
                source_sample_rate=44100.0,
                target_sample_rate=48000.0
            )

        error = exc_info.value
        assert "44100" in str(error) or "sample rate" in str(error).lower()


class TestTypeWithUnitsValidation:
    """Test comprehensive type validation with units and rates."""

    def test_scalar_types_same_units(self):
        """Scalar types with same units are compatible."""
        source = f32("m/s")
        target = f32("m/s")

        result = validate_type_with_units(source, target, "field", "agents")
        assert result is True

    def test_scalar_types_compatible_units(self):
        """Scalar types with dimensionally compatible units are valid."""
        source = f32("m/s")
        target = f32("km/s")  # Different scale, same dimensions

        result = validate_type_with_units(source, target, "field", "agents")
        assert result is True

    def test_scalar_types_incompatible_units(self):
        """Scalar types with incompatible units raise error."""
        source = f32("m/s")
        target = f32("kg")

        with pytest.raises(CrossDomainTypeError) as exc_info:
            validate_type_with_units(source, target, "field", "agents")

        assert "Incompatible units" in str(exc_info.value)

    def test_vector_types_with_units(self):
        """Vector types with units are validated."""
        source = vec2("m")
        target = vec2("m")

        result = validate_type_with_units(source, target, "field", "agents")
        assert result is True

    def test_field_types_with_units(self):
        """Field types with units are validated."""
        source = field2d(f32("m/s"), "m/s")
        target = field2d(f32("m/s"), "m/s")

        result = validate_type_with_units(source, target, "field", "visual")
        assert result is True

    def test_stream_types_same_rate(self):
        """Stream types with same rate are compatible."""
        source = sig(f32(), sample_rate=44100.0)
        target = sig(f32(), sample_rate=44100.0)

        result = validate_type_with_units(source, target, "audio", "visual")
        assert result is True

    def test_stream_types_different_rate(self):
        """Stream types with different rates are incompatible."""
        source = sig(f32(), sample_rate=44100.0)
        target = ctl(f32())

        with pytest.raises(CrossDomainTypeError) as exc_info:
            validate_type_with_units(source, target, "audio", "visual")

        error = exc_info.value
        assert "Rate incompatibility" in str(error)

    def test_stream_types_different_sample_rate(self):
        """Stream types with different sample rates are incompatible."""
        source = sig(f32(), sample_rate=44100.0)
        target = sig(f32(), sample_rate=48000.0)

        with pytest.raises(CrossDomainTypeError):
            validate_type_with_units(source, target, "audio", "visual")

    def test_stream_types_with_units(self):
        """Stream types with physical units are validated."""
        source = sig(f32(), sample_rate=44100.0, unit="Pa")  # Pressure
        target = sig(f32(), sample_rate=44100.0, unit="Pa")

        result = validate_type_with_units(source, target, "audio", "acoustics")
        assert result is True

    def test_stream_types_incompatible_units(self):
        """Stream types with incompatible units raise error."""
        source = sig(f32(), sample_rate=44100.0, unit="Pa")  # Pressure
        target = sig(f32(), sample_rate=44100.0, unit="m/s")  # Velocity

        with pytest.raises(CrossDomainTypeError) as exc_info:
            validate_type_with_units(source, target, "audio", "acoustics")

        assert "Incompatible units" in str(exc_info.value)

    def test_incompatible_base_types(self):
        """Incompatible base types raise error."""
        source = f32("m")
        target = vec2("m")

        with pytest.raises(CrossDomainTypeError) as exc_info:
            validate_type_with_units(source, target, "field", "agents")

        assert "Incompatible types" in str(exc_info.value)


class TestUnitConversionDetection:
    """Test unit conversion factor detection."""

    def test_same_units_no_conversion(self):
        """Same units don't need conversion."""
        factor = check_unit_conversion_needed("m", "m")
        assert factor is None

    def test_compatible_units_conversion(self):
        """Compatible units return conversion factor."""
        factor = check_unit_conversion_needed("m", "cm")
        assert factor == 100.0

    def test_kilometers_to_meters(self):
        """Kilometer to meter conversion."""
        factor = check_unit_conversion_needed("km", "m")
        assert factor == pytest.approx(1000.0)

    def test_incompatible_units_no_conversion(self):
        """Incompatible units return None."""
        factor = check_unit_conversion_needed("m", "kg")
        assert factor is None

    def test_none_units_no_conversion(self):
        """None units don't need conversion."""
        factor = check_unit_conversion_needed(None, "m")
        assert factor is None

        factor = check_unit_conversion_needed("m", None)
        assert factor is None


class TestComplexScenarios:
    """Test complex cross-domain scenarios."""

    def test_field_to_audio_velocity(self):
        """Field velocity to audio signal conversion."""
        # Field with velocity in m/s
        source = field2d(f32("m/s"), "m/s")
        # Audio signal (frequency modulation) in Hz
        target = sig(f32("Hz"), sample_rate=44100.0, unit="Hz")

        # These should be incompatible (velocity vs frequency)
        with pytest.raises(CrossDomainTypeError):
            validate_type_with_units(source, target, "field", "audio")

    def test_audio_pressure_to_audio_pressure(self):
        """Audio pressure signals at same sample rate."""
        source = sig(f32(), sample_rate=44100.0, unit="Pa")
        target = sig(f32(), sample_rate=44100.0, unit="Pa")

        result = validate_type_with_units(source, target, "audio", "acoustics")
        assert result is True

    def test_control_to_visual_same_units(self):
        """Control signal to visual with same units."""
        source = ctl(f32(), unit="1")  # Normalized
        target = ctl(f32(), unit="1")

        result = validate_type_with_units(source, target, "control", "visual")
        assert result is True

    def test_multi_domain_physics_audio_flow(self):
        """Complex flow: physics → audio with unit tracking."""
        # Physics simulation outputs force in Newtons
        physics_force = f32("N")
        # Audio expects pressure in Pascals
        audio_pressure = sig(f32(), sample_rate=44100.0, unit="Pa")

        # These are incompatible (force vs pressure)
        with pytest.raises(CrossDomainTypeError):
            validate_type_with_units(
                physics_force,
                audio_pressure,
                "rigidbody",
                "audio"
            )


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_unit_strings(self):
        """Empty unit strings are dimensionless."""
        result = validate_unit_compatibility("", "", "field", "agents")
        assert result is True

    def test_whitespace_units(self):
        """Whitespace-only units are dimensionless."""
        result = validate_unit_compatibility("   ", "   ", "field", "agents")
        assert result is True

    def test_complex_unit_expressions(self):
        """Complex derived units are validated."""
        # Force: kg*m/s^2 == N
        result = validate_unit_compatibility("kg*m/s^2", "N", "field", "agents")
        assert result is True

    def test_error_message_quality(self):
        """Error messages are informative."""
        with pytest.raises(CrossDomainTypeError) as exc_info:
            validate_unit_compatibility("m/s", "kg", "field", "agents")

        error_msg = str(exc_info.value)
        assert "field" in error_msg
        assert "agents" in error_msg
        assert "m/s" in error_msg or "Incompatible" in error_msg
