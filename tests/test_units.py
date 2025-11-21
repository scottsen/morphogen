"""Comprehensive tests for the physical units and dimensional analysis system."""

import pytest
from fractions import Fraction

from morphogen.types.units import (
    Unit,
    Dimensions,
    parse_unit,
    check_unit_compatibility,
    get_unit_dimensions,
)


class TestDimensions:
    """Test the Dimensions class and dimensional algebra."""

    def test_dimensionless(self):
        """Test dimensionless quantities."""
        dim = Dimensions()
        assert dim.is_dimensionless()
        assert str(dim) == "1"

    def test_base_dimensions(self):
        """Test basic dimensional formulas."""
        length = Dimensions(length=Fraction(1))
        assert str(length) == "L"
        assert not length.is_dimensionless()

        mass = Dimensions(mass=Fraction(1))
        assert str(mass) == "M"

        time = Dimensions(time=Fraction(1))
        assert str(time) == "T"

    def test_dimension_multiplication(self):
        """Test multiplying dimensions."""
        length = Dimensions(length=Fraction(1))
        time = Dimensions(time=Fraction(1))

        area = length * length
        assert area.length == Fraction(2)
        assert str(area) == "L^2"

        velocity = length * time
        assert velocity.length == Fraction(1)
        assert velocity.time == Fraction(1)
        assert str(velocity) == "L·T"

    def test_dimension_division(self):
        """Test dividing dimensions."""
        length = Dimensions(length=Fraction(1))
        time = Dimensions(time=Fraction(1))

        velocity = length / time
        assert velocity.length == Fraction(1)
        assert velocity.time == Fraction(-1)
        assert str(velocity) == "L·T^-1"

    def test_dimension_power(self):
        """Test raising dimensions to a power."""
        length = Dimensions(length=Fraction(1))

        area = length ** 2
        assert area.length == Fraction(2)

        volume = length ** 3
        assert volume.length == Fraction(3)

        # Fractional exponents
        sqrt_length = length ** Fraction(1, 2)
        assert sqrt_length.length == Fraction(1, 2)

    def test_complex_dimensions(self):
        """Test complex dimensional formulas."""
        # Force: M·L·T^-2
        force = Dimensions(mass=Fraction(1), length=Fraction(1), time=Fraction(-2))
        assert str(force) == "M·L·T^-2"

        # Energy: M·L^2·T^-2
        energy = Dimensions(mass=Fraction(1), length=Fraction(2), time=Fraction(-2))
        assert str(energy) == "M·L^2·T^-2"

        # Power: M·L^2·T^-3
        power = Dimensions(mass=Fraction(1), length=Fraction(2), time=Fraction(-3))
        assert str(power) == "M·L^2·T^-3"


class TestUnit:
    """Test the Unit class and unit algebra."""

    def test_base_units(self):
        """Test SI base units."""
        meter = Unit.meter()
        assert meter.name == "meter"
        assert meter.symbol == "m"
        assert meter.dimensions.length == Fraction(1)
        assert meter.scale == 1.0

        kilogram = Unit.kilogram()
        assert kilogram.symbol == "kg"
        assert kilogram.dimensions.mass == Fraction(1)

        second = Unit.second()
        assert second.symbol == "s"
        assert second.dimensions.time == Fraction(1)

    def test_derived_units(self):
        """Test derived SI units."""
        newton = Unit.newton()
        assert newton.symbol == "N"
        assert newton.dimensions.mass == Fraction(1)
        assert newton.dimensions.length == Fraction(1)
        assert newton.dimensions.time == Fraction(-2)

        joule = Unit.joule()
        assert joule.symbol == "J"
        assert joule.dimensions.mass == Fraction(1)
        assert joule.dimensions.length == Fraction(2)
        assert joule.dimensions.time == Fraction(-2)

    def test_unit_multiplication(self):
        """Test multiplying units."""
        meter = Unit.meter()
        second = Unit.second()

        # m * s
        meter_second = meter * second
        assert meter_second.dimensions.length == Fraction(1)
        assert meter_second.dimensions.time == Fraction(1)

        # m * m = m^2 (area)
        area = meter * meter
        assert area.dimensions.length == Fraction(2)

    def test_unit_division(self):
        """Test dividing units."""
        meter = Unit.meter()
        second = Unit.second()

        # m / s (velocity)
        velocity = meter / second
        assert velocity.dimensions.length == Fraction(1)
        assert velocity.dimensions.time == Fraction(-1)

        # Verify against built-in derived units
        kilogram = Unit.kilogram()
        # kg·m/s^2
        force_computed = (kilogram * meter) / (second * second)
        force_builtin = Unit.newton()
        assert force_computed.is_compatible_with(force_builtin)

    def test_unit_power(self):
        """Test raising units to a power."""
        meter = Unit.meter()

        square_meter = meter ** 2
        assert square_meter.dimensions.length == Fraction(2)

        cubic_meter = meter ** 3
        assert cubic_meter.dimensions.length == Fraction(3)

    def test_unit_compatibility(self):
        """Test checking if units are compatible (convertible)."""
        meter = Unit.meter()
        centimeter = Unit.centimeter()
        kilometer = Unit.kilometer()
        second = Unit.second()

        # Same dimension units are compatible
        assert meter.is_compatible_with(centimeter)
        assert meter.is_compatible_with(kilometer)
        assert centimeter.is_compatible_with(kilometer)

        # Different dimension units are not compatible
        assert not meter.is_compatible_with(second)

        # Derived units with same dimensions are compatible
        # kg·m/s^2 (Newton)
        kilogram = Unit.kilogram()
        force1 = (kilogram * meter) / (second ** 2)
        force2 = Unit.newton()
        assert force1.is_compatible_with(force2)

    def test_unit_conversion(self):
        """Test converting values between compatible units."""
        meter = Unit.meter()
        centimeter = Unit.centimeter()
        kilometer = Unit.kilometer()

        # 1 meter = 100 centimeters
        assert centimeter.convert_to(meter, 100.0) == pytest.approx(1.0)
        assert meter.convert_to(centimeter, 1.0) == pytest.approx(100.0)

        # 1 kilometer = 1000 meters
        assert kilometer.convert_to(meter, 1.0) == pytest.approx(1000.0)
        assert meter.convert_to(kilometer, 1000.0) == pytest.approx(1.0)

        # 1 kilometer = 100000 centimeters
        assert kilometer.convert_to(centimeter, 1.0) == pytest.approx(100000.0)

    def test_unit_conversion_incompatible(self):
        """Test that converting incompatible units raises an error."""
        meter = Unit.meter()
        second = Unit.second()

        with pytest.raises(ValueError, match="incompatible dimensions"):
            meter.convert_to(second, 10.0)

    def test_dimensionless_unit(self):
        """Test dimensionless units."""
        dimensionless = Unit.dimensionless()
        assert dimensionless.is_dimensionless()
        assert dimensionless.dimensions.is_dimensionless()


class TestParseUnit:
    """Test parsing unit expression strings."""

    def test_parse_simple_units(self):
        """Test parsing simple unit symbols."""
        meter = parse_unit("m")
        assert meter.symbol == "m"
        assert meter.dimensions.length == Fraction(1)

        kilogram = parse_unit("kg")
        assert kilogram.symbol == "kg"
        assert kilogram.dimensions.mass == Fraction(1)

        second = parse_unit("s")
        assert second.symbol == "s"
        assert second.dimensions.time == Fraction(1)

    def test_parse_products(self):
        """Test parsing unit products."""
        # kg*m
        result = parse_unit("kg*m")
        assert result.dimensions.mass == Fraction(1)
        assert result.dimensions.length == Fraction(1)

        # m*s
        result = parse_unit("m*s")
        assert result.dimensions.length == Fraction(1)
        assert result.dimensions.time == Fraction(1)

    def test_parse_quotients(self):
        """Test parsing unit quotients."""
        # m/s (velocity)
        velocity = parse_unit("m/s")
        assert velocity.dimensions.length == Fraction(1)
        assert velocity.dimensions.time == Fraction(-1)

        # kg/m^3 (density)
        density = parse_unit("kg/m^3")
        assert density.dimensions.mass == Fraction(1)
        assert density.dimensions.length == Fraction(-3)

    def test_parse_powers(self):
        """Test parsing unit powers."""
        # m^2 (area)
        area = parse_unit("m^2")
        assert area.dimensions.length == Fraction(2)

        # m^3 (volume)
        volume = parse_unit("m^3")
        assert volume.dimensions.length == Fraction(3)

        # s^-1 (frequency)
        frequency = parse_unit("s^-1")
        assert frequency.dimensions.time == Fraction(-1)

    def test_parse_complex_expressions(self):
        """Test parsing complex unit expressions."""
        # kg*m/s^2 (force)
        force = parse_unit("kg*m/s^2")
        expected_force = Unit.newton()
        assert force.is_compatible_with(expected_force)

        # kg*m^2/s^2 (energy)
        energy = parse_unit("kg*m^2/s^2")
        expected_energy = Unit.joule()
        assert energy.is_compatible_with(expected_energy)

        # kg*m^2/s^3 (power)
        power = parse_unit("kg*m^2/s^3")
        expected_power = Unit.watt()
        assert power.is_compatible_with(expected_power)

    def test_parse_dimensionless(self):
        """Test parsing dimensionless units."""
        dim1 = parse_unit("")
        assert dim1.is_dimensionless()

        dim2 = parse_unit("1")
        assert dim2.is_dimensionless()

        dim3 = parse_unit("  ")
        assert dim3.is_dimensionless()

    def test_parse_invalid_units(self):
        """Test parsing invalid unit expressions."""
        with pytest.raises(ValueError, match="Unknown unit"):
            parse_unit("xyz")

        with pytest.raises(ValueError, match="Invalid exponent"):
            parse_unit("m^abc")


class TestUnitCompatibility:
    """Test unit compatibility checking functions."""

    def test_check_compatible_units(self):
        """Test checking compatible unit strings."""
        # Same units are compatible
        assert check_unit_compatibility("m", "m")
        assert check_unit_compatibility("kg", "kg")

        # Different length units are compatible
        assert check_unit_compatibility("m", "cm")
        assert check_unit_compatibility("m", "km")

        # Dimensionally equivalent units are compatible
        assert check_unit_compatibility("kg*m/s^2", "N")

    def test_check_incompatible_units(self):
        """Test checking incompatible unit strings."""
        # Different dimensions are incompatible
        assert not check_unit_compatibility("m", "s")
        assert not check_unit_compatibility("kg", "m/s")

    def test_check_none_compatibility(self):
        """Test that None is compatible with any unit."""
        assert check_unit_compatibility(None, "m")
        assert check_unit_compatibility("m", None)
        assert check_unit_compatibility(None, None)

    def test_get_dimensions(self):
        """Test getting dimensions from unit strings."""
        # Length
        dim = get_unit_dimensions("m")
        assert dim is not None
        assert dim.length == Fraction(1)

        # Velocity
        dim = get_unit_dimensions("m/s")
        assert dim is not None
        assert dim.length == Fraction(1)
        assert dim.time == Fraction(-1)

        # Force
        dim = get_unit_dimensions("kg*m/s^2")
        assert dim is not None
        assert dim.mass == Fraction(1)
        assert dim.length == Fraction(1)
        assert dim.time == Fraction(-2)

        # None returns None
        assert get_unit_dimensions(None) is None

        # Invalid unit returns None
        assert get_unit_dimensions("invalid") is None


class TestCrossDomainUnits:
    """Test units in cross-domain scenarios."""

    def test_field_agent_units(self):
        """Test unit compatibility for field-agent interactions."""
        # Temperature field in Kelvin
        field_unit = parse_unit("K")

        # Agent property in Kelvin
        agent_unit = parse_unit("K")

        assert field_unit.is_compatible_with(agent_unit)

    def test_physics_audio_units(self):
        """Test unit compatibility for physics-audio sonification."""
        # Force in Newtons
        force_unit = parse_unit("N")

        # Manually computed force
        computed_force = parse_unit("kg*m/s^2")

        assert force_unit.is_compatible_with(computed_force)

    def test_spatial_temporal_units(self):
        """Test spatial and temporal unit interactions."""
        # Position: meters
        position = parse_unit("m")

        # Velocity: meters per second
        velocity = parse_unit("m/s")

        # Time: seconds
        time = parse_unit("s")

        # velocity * time should give position-compatible units
        result_unit = velocity * time
        assert result_unit.is_compatible_with(position)

    def test_energy_dissipation(self):
        """Test energy and power unit relationships."""
        # Energy: Joules
        energy = parse_unit("J")

        # Power: Watts
        power = parse_unit("W")

        # Time: seconds
        time = parse_unit("s")

        # power * time should give energy
        result_unit = power * time
        assert result_unit.is_compatible_with(energy)


class TestUnitScaling:
    """Test unit scaling and prefixes."""

    def test_length_scaling(self):
        """Test length unit conversions."""
        # 1 km = 1000 m
        km = Unit.kilometer()
        m = Unit.meter()
        assert km.convert_to(m, 1.0) == pytest.approx(1000.0)

        # 1 m = 100 cm
        cm = Unit.centimeter()
        assert m.convert_to(cm, 1.0) == pytest.approx(100.0)

        # 1 km = 100000 cm
        assert km.convert_to(cm, 1.0) == pytest.approx(100000.0)

    def test_mass_scaling(self):
        """Test mass unit conversions."""
        # 1 kg = 1000 g
        kg = Unit.kilogram()
        g = Unit.gram()
        assert kg.convert_to(g, 1.0) == pytest.approx(1000.0)
        assert g.convert_to(kg, 1000.0) == pytest.approx(1.0)

    def test_time_scaling(self):
        """Test time unit conversions."""
        # 1 s = 1000 ms
        s = Unit.second()
        ms = Unit.millisecond()
        assert s.convert_to(ms, 1.0) == pytest.approx(1000.0)
        assert ms.convert_to(s, 1000.0) == pytest.approx(1.0)


class TestDimensionalAnalysisEdgeCases:
    """Test edge cases in dimensional analysis."""

    def test_inverse_units(self):
        """Test inverse units (like frequency)."""
        # 1/s = Hz
        second = Unit.second()
        one = Unit.dimensionless()

        frequency = one / second
        assert frequency.dimensions.time == Fraction(-1)

        hz = Unit.hertz()
        assert frequency.is_compatible_with(hz)

    def test_square_root_dimensions(self):
        """Test fractional dimensional exponents."""
        meter = Unit.meter()

        # sqrt(meter) = m^(1/2)
        sqrt_meter = meter ** 0.5
        assert sqrt_meter.dimensions.length == Fraction(1, 2)

        # (m^1/2)^2 = m
        meter_again = sqrt_meter ** 2
        assert meter_again.is_compatible_with(meter)

    def test_unit_cancellation(self):
        """Test that units cancel properly."""
        meter = Unit.meter()
        second = Unit.second()

        # (m/s) * s = m
        velocity = meter / second
        result = velocity * second
        assert result.is_compatible_with(meter)

        # (m*s) / s = m
        result2 = (meter * second) / second
        assert result2.is_compatible_with(meter)
