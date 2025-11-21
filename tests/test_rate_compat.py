"""Tests for rate compatibility checking system."""

import pytest
from morphogen.types.rate_compat import (
    Rate,
    RateInfo,
    RateCompatibilityError,
    check_rate_compatibility,
    check_sample_rate_compatibility,
    get_conversion_factor,
    requires_resampling,
    recommend_conversion_operator,
    validate_rate_compatibility,
    RATE_CONFIGS,
)


class TestRateInfo:
    """Test RateInfo dataclass."""

    def test_audio_rate_default_sample_rate(self):
        """Audio rate should get default sample rate if not specified."""
        info = RateInfo(Rate.AUDIO)
        assert info.sample_rate == 44100.0

    def test_audio_rate_explicit_sample_rate(self):
        """Audio rate can have explicit sample rate."""
        info = RateInfo(Rate.AUDIO, sample_rate=48000.0)
        assert info.sample_rate == 48000.0

    def test_control_rate_no_sample_rate(self):
        """Control rate uses frequency, not sample rate."""
        info = RateInfo(Rate.CONTROL, frequency=100.0)
        assert info.frequency == 100.0
        assert info.sample_rate is None

    def test_event_rate_no_frequency(self):
        """Event rate cannot have fixed frequency."""
        with pytest.raises(ValueError, match="Event rate cannot have a fixed frequency"):
            RateInfo(Rate.EVENT, frequency=100.0)

    def test_event_rate_valid(self):
        """Event rate without frequency is valid."""
        info = RateInfo(Rate.EVENT)
        assert info.rate == Rate.EVENT
        assert info.frequency is None


class TestRateCompatibility:
    """Test basic rate compatibility checking."""

    def test_same_rate_compatible(self):
        """Same rates are compatible."""
        assert check_rate_compatibility(Rate.AUDIO, Rate.AUDIO)
        assert check_rate_compatibility(Rate.CONTROL, Rate.CONTROL)
        assert check_rate_compatibility(Rate.EVENT, Rate.EVENT)

    def test_different_rates_incompatible(self):
        """Different rates are incompatible."""
        assert not check_rate_compatibility(Rate.AUDIO, Rate.CONTROL)
        assert not check_rate_compatibility(Rate.CONTROL, Rate.VISUAL)
        assert not check_rate_compatibility(Rate.AUDIO, Rate.EVENT)

    def test_sample_rate_compatibility_exact(self):
        """Exact sample rates are compatible."""
        assert check_sample_rate_compatibility(44100.0, 44100.0)
        assert check_sample_rate_compatibility(48000.0, 48000.0)

    def test_sample_rate_compatibility_close(self):
        """Sample rates within tolerance are compatible."""
        assert check_sample_rate_compatibility(44100.0, 44100.05)
        assert not check_sample_rate_compatibility(44100.0, 48000.0)

    def test_sample_rate_compatibility_none(self):
        """None sample rates are always compatible."""
        assert check_sample_rate_compatibility(None, 44100.0)
        assert check_sample_rate_compatibility(44100.0, None)
        assert check_sample_rate_compatibility(None, None)


class TestConversionFactor:
    """Test conversion factor calculation."""

    def test_audio_to_control_conversion(self):
        """Calculate audio to control rate conversion."""
        source = RateInfo(Rate.AUDIO, sample_rate=44100.0)
        target = RateInfo(Rate.CONTROL, frequency=100.0)
        factor = get_conversion_factor(source, target)
        assert factor == 441.0  # 44100 / 100

    def test_control_to_audio_conversion(self):
        """Calculate control to audio rate conversion."""
        source = RateInfo(Rate.CONTROL, frequency=100.0)
        target = RateInfo(Rate.AUDIO, sample_rate=44100.0)
        factor = get_conversion_factor(source, target)
        assert factor == pytest.approx(0.00226757, rel=1e-5)  # 100 / 44100

    def test_same_sample_rate_factor_one(self):
        """Same sample rates give factor of 1."""
        source = RateInfo(Rate.AUDIO, sample_rate=44100.0)
        target = RateInfo(Rate.AUDIO, sample_rate=44100.0)
        factor = get_conversion_factor(source, target)
        assert factor == 1.0

    def test_event_rate_no_factor(self):
        """Event rate returns None (no fixed rate)."""
        source = RateInfo(Rate.EVENT)
        target = RateInfo(Rate.AUDIO, sample_rate=44100.0)
        factor = get_conversion_factor(source, target)
        assert factor is None


class TestResamplingRequirement:
    """Test resampling requirement detection."""

    def test_same_rate_no_resampling(self):
        """Same rate types don't need resampling."""
        source = RateInfo(Rate.AUDIO, sample_rate=44100.0)
        target = RateInfo(Rate.AUDIO, sample_rate=44100.0)
        assert not requires_resampling(source, target)

    def test_different_continuous_rates_need_resampling(self):
        """Different continuous rates need resampling."""
        source = RateInfo(Rate.AUDIO, sample_rate=44100.0)
        target = RateInfo(Rate.CONTROL, frequency=100.0)
        assert requires_resampling(source, target)

    def test_different_audio_sample_rates_need_resampling(self):
        """Different audio sample rates need resampling."""
        source = RateInfo(Rate.AUDIO, sample_rate=44100.0)
        target = RateInfo(Rate.AUDIO, sample_rate=48000.0)
        assert requires_resampling(source, target)

    def test_event_rate_no_resampling(self):
        """Event streams don't resample."""
        source = RateInfo(Rate.EVENT)
        target = RateInfo(Rate.AUDIO, sample_rate=44100.0)
        assert not requires_resampling(source, target)


class TestConversionOperatorRecommendations:
    """Test conversion operator recommendations."""

    def test_audio_to_control(self):
        """Recommend audio to control conversion."""
        op = recommend_conversion_operator(Rate.AUDIO, Rate.CONTROL)
        assert op == "audio_to_control"

    def test_control_to_audio(self):
        """Recommend control to audio conversion."""
        op = recommend_conversion_operator(Rate.CONTROL, Rate.AUDIO)
        assert op == "control_to_audio"

    def test_audio_to_visual(self):
        """Recommend audio to visual conversion."""
        op = recommend_conversion_operator(Rate.AUDIO, Rate.VISUAL)
        assert op == "audio_to_visual"

    def test_visual_to_control(self):
        """Recommend visual to control conversion."""
        op = recommend_conversion_operator(Rate.VISUAL, Rate.CONTROL)
        assert op == "visual_to_control"

    def test_event_to_audio(self):
        """Recommend event to audio conversion."""
        op = recommend_conversion_operator(Rate.EVENT, Rate.AUDIO)
        assert op == "events_to_audio"

    def test_generic_fallback(self):
        """Fallback to generic conversion for uncommon pairs."""
        op = recommend_conversion_operator(Rate.SIMULATION, Rate.VISUAL)
        assert op == "sim_to_visual"


class TestRateValidation:
    """Test comprehensive rate validation."""

    def test_valid_same_rate(self):
        """Validate matching rates."""
        valid, msg = validate_rate_compatibility(Rate.AUDIO, Rate.AUDIO, 44100.0, 44100.0)
        assert valid
        assert msg is None

    def test_invalid_different_rate(self):
        """Invalidate different rates."""
        valid, msg = validate_rate_compatibility(Rate.AUDIO, Rate.CONTROL)
        assert not valid
        assert "audio → control requires explicit conversion" in msg
        assert "audio_to_control()" in msg

    def test_invalid_different_sample_rate(self):
        """Invalidate different audio sample rates."""
        valid, msg = validate_rate_compatibility(Rate.AUDIO, Rate.AUDIO, 44100.0, 48000.0)
        assert not valid
        assert "44100" in msg
        assert "48000" in msg
        assert "resample" in msg

    def test_valid_matching_sample_rate(self):
        """Validate matching audio sample rates."""
        valid, msg = validate_rate_compatibility(Rate.AUDIO, Rate.AUDIO, 44100.0, 44100.0)
        assert valid
        assert msg is None


class TestRateConfigs:
    """Test predefined rate configurations."""

    def test_cd_audio_config(self):
        """CD audio is 44.1kHz."""
        config = RATE_CONFIGS["cd_audio"]
        assert config.rate == Rate.AUDIO
        assert config.sample_rate == 44100.0

    def test_dvd_audio_config(self):
        """DVD audio is 48kHz."""
        config = RATE_CONFIGS["dvd_audio"]
        assert config.rate == Rate.AUDIO
        assert config.sample_rate == 48000.0

    def test_hd_audio_config(self):
        """HD audio is 96kHz."""
        config = RATE_CONFIGS["hd_audio"]
        assert config.rate == Rate.AUDIO
        assert config.sample_rate == 96000.0

    def test_control_medium_config(self):
        """Medium control rate is 100Hz."""
        config = RATE_CONFIGS["control_medium"]
        assert config.rate == Rate.CONTROL
        assert config.frequency == 100.0

    def test_visual_60fps_config(self):
        """60fps visual rate."""
        config = RATE_CONFIGS["visual_60fps"]
        assert config.rate == Rate.VISUAL
        assert config.frequency == 60.0

    def test_events_config(self):
        """Event rate configuration."""
        config = RATE_CONFIGS["events"]
        assert config.rate == Rate.EVENT
        assert config.frequency is None


class TestRateCompatibilityError:
    """Test RateCompatibilityError exception."""

    def test_error_message(self):
        """Error message includes rates and suggestion."""
        error = RateCompatibilityError(Rate.AUDIO, Rate.CONTROL)
        msg = str(error)
        assert "audio" in msg
        assert "control" in msg
        assert "conversion" in msg

    def test_custom_message(self):
        """Custom error message."""
        error = RateCompatibilityError(
            Rate.AUDIO,
            Rate.CONTROL,
            "Custom error message"
        )
        assert str(error) == "Custom error message"

    def test_attributes(self):
        """Error has source and target rate attributes."""
        error = RateCompatibilityError(Rate.AUDIO, Rate.CONTROL)
        assert error.source_rate == Rate.AUDIO
        assert error.target_rate == Rate.CONTROL
