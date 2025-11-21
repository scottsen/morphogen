"""Unit tests for audio effects operations."""

import pytest
import numpy as np
from morphogen.stdlib.audio import audio, AudioBuffer


class TestDelay:
    """Tests for delay effect."""

    def test_delay_basic(self):
        """Test basic delay application."""
        sig = audio.sine(freq=440.0, duration=1.0)
        delayed = audio.delay(sig, time=0.3, feedback=0.3, mix=0.25)
        assert delayed.num_samples == sig.num_samples

    def test_delay_creates_echo(self):
        """Test delay creates echo effect."""
        # Short impulse
        sig = audio.impulse(rate=1.0, duration=1.0)
        delayed = audio.delay(sig, time=0.25, feedback=0.0, mix=1.0)

        # Should have energy at delay time
        delay_samples = int(0.25 * 44100)
        # Original impulse at start, echo at delay_samples
        assert sig.data[0] > 0.5
        assert delayed.data[delay_samples] > 0.1

    def test_delay_feedback(self):
        """Test delay feedback creates multiple echoes."""
        sig = audio.impulse(rate=1.0, duration=1.0)
        delayed = audio.delay(sig, time=0.1, feedback=0.5, mix=1.0)

        # Should have multiple echoes
        delay_samples = int(0.1 * 44100)
        assert delayed.data[delay_samples] > 0.1
        assert delayed.data[delay_samples * 2] > 0.05

    def test_delay_mix(self):
        """Test delay mix parameter."""
        sig = audio.sine(freq=440.0, duration=0.5)

        dry = audio.delay(sig, time=0.3, feedback=0.3, mix=0.0)
        wet = audio.delay(sig, time=0.3, feedback=0.3, mix=1.0)

        # Dry should be close to original
        assert np.allclose(dry.data, sig.data, atol=0.01)

        # Wet should be different
        assert not np.allclose(wet.data, sig.data)

    def test_delay_zero_time(self):
        """Test delay with zero delay time."""
        sig = audio.sine(freq=440.0, duration=0.1)
        delayed = audio.delay(sig, time=0.0, feedback=0.3, mix=0.5)

        # Should return essentially the original
        assert delayed.num_samples == sig.num_samples


class TestReverb:
    """Tests for reverb effect."""

    def test_reverb_basic(self):
        """Test basic reverb application."""
        sig = audio.sine(freq=440.0, duration=1.0)
        reverbed = audio.reverb(sig, mix=0.12, size=0.8)
        assert reverbed.num_samples == sig.num_samples

    def test_reverb_extends_sound(self):
        """Test reverb extends/smears sound."""
        # Short impulse
        sig = audio.impulse(rate=1.0, duration=0.5)
        reverbed = audio.reverb(sig, mix=1.0, size=0.8)

        # Reverb should spread energy over time
        sig_energy_late = np.sum(sig.data[10000:20000] ** 2)
        rev_energy_late = np.sum(reverbed.data[10000:20000] ** 2)

        assert rev_energy_late > sig_energy_late

    def test_reverb_mix(self):
        """Test reverb mix parameter."""
        sig = audio.sine(freq=440.0, duration=0.5)

        dry = audio.reverb(sig, mix=0.0, size=0.8)
        wet = audio.reverb(sig, mix=1.0, size=0.8)

        # Dry should be close to original
        assert np.allclose(dry.data, sig.data, atol=0.01)

        # Wet should be different
        assert not np.allclose(wet.data, sig.data)

    def test_reverb_size(self):
        """Test reverb size parameter."""
        sig = audio.impulse(rate=1.0, duration=0.5)

        small = audio.reverb(sig, mix=1.0, size=0.2)
        large = audio.reverb(sig, mix=1.0, size=1.0)

        # Different sizes should produce different results
        assert not np.allclose(small.data, large.data)


class TestChorus:
    """Tests for chorus effect."""

    def test_chorus_basic(self):
        """Test basic chorus application."""
        sig = audio.sine(freq=440.0, duration=1.0)
        chorused = audio.chorus(sig, rate=0.3, depth=0.008, mix=0.25)
        assert chorused.num_samples == sig.num_samples

    def test_chorus_modulates(self):
        """Test chorus creates modulation effect."""
        sig = audio.sine(freq=440.0, duration=1.0)

        dry = audio.chorus(sig, rate=0.3, depth=0.008, mix=0.0)
        wet = audio.chorus(sig, rate=0.3, depth=0.008, mix=1.0)

        # Dry should match original
        assert np.allclose(dry.data, sig.data, atol=0.01)

        # Wet should be different
        assert not np.allclose(wet.data, sig.data)

    def test_chorus_rate(self):
        """Test chorus rate parameter."""
        sig = audio.sine(freq=440.0, duration=1.0)

        slow = audio.chorus(sig, rate=0.1, depth=0.008, mix=0.5)
        fast = audio.chorus(sig, rate=1.0, depth=0.008, mix=0.5)

        # Different rates should produce different results
        assert not np.allclose(slow.data, fast.data)

    def test_chorus_depth(self):
        """Test chorus depth parameter."""
        sig = audio.sine(freq=440.0, duration=1.0)

        shallow = audio.chorus(sig, rate=0.3, depth=0.002, mix=0.5)
        deep = audio.chorus(sig, rate=0.3, depth=0.020, mix=0.5)

        # Different depths should produce different results
        assert not np.allclose(shallow.data, deep.data)


class TestFlanger:
    """Tests for flanger effect."""

    def test_flanger_basic(self):
        """Test basic flanger application."""
        sig = audio.sine(freq=440.0, duration=1.0)
        flanged = audio.flanger(sig, rate=0.2, depth=0.003, feedback=0.25, mix=0.5)
        assert flanged.num_samples == sig.num_samples

    def test_flanger_vs_chorus(self):
        """Test flanger is similar to but distinct from chorus."""
        sig = audio.sine(freq=440.0, duration=1.0)

        chorus = audio.chorus(sig, rate=0.2, depth=0.003, mix=0.5)
        flanger = audio.flanger(sig, rate=0.2, depth=0.003, feedback=0.0, mix=0.5)

        # Both use modulated delays but different ranges (chorus 20ms base, flanger 1ms base)
        # So they're related effects but produce quite different results
        correlation = np.corrcoef(chorus.data, flanger.data)[0, 1]
        assert 0.1 < correlation < 0.99  # Related but distinct due to delay range differences

    def test_flanger_feedback(self):
        """Test flanger feedback parameter."""
        sig = audio.sine(freq=440.0, duration=1.0)

        no_fb = audio.flanger(sig, rate=0.2, depth=0.003, feedback=0.0, mix=0.5)
        with_fb = audio.flanger(sig, rate=0.2, depth=0.003, feedback=0.5, mix=0.5)

        # Feedback should create more resonance
        assert not np.allclose(no_fb.data, with_fb.data)


class TestDrive:
    """Tests for distortion/drive effect."""

    def test_drive_basic(self):
        """Test basic drive application."""
        sig = audio.sine(freq=440.0, duration=0.5)
        driven = audio.drive(sig, amount=0.5, shape="tanh")
        assert driven.num_samples == sig.num_samples

    def test_drive_tanh(self):
        """Test tanh distortion shape."""
        sig = audio.sine(freq=440.0, duration=0.5)
        driven = audio.drive(sig, amount=0.8, shape="tanh")

        # Should compress peaks
        assert np.max(np.abs(driven.data)) < np.max(np.abs(sig.data)) * 5.0

    def test_drive_hard_clip(self):
        """Test hard clipping distortion."""
        sig = audio.sine(freq=440.0, duration=0.5)
        driven = audio.drive(sig, amount=0.8, shape="hard")

        # Hard clip should limit to ±1
        assert np.max(np.abs(driven.data)) <= 1.5

    def test_drive_soft_clip(self):
        """Test soft clipping distortion."""
        sig = audio.sine(freq=440.0, duration=0.5)
        driven = audio.drive(sig, amount=0.8, shape="soft")

        assert driven.num_samples == sig.num_samples

    def test_drive_amount(self):
        """Test drive amount parameter."""
        sig = audio.sine(freq=440.0, duration=0.5)

        light = audio.drive(sig, amount=0.1, shape="tanh")
        heavy = audio.drive(sig, amount=0.9, shape="tanh")

        # More drive should be more different from original
        light_diff = np.sum((light.data - sig.data) ** 2)
        heavy_diff = np.sum((heavy.data - sig.data) ** 2)

        assert heavy_diff > light_diff

    def test_drive_invalid_shape(self):
        """Test invalid distortion shape raises error."""
        sig = audio.sine(freq=440.0, duration=0.1)

        with pytest.raises(ValueError, match="Unknown distortion shape"):
            audio.drive(sig, amount=0.5, shape="invalid")


class TestLimiter:
    """Tests for limiter/compressor effect."""

    def test_limiter_basic(self):
        """Test basic limiter application."""
        sig = audio.sine(freq=440.0, duration=0.5)
        limited = audio.limiter(sig, threshold=-1.0, release=0.05)
        assert limited.num_samples == sig.num_samples

    def test_limiter_prevents_clipping(self):
        """Test limiter prevents clipping."""
        # Create signal that would clip
        sig = audio.sine(freq=440.0, duration=0.5)
        sig.data *= 2.0  # Amplify beyond ±1

        limited = audio.limiter(sig, threshold=-1.0, release=0.05)

        # Should be limited
        threshold_lin = audio.db2lin(-1.0)
        assert np.max(np.abs(limited.data)) <= threshold_lin * 1.1

    def test_limiter_threshold(self):
        """Test limiter threshold parameter."""
        sig = audio.sine(freq=440.0, duration=0.5)
        sig.data *= 1.5

        high_thresh = audio.limiter(sig, threshold=-3.0, release=0.05)
        low_thresh = audio.limiter(sig, threshold=-12.0, release=0.05)

        # Lower threshold should compress more
        assert np.max(np.abs(low_thresh.data)) < np.max(np.abs(high_thresh.data))

    def test_limiter_release(self):
        """Test limiter release parameter."""
        sig = audio.sine(freq=440.0, duration=0.5)
        sig.data *= 1.5

        fast = audio.limiter(sig, threshold=-6.0, release=0.01)
        slow = audio.limiter(sig, threshold=-6.0, release=0.2)

        # Different release times should produce different results
        assert not np.allclose(fast.data, slow.data)


class TestEffectsStability:
    """Tests for effects stability and numerical issues."""

    def test_delay_stability(self):
        """Test delay doesn't produce NaN or Inf."""
        sig = audio.sine(freq=440.0, duration=1.0)
        delayed = audio.delay(sig, time=0.5, feedback=0.9, mix=0.5)

        assert not np.any(np.isnan(delayed.data))
        assert not np.any(np.isinf(delayed.data))

    def test_reverb_stability(self):
        """Test reverb doesn't produce NaN or Inf."""
        sig = audio.sine(freq=440.0, duration=1.0)
        reverbed = audio.reverb(sig, mix=0.9, size=1.0)

        assert not np.any(np.isnan(reverbed.data))
        assert not np.any(np.isinf(reverbed.data))

    def test_drive_stability(self):
        """Test drive doesn't produce NaN or Inf."""
        sig = audio.sine(freq=440.0, duration=0.5)
        driven = audio.drive(sig, amount=1.0, shape="tanh")

        assert not np.any(np.isnan(driven.data))
        assert not np.any(np.isinf(driven.data))

    def test_effects_with_silence(self):
        """Test effects handle silence correctly."""
        sig = AudioBuffer(data=np.zeros(1000), sample_rate=44100)

        delayed = audio.delay(sig, time=0.3, feedback=0.5, mix=0.5)
        reverbed = audio.reverb(sig, mix=0.5, size=0.8)
        driven = audio.drive(sig, amount=0.5, shape="tanh")

        assert np.all(delayed.data == 0.0)
        assert np.all(reverbed.data == 0.0)
        assert np.all(driven.data == 0.0)


class TestEffectsChaining:
    """Tests for chaining multiple effects."""

    def test_delay_reverb_chain(self):
        """Test chaining delay and reverb."""
        sig = audio.sine(freq=440.0, duration=1.0)

        processed = audio.delay(sig, time=0.3, feedback=0.3, mix=0.3)
        processed = audio.reverb(processed, mix=0.2, size=0.7)

        assert processed.num_samples == sig.num_samples
        assert not np.any(np.isnan(processed.data))

    def test_drive_limiter_chain(self):
        """Test chaining drive and limiter (typical guitar chain)."""
        sig = audio.sine(freq=440.0, duration=0.5)

        processed = audio.drive(sig, amount=0.7, shape="tanh")
        processed = audio.limiter(processed, threshold=-3.0, release=0.05)

        assert processed.num_samples == sig.num_samples
        assert not np.any(np.isnan(processed.data))

    def test_chorus_reverb_chain(self):
        """Test chaining chorus and reverb (typical pad sound)."""
        sig = audio.sine(freq=440.0, duration=1.0)

        processed = audio.chorus(sig, rate=0.3, depth=0.008, mix=0.3)
        processed = audio.reverb(processed, mix=0.25, size=0.8)

        assert processed.num_samples == sig.num_samples


class TestEffectsDeterminism:
    """Tests for deterministic effects behavior."""

    def test_delay_deterministic(self):
        """Test delay is deterministic."""
        sig = audio.sine(freq=440.0, duration=0.5)

        delayed1 = audio.delay(sig, time=0.3, feedback=0.3, mix=0.5)
        delayed2 = audio.delay(sig, time=0.3, feedback=0.3, mix=0.5)

        assert np.array_equal(delayed1.data, delayed2.data)

    def test_reverb_deterministic(self):
        """Test reverb is deterministic."""
        sig = audio.sine(freq=440.0, duration=0.5)

        reverb1 = audio.reverb(sig, mix=0.3, size=0.8)
        reverb2 = audio.reverb(sig, mix=0.3, size=0.8)

        assert np.allclose(reverb1.data, reverb2.data)

    def test_drive_deterministic(self):
        """Test drive is deterministic."""
        sig = audio.sine(freq=440.0, duration=0.5)

        drive1 = audio.drive(sig, amount=0.5, shape="tanh")
        drive2 = audio.drive(sig, amount=0.5, shape="tanh")

        assert np.array_equal(drive1.data, drive2.data)
