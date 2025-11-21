"""Unit tests for audio envelope generators."""

import pytest
import numpy as np
from morphogen.stdlib.audio import audio, AudioBuffer


class TestADSREnvelope:
    """Tests for ADSR envelope generator."""

    def test_adsr_basic(self):
        """Test basic ADSR envelope generation."""
        env = audio.adsr(attack=0.1, decay=0.1, sustain=0.7, release=0.2, duration=1.0)
        assert env.num_samples == 44100
        assert env.sample_rate == 44100

    def test_adsr_starts_at_zero(self):
        """Test ADSR starts at zero."""
        env = audio.adsr(attack=0.01, decay=0.05, sustain=0.6, release=0.1, duration=0.5)
        assert abs(env.data[0]) < 0.01

    def test_adsr_attack_peak(self):
        """Test ADSR reaches peak of 1.0 after attack."""
        env = audio.adsr(attack=0.1, decay=0.1, sustain=0.7, release=0.2, duration=1.0,
                        sample_rate=1000)
        # After attack (100 samples), should be at or near 1.0
        peak_idx = 100
        assert 0.95 < env.data[peak_idx] <= 1.0

    def test_adsr_sustain_level(self):
        """Test ADSR sustain level is correct."""
        env = audio.adsr(attack=0.1, decay=0.1, sustain=0.5, release=0.1, duration=1.0,
                        sample_rate=1000)
        # During sustain phase (after attack + decay), should be at sustain level
        sustain_idx = 250  # After attack (100) and decay (100)
        assert 0.45 < env.data[sustain_idx] < 0.55

    def test_adsr_ends_at_zero(self):
        """Test ADSR ends at zero after release."""
        env = audio.adsr(attack=0.05, decay=0.05, sustain=0.6, release=0.1, duration=0.5)
        # Last sample should be near zero
        assert abs(env.data[-1]) < 0.1

    def test_adsr_shapes(self):
        """Test ADSR has expected shape characteristics."""
        env = audio.adsr(attack=0.1, decay=0.1, sustain=0.7, release=0.2, duration=1.0,
                        sample_rate=1000)

        # Attack should be rising
        attack_region = env.data[10:90]
        assert np.all(np.diff(attack_region) >= -0.01)  # Monotonically increasing

        # Decay should be falling
        decay_region = env.data[110:190]
        assert np.all(np.diff(decay_region) <= 0.01)  # Monotonically decreasing

    def test_adsr_zero_sustain(self):
        """Test ADSR with zero sustain level."""
        env = audio.adsr(attack=0.1, decay=0.1, sustain=0.0, release=0.1, duration=0.5)
        # Should work and reach zero during sustain
        assert env.num_samples > 0

    def test_adsr_full_sustain(self):
        """Test ADSR with full sustain (1.0)."""
        env = audio.adsr(attack=0.1, decay=0.0, sustain=1.0, release=0.1, duration=0.5)
        # No decay, so should stay at 1.0
        assert env.num_samples > 0


class TestAREnvelope:
    """Tests for AR (Attack-Release) envelope generator."""

    def test_ar_basic(self):
        """Test basic AR envelope generation."""
        env = audio.ar(attack=0.005, release=0.3, duration=1.0)
        assert env.num_samples == 44100

    def test_ar_starts_at_zero(self):
        """Test AR starts at zero."""
        env = audio.ar(attack=0.01, release=0.1, duration=0.5)
        assert abs(env.data[0]) < 0.01

    def test_ar_reaches_peak(self):
        """Test AR reaches peak of 1.0."""
        env = audio.ar(attack=0.1, release=0.3, duration=1.0, sample_rate=1000)
        # After attack, should be at 1.0
        peak_idx = 100
        assert 0.95 < env.data[peak_idx] <= 1.0

    def test_ar_ends_at_zero(self):
        """Test AR ends at zero."""
        env = audio.ar(attack=0.05, release=0.1, duration=0.3)
        assert abs(env.data[-1]) < 0.1

    def test_ar_vs_adsr(self):
        """Test AR is simpler than ADSR (no decay/sustain)."""
        ar_env = audio.ar(attack=0.1, release=0.3, duration=0.5)
        adsr_env = audio.adsr(attack=0.1, decay=0.0, sustain=1.0, release=0.3, duration=0.5)

        # Should be very similar
        assert ar_env.num_samples == adsr_env.num_samples


class TestExpEnvelope:
    """Tests for exponential decay envelope."""

    def test_envexp_basic(self):
        """Test basic exponential envelope generation."""
        env = audio.envexp(time_constant=0.05, duration=1.0)
        assert env.num_samples == 44100

    def test_envexp_starts_at_one(self):
        """Test exponential envelope starts at 1.0."""
        env = audio.envexp(time_constant=0.1, duration=0.5)
        assert abs(env.data[0] - 1.0) < 0.01

    def test_envexp_decays(self):
        """Test exponential envelope decays over time."""
        env = audio.envexp(time_constant=0.1, duration=1.0)
        # Should be monotonically decreasing
        assert np.all(np.diff(env.data) <= 0.0)

    def test_envexp_63_percent_decay(self):
        """Test exponential decay reaches ~37% at time constant."""
        env = audio.envexp(time_constant=0.1, duration=0.5, sample_rate=1000)
        # At time constant (100 samples), should be at e^-1 ≈ 0.368
        tc_idx = 100
        assert 0.3 < env.data[tc_idx] < 0.4

    def test_envexp_fast_decay(self):
        """Test fast exponential decay."""
        env = audio.envexp(time_constant=0.01, duration=0.5)
        # Should decay quickly (e^(-t/τ) at t=1000/44100≈0.0227s with τ=0.01 gives ~0.103)
        assert env.data[1000] < 0.11

    def test_envexp_slow_decay(self):
        """Test slow exponential decay."""
        env = audio.envexp(time_constant=1.0, duration=0.5)
        # Should decay slowly, still > 0.5 after 0.5s
        assert env.data[-1] > 0.5


class TestEnvelopeApplication:
    """Tests for applying envelopes to signals."""

    def test_adsr_on_sine(self):
        """Test applying ADSR to sine wave."""
        sine = audio.sine(freq=440.0, duration=1.0)
        env = audio.adsr(attack=0.01, decay=0.1, sustain=0.7, release=0.3, duration=1.0)

        # Apply envelope by multiplication
        shaped = AudioBuffer(data=sine.data * env.data, sample_rate=44100)

        # Should start quiet, get loud, then fade
        assert abs(shaped.data[0]) < 0.1
        assert np.max(np.abs(shaped.data[1000:10000])) > 0.5

    def test_ar_on_noise(self):
        """Test applying AR to noise."""
        noise = audio.noise(noise_type="white", seed=42, duration=0.5)
        env = audio.ar(attack=0.01, release=0.2, duration=0.5)

        shaped = AudioBuffer(data=noise.data * env.data, sample_rate=44100)

        # Envelope should shape the noise
        assert shaped.num_samples == noise.num_samples

    def test_envexp_on_impulse(self):
        """Test exponential envelope on impulse (percussive sound)."""
        impulse = audio.impulse(rate=1.0, duration=0.001)
        # Extend with zeros for envelope
        extended = np.pad(impulse.data, (0, 44100))

        env = audio.envexp(time_constant=0.1, duration=1.0)

        shaped = AudioBuffer(data=extended[:44100] + env.data * 0.5, sample_rate=44100)

        assert shaped.num_samples == 44100


class TestEnvelopeDeterminism:
    """Tests for deterministic envelope generation."""

    def test_adsr_deterministic(self):
        """Test ADSR is deterministic."""
        env1 = audio.adsr(attack=0.01, decay=0.1, sustain=0.7, release=0.2, duration=1.0)
        env2 = audio.adsr(attack=0.01, decay=0.1, sustain=0.7, release=0.2, duration=1.0)

        assert np.array_equal(env1.data, env2.data)

    def test_ar_deterministic(self):
        """Test AR is deterministic."""
        env1 = audio.ar(attack=0.01, release=0.3, duration=1.0)
        env2 = audio.ar(attack=0.01, release=0.3, duration=1.0)

        assert np.array_equal(env1.data, env2.data)

    def test_envexp_deterministic(self):
        """Test exponential envelope is deterministic."""
        env1 = audio.envexp(time_constant=0.1, duration=1.0)
        env2 = audio.envexp(time_constant=0.1, duration=1.0)

        assert np.array_equal(env1.data, env2.data)


class TestEnvelopeEdgeCases:
    """Tests for envelope edge cases."""

    def test_zero_attack(self):
        """Test envelope with zero attack time."""
        env = audio.adsr(attack=0.0, decay=0.1, sustain=0.7, release=0.2, duration=0.5)
        # Should start at peak immediately
        assert env.data[0] > 0.9

    def test_zero_release(self):
        """Test envelope with zero release time."""
        env = audio.adsr(attack=0.05, decay=0.05, sustain=0.7, release=0.0, duration=0.3)
        # Should drop immediately at end
        assert env.num_samples > 0

    def test_very_short_envelope(self):
        """Test very short envelope duration."""
        env = audio.adsr(attack=0.001, decay=0.001, sustain=0.7, release=0.001,
                        duration=0.01)
        assert env.num_samples > 0
        assert not np.any(np.isnan(env.data))

    def test_very_long_envelope(self):
        """Test very long envelope duration."""
        env = audio.adsr(attack=1.0, decay=1.0, sustain=0.7, release=1.0,
                        duration=10.0)
        assert env.num_samples == 441000

    def test_envelope_longer_than_duration(self):
        """Test when envelope stages exceed duration."""
        # Attack + decay + release = 0.6s but duration = 0.3s
        env = audio.adsr(attack=0.2, decay=0.2, sustain=0.7, release=0.2,
                        duration=0.3)
        # Should still generate something reasonable
        assert env.num_samples > 0
        assert not np.any(np.isnan(env.data))


class TestEnvelopeNormalization:
    """Tests for envelope normalization and range."""

    def test_adsr_range(self):
        """Test ADSR stays in [0, 1] range."""
        env = audio.adsr(attack=0.01, decay=0.1, sustain=0.7, release=0.2, duration=1.0)
        assert np.min(env.data) >= -0.01
        assert np.max(env.data) <= 1.01

    def test_ar_range(self):
        """Test AR stays in [0, 1] range."""
        env = audio.ar(attack=0.01, release=0.3, duration=1.0)
        assert np.min(env.data) >= -0.01
        assert np.max(env.data) <= 1.01

    def test_envexp_range(self):
        """Test exponential envelope starts at 1.0 and decays."""
        env = audio.envexp(time_constant=0.1, duration=1.0)
        assert abs(env.data[0] - 1.0) < 0.01
        assert np.min(env.data) >= 0.0
        assert np.max(env.data) <= 1.01
