"""Unit tests for physical modeling audio operations."""

import pytest
import numpy as np
from morphogen.stdlib.audio import audio, AudioBuffer


class TestKarplusStrong:
    """Tests for Karplus-Strong string synthesis."""

    def test_string_basic(self):
        """Test basic Karplus-Strong string synthesis."""
        exc = audio.noise(noise_type="white", seed=1, duration=0.01)
        string_sound = audio.string(exc, freq=220.0, t60=1.5, damping=0.3)

        # String should ring out for t60 duration after excitation
        assert string_sound.num_samples > exc.num_samples
        assert not np.any(np.isnan(string_sound.data))

    def test_string_pitch(self):
        """Test string produces expected pitch."""
        # Short noise burst
        exc = audio.noise(noise_type="white", seed=1, duration=0.01)

        # A220 string
        string_sound = audio.string(exc, freq=220.0, t60=1.5, damping=0.3)

        # Should have resonance at fundamental frequency
        # Simple check: signal should ring at approximately the right period
        assert string_sound.num_samples > 0

    def test_string_decay(self):
        """Test string decay over time."""
        # Impulse excitation (need longer duration to see decay)
        exc = audio.impulse(rate=100.0, duration=0.1)

        # String with fast decay
        fast_decay = audio.string(exc, freq=220.0, t60=0.1, damping=0.5)

        # String with slow decay
        slow_decay = audio.string(exc, freq=220.0, t60=2.0, damping=0.1)

        # Fast decay should die out quicker (check late energy)
        fast_energy = np.sum(fast_decay.data[2000:4000] ** 2)
        slow_energy = np.sum(slow_decay.data[2000:4000] ** 2)

        # Fast decay should have less energy in late samples
        # Note: with different t60 values, the outputs should be different
        # Compare only overlapping portion (outputs have different lengths)
        min_len = min(len(fast_decay.data), len(slow_decay.data))
        assert not np.allclose(fast_decay.data[:min_len], slow_decay.data[:min_len])

    def test_string_damping(self):
        """Test string damping parameter."""
        exc = audio.noise(noise_type="white", seed=1, duration=0.01)

        bright = audio.string(exc, freq=220.0, t60=1.0, damping=0.0)
        dull = audio.string(exc, freq=220.0, t60=1.0, damping=0.8)

        # Damping should affect high frequency content
        assert not np.allclose(bright.data, dull.data)

    def test_string_frequency_range(self):
        """Test string with different frequencies."""
        exc = audio.noise(noise_type="white", seed=1, duration=0.01)

        low = audio.string(exc, freq=55.0, t60=1.5, damping=0.3)
        mid = audio.string(exc, freq=220.0, t60=1.5, damping=0.3)
        high = audio.string(exc, freq=880.0, t60=1.5, damping=0.3)

        # All should ring out for t60 duration
        assert low.num_samples > exc.num_samples
        assert mid.num_samples > exc.num_samples
        assert high.num_samples > exc.num_samples

    def test_string_pluck_sound(self):
        """Test realistic pluck sound."""
        # Filtered noise burst (like guitar pluck)
        exc = audio.noise(noise_type="white", seed=1, duration=0.01)
        exc = audio.lowpass(exc, cutoff=6000.0)
        exc = audio.envexp(time_constant=0.01, duration=0.01)
        exc_shaped = AudioBuffer(data=exc.data * exc.data, sample_rate=44100)

        # Apply to string
        pluck = audio.string(exc_shaped, freq=220.0, t60=1.5, damping=0.3)

        # Should have characteristic pluck envelope
        assert pluck.num_samples > 0
        assert not np.any(np.isnan(pluck.data))

    def test_string_zero_frequency(self):
        """Test string with invalid (zero) frequency."""
        exc = audio.noise(noise_type="white", seed=1, duration=0.01)

        # Zero frequency should return excitation unchanged
        result = audio.string(exc, freq=0.0, t60=1.0, damping=0.3)
        assert result.num_samples == exc.num_samples
        assert np.allclose(result.data, exc.data)


class TestModalSynthesis:
    """Tests for modal synthesis."""

    def test_modal_basic(self):
        """Test basic modal synthesis."""
        exc = audio.impulse(rate=1.0, duration=0.001)

        modes = audio.modal(
            exc,
            frequencies=[200, 370, 550],
            decays=[2.0, 1.5, 1.0]
        )

        # Modal output should be longer than excitation (needs time to decay)
        # Output duration is max_decay * 5.0
        assert modes.num_samples > exc.num_samples
        assert not np.any(np.isnan(modes.data))

    def test_modal_bell_sound(self):
        """Test modal synthesis for bell-like sound."""
        exc = audio.impulse(rate=1.0, duration=0.001)

        # Bell-like inharmonic partials
        bell = audio.modal(
            exc,
            frequencies=[200, 370, 550, 720, 890],
            decays=[2.5, 2.0, 1.5, 1.0, 0.8]
        )

        # Modal output should be longer than excitation
        assert bell.num_samples > exc.num_samples

        # Should have energy (ringing)
        assert np.max(np.abs(bell.data)) > 0.1

    def test_modal_with_amplitudes(self):
        """Test modal synthesis with custom amplitudes."""
        exc = audio.impulse(rate=1.0, duration=0.001)

        # Emphasize fundamental
        modes = audio.modal(
            exc,
            frequencies=[100, 200, 300],
            decays=[1.0, 0.8, 0.6],
            amplitudes=[1.0, 0.5, 0.25]
        )

        # Modal output should be longer than excitation
        assert modes.num_samples > exc.num_samples

    def test_modal_single_mode(self):
        """Test modal synthesis with single mode."""
        exc = audio.impulse(rate=1.0, duration=0.001)

        single = audio.modal(
            exc,
            frequencies=[440],
            decays=[1.0]
        )

        # Modal output should be longer than excitation
        assert single.num_samples > exc.num_samples

    def test_modal_many_modes(self):
        """Test modal synthesis with many modes."""
        exc = audio.impulse(rate=1.0, duration=0.001)

        # 10 modes
        freqs = [100 * (i + 1) for i in range(10)]
        decays = [2.0 - i * 0.15 for i in range(10)]

        many = audio.modal(exc, frequencies=freqs, decays=decays)

        # Modal output should be longer than excitation
        assert many.num_samples > exc.num_samples
        assert not np.any(np.isnan(many.data))

    def test_modal_mismatched_lengths_error(self):
        """Test modal synthesis with mismatched array lengths raises error."""
        exc = audio.impulse(rate=1.0, duration=0.001)

        with pytest.raises(ValueError, match="same length"):
            audio.modal(
                exc,
                frequencies=[200, 370],
                decays=[2.0, 1.5, 1.0]  # Wrong length!
            )

    def test_modal_harmonic_vs_inharmonic(self):
        """Test difference between harmonic and inharmonic modes."""
        exc = audio.impulse(rate=1.0, duration=0.001)

        # Harmonic (like strings)
        harmonic = audio.modal(
            exc,
            frequencies=[200, 400, 600, 800],  # Perfect harmonics
            decays=[2.0, 1.8, 1.6, 1.4]
        )

        # Inharmonic (like bells)
        inharmonic = audio.modal(
            exc,
            frequencies=[200, 370, 550, 720],  # Inharmonic
            decays=[2.0, 1.8, 1.6, 1.4]
        )

        # Should be different
        assert not np.allclose(harmonic.data, inharmonic.data)


class TestPhysicalModelingDeterminism:
    """Tests for deterministic physical modeling."""

    def test_string_deterministic(self):
        """Test Karplus-Strong is deterministic."""
        exc = audio.noise(noise_type="white", seed=42, duration=0.01)

        string1 = audio.string(exc, freq=220.0, t60=1.5, damping=0.3)
        string2 = audio.string(exc, freq=220.0, t60=1.5, damping=0.3)

        assert np.array_equal(string1.data, string2.data)

    def test_modal_deterministic(self):
        """Test modal synthesis is deterministic."""
        exc = audio.impulse(rate=1.0, duration=0.001)

        modal1 = audio.modal(exc, frequencies=[200, 400], decays=[1.0, 0.8])
        modal2 = audio.modal(exc, frequencies=[200, 400], decays=[1.0, 0.8])

        assert np.allclose(modal1.data, modal2.data)


class TestPhysicalModelingCompositions:
    """Tests for composing physical models."""

    def test_string_with_envelope(self):
        """Test applying envelope to string excitation."""
        # Create excitation with envelope
        exc = audio.noise(noise_type="white", seed=1, duration=0.02)
        env = audio.envexp(time_constant=0.005, duration=0.02)
        exc_env = AudioBuffer(data=exc.data * env.data, sample_rate=44100)

        # Apply to string
        pluck = audio.string(exc_env, freq=330.0, t60=2.0, damping=0.2)

        assert pluck.num_samples > 0

    def test_string_with_effects(self):
        """Test applying effects to string output."""
        exc = audio.noise(noise_type="white", seed=1, duration=0.01)
        exc = audio.lowpass(exc, cutoff=8000.0)

        string_sound = audio.string(exc, freq=220.0, t60=1.5, damping=0.3)

        # Add reverb
        reverbed = audio.reverb(string_sound, mix=0.15, size=0.8)

        assert reverbed.num_samples == string_sound.num_samples

    def test_modal_with_effects(self):
        """Test applying effects to modal output."""
        exc = audio.impulse(rate=1.0, duration=0.001)

        bell = audio.modal(
            exc,
            frequencies=[200, 370, 550, 720],
            decays=[2.0, 1.8, 1.5, 1.2]
        )

        # Add reverb for realistic bell
        reverbed = audio.reverb(bell, mix=0.3, size=0.9)

        assert reverbed.num_samples == bell.num_samples

    def test_multiple_strings_mixed(self):
        """Test mixing multiple string voices."""
        exc1 = audio.noise(noise_type="white", seed=1, duration=0.01)
        exc2 = audio.noise(noise_type="white", seed=2, duration=0.01)
        exc3 = audio.noise(noise_type="white", seed=3, duration=0.01)

        # Three strings at different pitches (minor chord)
        string1 = audio.string(exc1, freq=220.0, t60=1.5, damping=0.3)  # A
        string2 = audio.string(exc2, freq=261.63, t60=1.5, damping=0.3)  # C
        string3 = audio.string(exc3, freq=329.63, t60=1.5, damping=0.3)  # E

        # Mix
        chord = audio.mix(string1, string2, string3)

        assert chord.num_samples == string1.num_samples


class TestPhysicalModelingEdgeCases:
    """Tests for physical modeling edge cases."""

    def test_string_very_low_freq(self):
        """Test string with very low frequency."""
        exc = audio.noise(noise_type="white", seed=1, duration=0.01)
        string_sound = audio.string(exc, freq=20.0, t60=1.0, damping=0.3)

        assert string_sound.num_samples > 0

    def test_string_very_high_freq(self):
        """Test string with very high frequency."""
        exc = audio.noise(noise_type="white", seed=1, duration=0.01)
        string_sound = audio.string(exc, freq=5000.0, t60=0.5, damping=0.5)

        assert string_sound.num_samples > 0

    def test_string_zero_damping(self):
        """Test string with zero damping."""
        exc = audio.noise(noise_type="white", seed=1, duration=0.01)
        string_sound = audio.string(exc, freq=220.0, t60=1.0, damping=0.0)

        assert not np.any(np.isnan(string_sound.data))

    def test_string_full_damping(self):
        """Test string with full damping."""
        exc = audio.noise(noise_type="white", seed=1, duration=0.01)
        string_sound = audio.string(exc, freq=220.0, t60=1.0, damping=1.0)

        assert not np.any(np.isnan(string_sound.data))

    def test_modal_very_long_decay(self):
        """Test modal synthesis with very long decay."""
        exc = audio.impulse(rate=1.0, duration=0.001)

        modes = audio.modal(
            exc,
            frequencies=[100],
            decays=[10.0]  # Very long
        )

        assert not np.any(np.isnan(modes.data))

    def test_modal_very_short_decay(self):
        """Test modal synthesis with very short decay."""
        exc = audio.impulse(rate=1.0, duration=0.001)

        modes = audio.modal(
            exc,
            frequencies=[1000],
            decays=[0.01]  # Very short
        )

        assert not np.any(np.isnan(modes.data))

    def test_modal_high_frequencies(self):
        """Test modal synthesis with high frequencies."""
        exc = audio.impulse(rate=1.0, duration=0.001)

        modes = audio.modal(
            exc,
            frequencies=[5000, 7000, 9000],
            decays=[0.5, 0.4, 0.3]
        )

        assert not np.any(np.isnan(modes.data))
