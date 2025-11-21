"""Unit tests for audio filter operations."""

import pytest
import numpy as np
from morphogen.stdlib.audio import audio, AudioBuffer


class TestLowpassFilter:
    """Tests for lowpass filter."""

    def test_lowpass_basic(self):
        """Test basic lowpass filter application."""
        sig = audio.sine(freq=440.0, duration=0.1)
        filtered = audio.lowpass(sig, cutoff=2000.0)
        assert filtered.num_samples == sig.num_samples
        assert filtered.sample_rate == sig.sample_rate

    def test_lowpass_removes_high_freq(self):
        """Test that lowpass removes high frequencies."""
        # Mix low and high frequency
        low = audio.sine(freq=100.0, duration=0.5)
        high = audio.sine(freq=5000.0, duration=0.5)
        mixed = audio.mix(low, high)

        # Filter with cutoff between the two
        filtered = audio.lowpass(mixed, cutoff=1000.0, q=0.707)

        # High frequency should be attenuated
        # Check by comparing frequency content (simple energy check)
        assert np.std(filtered.data) < np.std(mixed.data)

    def test_lowpass_preserves_low_freq(self):
        """Test that lowpass preserves low frequencies."""
        sig = audio.sine(freq=200.0, duration=0.5)
        filtered = audio.lowpass(sig, cutoff=2000.0)

        # Should be mostly preserved
        correlation = np.corrcoef(sig.data, filtered.data)[0, 1]
        assert correlation > 0.9

    def test_lowpass_q_factor(self):
        """Test Q factor affects resonance."""
        sig = audio.noise(noise_type="white", seed=42, duration=0.5)

        filt1 = audio.lowpass(sig, cutoff=1000.0, q=0.707)
        filt2 = audio.lowpass(sig, cutoff=1000.0, q=5.0)

        # Higher Q should have different response
        assert not np.allclose(filt1.data, filt2.data)


class TestHighpassFilter:
    """Tests for highpass filter."""

    def test_highpass_basic(self):
        """Test basic highpass filter application."""
        sig = audio.sine(freq=440.0, duration=0.1)
        filtered = audio.highpass(sig, cutoff=120.0)
        assert filtered.num_samples == sig.num_samples

    def test_highpass_removes_low_freq(self):
        """Test that highpass removes low frequencies."""
        low = audio.sine(freq=50.0, duration=0.5)
        high = audio.sine(freq=5000.0, duration=0.5)
        mixed = audio.mix(low, high)

        filtered = audio.highpass(mixed, cutoff=1000.0)

        # Low frequency should be attenuated
        assert np.std(filtered.data) < np.std(mixed.data)

    def test_highpass_preserves_high_freq(self):
        """Test that highpass preserves high frequencies."""
        sig = audio.sine(freq=5000.0, duration=0.5)
        filtered = audio.highpass(sig, cutoff=1000.0)

        # Should be mostly preserved
        correlation = np.corrcoef(sig.data, filtered.data)[0, 1]
        assert correlation > 0.8


class TestBandpassFilter:
    """Tests for bandpass filter."""

    def test_bandpass_basic(self):
        """Test basic bandpass filter application."""
        sig = audio.sine(freq=1000.0, duration=0.1)
        filtered = audio.bandpass(sig, center=1000.0, q=1.0)
        assert filtered.num_samples == sig.num_samples

    def test_bandpass_passes_center_freq(self):
        """Test that bandpass passes center frequency."""
        sig = audio.sine(freq=1000.0, duration=0.5)
        filtered = audio.bandpass(sig, center=1000.0, q=2.0)

        # Center frequency should pass through
        correlation = np.corrcoef(sig.data, filtered.data)[0, 1]
        assert correlation > 0.7

    def test_bandpass_rejects_outside_band(self):
        """Test that bandpass rejects frequencies outside band."""
        # Three frequencies: below, at, and above center
        low = audio.sine(freq=200.0, duration=0.5)
        mid = audio.sine(freq=1000.0, duration=0.5)
        high = audio.sine(freq=5000.0, duration=0.5)

        # Equal mix
        mixed = AudioBuffer(
            data=(low.data + mid.data + high.data) / 3.0,
            sample_rate=44100
        )

        filtered = audio.bandpass(mixed, center=1000.0, q=5.0)

        # Filtered should be mostly the middle frequency
        # Check that energy is reduced (rejecting low and high)
        assert np.std(filtered.data) < np.std(mixed.data)


class TestNotchFilter:
    """Tests for notch (band-stop) filter."""

    def test_notch_basic(self):
        """Test basic notch filter application."""
        sig = audio.sine(freq=1000.0, duration=0.1)
        filtered = audio.notch(sig, center=1000.0, q=1.0)
        assert filtered.num_samples == sig.num_samples

    def test_notch_removes_center_freq(self):
        """Test that notch removes center frequency."""
        sig = audio.sine(freq=1000.0, duration=0.5)
        filtered = audio.notch(sig, center=1000.0, q=5.0)

        # Should significantly attenuate the signal
        sig_energy = np.sum(sig.data ** 2)
        filt_energy = np.sum(filtered.data ** 2)
        assert filt_energy < sig_energy * 0.5

    def test_notch_preserves_other_freqs(self):
        """Test that notch preserves other frequencies."""
        sig = audio.sine(freq=500.0, duration=0.5)
        filtered = audio.notch(sig, center=1000.0, q=2.0)

        # Should mostly preserve signal far from notch
        correlation = np.corrcoef(sig.data, filtered.data)[0, 1]
        assert correlation > 0.8


class TestEQ3Band:
    """Tests for 3-band equalizer."""

    def test_eq3_basic(self):
        """Test basic 3-band EQ application."""
        sig = audio.sine(freq=440.0, duration=0.1)
        eq = audio.eq3(sig, bass=0.0, mid=0.0, treble=0.0)
        assert eq.num_samples == sig.num_samples

    def test_eq3_flat_preserves_signal(self):
        """Test that flat EQ (0dB all bands) preserves signal."""
        sig = audio.noise(noise_type="white", seed=42, duration=0.1)
        eq = audio.eq3(sig, bass=0.0, mid=0.0, treble=0.0)

        # Should be very close to original
        assert np.allclose(sig.data, eq.data, atol=0.1)

    def test_eq3_bass_boost(self):
        """Test bass boost."""
        sig = audio.noise(noise_type="white", seed=42, duration=0.5)
        eq = audio.eq3(sig, bass=6.0, mid=0.0, treble=0.0)

        # Bass boost should increase energy
        assert not np.allclose(sig.data, eq.data)

    def test_eq3_mid_boost(self):
        """Test mid boost."""
        sig = audio.noise(noise_type="white", seed=42, duration=0.5)
        eq = audio.eq3(sig, bass=0.0, mid=6.0, treble=0.0)

        assert not np.allclose(sig.data, eq.data)

    def test_eq3_treble_boost(self):
        """Test treble boost."""
        sig = audio.noise(noise_type="white", seed=42, duration=0.5)
        eq = audio.eq3(sig, bass=0.0, mid=0.0, treble=6.0)

        assert not np.allclose(sig.data, eq.data)

    def test_eq3_all_bands(self):
        """Test adjusting all three bands."""
        sig = audio.noise(noise_type="white", seed=42, duration=0.5)
        eq = audio.eq3(sig, bass=3.0, mid=-3.0, treble=6.0)

        assert eq.num_samples == sig.num_samples

    def test_eq3_cut(self):
        """Test EQ cuts (negative gain)."""
        sig = audio.noise(noise_type="white", seed=42, duration=0.5)
        eq = audio.eq3(sig, bass=-6.0, mid=-6.0, treble=-6.0)

        # Overall energy should be reduced
        sig_energy = np.sum(sig.data ** 2)
        eq_energy = np.sum(eq.data ** 2)
        assert eq_energy < sig_energy


class TestFilterStability:
    """Tests for filter stability and numerical issues."""

    def test_lowpass_stability(self):
        """Test lowpass filter doesn't produce NaN or Inf."""
        sig = audio.sine(freq=440.0, duration=1.0)
        filtered = audio.lowpass(sig, cutoff=100.0, q=0.707)

        assert not np.any(np.isnan(filtered.data))
        assert not np.any(np.isinf(filtered.data))

    def test_highpass_stability(self):
        """Test highpass filter doesn't produce NaN or Inf."""
        sig = audio.sine(freq=440.0, duration=1.0)
        filtered = audio.highpass(sig, cutoff=10000.0, q=0.707)

        assert not np.any(np.isnan(filtered.data))
        assert not np.any(np.isinf(filtered.data))

    def test_bandpass_stability(self):
        """Test bandpass filter doesn't produce NaN or Inf."""
        sig = audio.sine(freq=440.0, duration=1.0)
        filtered = audio.bandpass(sig, center=1000.0, q=10.0)

        assert not np.any(np.isnan(filtered.data))
        assert not np.any(np.isinf(filtered.data))

    def test_filter_with_silence(self):
        """Test filters handle silence correctly."""
        sig = AudioBuffer(data=np.zeros(1000), sample_rate=44100)

        lpf = audio.lowpass(sig, cutoff=1000.0)
        hpf = audio.highpass(sig, cutoff=1000.0)
        bpf = audio.bandpass(sig, center=1000.0)

        assert np.all(lpf.data == 0.0)
        assert np.all(hpf.data == 0.0)
        assert np.all(bpf.data == 0.0)


class TestFilterChaining:
    """Tests for chaining multiple filters."""

    def test_lowpass_chain(self):
        """Test chaining multiple lowpass filters."""
        sig = audio.sine(freq=440.0, duration=0.5)

        filtered = audio.lowpass(sig, cutoff=2000.0)
        filtered = audio.lowpass(filtered, cutoff=1000.0)

        assert filtered.num_samples == sig.num_samples
        assert not np.any(np.isnan(filtered.data))

    def test_highpass_lowpass_chain(self):
        """Test chaining highpass and lowpass (bandpass effect)."""
        sig = audio.noise(noise_type="white", seed=42, duration=0.5)

        # HPF at 200Hz, LPF at 2000Hz = bandpass 200-2000Hz
        filtered = audio.highpass(sig, cutoff=200.0)
        filtered = audio.lowpass(filtered, cutoff=2000.0)

        assert filtered.num_samples == sig.num_samples

    def test_eq_after_filter(self):
        """Test EQ after filter."""
        sig = audio.sine(freq=440.0, duration=0.5)

        filtered = audio.lowpass(sig, cutoff=2000.0)
        eq = audio.eq3(filtered, bass=3.0, mid=0.0, treble=-3.0)

        assert eq.num_samples == sig.num_samples


class TestFilterDeterminism:
    """Tests for deterministic filter behavior."""

    def test_lowpass_deterministic(self):
        """Test lowpass filter is deterministic."""
        sig = audio.sine(freq=440.0, duration=0.1)

        filt1 = audio.lowpass(sig, cutoff=1000.0)
        filt2 = audio.lowpass(sig, cutoff=1000.0)

        assert np.array_equal(filt1.data, filt2.data)

    def test_highpass_deterministic(self):
        """Test highpass filter is deterministic."""
        sig = audio.sine(freq=440.0, duration=0.1)

        filt1 = audio.highpass(sig, cutoff=1000.0)
        filt2 = audio.highpass(sig, cutoff=1000.0)

        assert np.array_equal(filt1.data, filt2.data)

    def test_eq3_deterministic(self):
        """Test 3-band EQ is deterministic."""
        sig = audio.noise(noise_type="white", seed=42, duration=0.1)

        eq1 = audio.eq3(sig, bass=3.0, mid=-3.0, treble=6.0)
        eq2 = audio.eq3(sig, bass=3.0, mid=-3.0, treble=6.0)

        assert np.allclose(eq1.data, eq2.data)


class TestFilterEdgeCases:
    """Tests for filter edge cases."""

    def test_very_low_cutoff(self):
        """Test filters with very low cutoff."""
        sig = audio.sine(freq=440.0, duration=0.1)
        filtered = audio.lowpass(sig, cutoff=10.0)

        # Should heavily attenuate
        assert np.max(np.abs(filtered.data)) < np.max(np.abs(sig.data))

    def test_very_high_cutoff(self):
        """Test filters with very high cutoff."""
        sig = audio.sine(freq=440.0, duration=0.1)
        filtered = audio.lowpass(sig, cutoff=20000.0)

        # Should mostly pass through
        correlation = np.corrcoef(sig.data, filtered.data)[0, 1]
        assert correlation > 0.95

    def test_nyquist_frequency(self):
        """Test filter at Nyquist frequency."""
        sig = audio.sine(freq=440.0, duration=0.1, sample_rate=44100)
        # Nyquist is 22050 Hz
        filtered = audio.lowpass(sig, cutoff=22050.0)

        assert not np.any(np.isnan(filtered.data))
