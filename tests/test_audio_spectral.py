"""Unit tests for audio spectral operations (FFT, analysis, processing)."""

import pytest
import numpy as np
from morphogen.stdlib.audio import audio, AudioBuffer


class TestFFTOperations:
    """Tests for FFT and spectral transforms."""

    def test_fft_basic(self):
        """Test basic FFT computation."""
        # Create 440 Hz sine wave
        buf = audio.sine(freq=440.0, duration=1.0)

        freqs, spectrum = audio.fft(buf)
        assert len(freqs) == len(spectrum)
        assert freqs[0] == 0.0  # DC component
        assert freqs[-1] <= buf.sample_rate / 2  # Nyquist

    def test_fft_peak_detection(self):
        """Test that FFT detects correct frequency peak."""
        # Create pure 1000 Hz tone
        buf = audio.sine(freq=1000.0, duration=1.0)

        freqs, spectrum = audio.fft(buf)
        magnitudes = np.abs(spectrum)

        # Find peak
        peak_idx = np.argmax(magnitudes)
        peak_freq = freqs[peak_idx]

        # Should be close to 1000 Hz
        assert abs(peak_freq - 1000.0) < 10.0

    def test_fft_stereo_raises_error(self):
        """Test that FFT on stereo signal raises error."""
        mono = audio.sine(freq=440.0, duration=0.1)
        stereo = audio.pan(mono, position=0.0)

        with pytest.raises(ValueError):
            audio.fft(stereo)

    def test_ifft_reconstruction(self):
        """Test that IFFT reconstructs original signal."""
        buf = audio.sine(freq=440.0, duration=0.1)

        # FFT and IFFT
        freqs, spectrum = audio.fft(buf)
        reconstructed = audio.ifft(spectrum, buf.sample_rate)

        # Should match original (within floating point error)
        np.testing.assert_array_almost_equal(buf.data, reconstructed.data, decimal=5)

    def test_spectrum_basic(self):
        """Test magnitude spectrum computation."""
        buf = audio.sine(freq=440.0, duration=1.0)

        freqs, magnitudes = audio.spectrum(buf)
        assert len(freqs) == len(magnitudes)
        assert np.all(magnitudes >= 0)  # Magnitudes are non-negative

    def test_phase_spectrum_basic(self):
        """Test phase spectrum computation."""
        buf = audio.sine(freq=440.0, duration=1.0)

        freqs, phases = audio.phase_spectrum(buf)
        assert len(freqs) == len(phases)
        # Phases should be in range [-π, π]
        assert np.all(phases >= -np.pi)
        assert np.all(phases <= np.pi)

    def test_stft_basic(self):
        """Test Short-Time Fourier Transform."""
        buf = audio.sine(freq=440.0, duration=1.0)

        times, freqs, stft_matrix = audio.stft(buf, window_size=2048, hop_size=512)

        # Check dimensions
        assert len(times) == stft_matrix.shape[1]
        assert len(freqs) == stft_matrix.shape[0]
        assert stft_matrix.dtype == np.complex128

    def test_stft_time_axis(self):
        """Test STFT time axis is correct."""
        buf = audio.sine(freq=440.0, duration=1.0)

        times, freqs, stft_matrix = audio.stft(buf, hop_size=512)

        # Times should span roughly the buffer duration
        assert times[0] >= 0.0
        assert times[-1] <= buf.duration

    def test_istft_reconstruction(self):
        """Test that ISTFT reconstructs signal."""
        buf = audio.sine(freq=440.0, duration=1.0)

        # STFT and ISTFT
        times, freqs, stft_matrix = audio.stft(buf, hop_size=512)
        reconstructed = audio.istft(stft_matrix, hop_size=512, sample_rate=buf.sample_rate)

        # Should be close (some error due to windowing)
        # Compare middle portion to avoid edge effects
        mid_start = buf.num_samples // 4
        mid_end = 3 * buf.num_samples // 4
        np.testing.assert_array_almost_equal(
            buf.data[mid_start:mid_end],
            reconstructed.data[mid_start:mid_end],
            decimal=2
        )


class TestSpectralAnalysis:
    """Tests for spectral analysis operations."""

    def test_spectral_centroid_basic(self):
        """Test spectral centroid calculation."""
        # Low frequency sine wave
        buf = audio.sine(freq=200.0, duration=1.0)
        centroid = audio.spectral_centroid(buf)

        # Centroid should be around 200 Hz
        assert 150 < centroid < 250

    def test_spectral_centroid_high_vs_low(self):
        """Test that high freq has higher centroid than low freq."""
        low = audio.sine(freq=200.0, duration=1.0)
        high = audio.sine(freq=2000.0, duration=1.0)

        centroid_low = audio.spectral_centroid(low)
        centroid_high = audio.spectral_centroid(high)

        assert centroid_high > centroid_low

    def test_spectral_centroid_noise(self):
        """Test spectral centroid on noise."""
        noise = audio.noise(noise_type="white", seed=42, duration=1.0)
        centroid = audio.spectral_centroid(noise)

        # White noise should have centroid around middle of spectrum
        assert 5000 < centroid < 15000

    def test_spectral_rolloff_basic(self):
        """Test spectral rolloff calculation."""
        buf = audio.sine(freq=1000.0, duration=1.0)
        rolloff = audio.spectral_rolloff(buf, threshold=0.85)

        # For pure tone, rolloff should be near the tone frequency
        assert 800 < rolloff < 1200

    def test_spectral_rolloff_threshold(self):
        """Test that higher threshold gives higher rolloff."""
        buf = audio.noise(noise_type="white", seed=42, duration=1.0)

        rolloff_50 = audio.spectral_rolloff(buf, threshold=0.5)
        rolloff_90 = audio.spectral_rolloff(buf, threshold=0.9)

        assert rolloff_90 > rolloff_50

    def test_spectral_flux_basic(self):
        """Test spectral flux calculation."""
        buf = audio.sine(freq=440.0, duration=1.0)
        flux = audio.spectral_flux(buf)

        # Flux should be an array
        assert isinstance(flux, np.ndarray)
        assert len(flux) > 0
        assert np.all(flux >= 0)  # Flux is non-negative

    def test_spectral_flux_onset_detection(self):
        """Test spectral flux for onset detection."""
        # Create buffer with distinct onset
        silence = AudioBuffer(data=np.zeros(22050), sample_rate=44100)
        tone = audio.sine(freq=440.0, duration=0.5, sample_rate=44100)
        buf = audio.concat(silence, tone)

        flux = audio.spectral_flux(buf)

        # There should be a spike in flux at the onset
        max_flux_idx = np.argmax(flux)
        # Should be somewhere around the middle (where tone starts)
        assert 0.3 * len(flux) < max_flux_idx < 0.7 * len(flux)

    def test_spectral_peaks_basic(self):
        """Test spectral peak finding."""
        buf = audio.sine(freq=440.0, duration=1.0)

        peak_freqs, peak_mags = audio.spectral_peaks(buf, num_peaks=3)

        # Should find peaks
        assert len(peak_freqs) >= 1
        assert len(peak_freqs) == len(peak_mags)

        # Strongest peak should be near 440 Hz
        assert abs(peak_freqs[0] - 440.0) < 50.0

    def test_spectral_peaks_multiple_tones(self):
        """Test finding peaks with multiple tones."""
        # Create multi-tone signal
        tone1 = audio.sine(freq=200.0, duration=1.0)
        tone2 = audio.sine(freq=600.0, duration=1.0)
        tone3 = audio.sine(freq=1200.0, duration=1.0)
        buf = audio.mix(tone1, tone2, tone3)

        peak_freqs, peak_mags = audio.spectral_peaks(buf, num_peaks=5)

        # Should find peaks near the three frequencies
        # Check if frequencies 200, 600, 1200 are in the vicinity
        assert any(abs(f - 200) < 50 for f in peak_freqs)
        assert any(abs(f - 600) < 50 for f in peak_freqs)
        assert any(abs(f - 1200) < 50 for f in peak_freqs)

    def test_rms_basic(self):
        """Test RMS level calculation."""
        # Silent buffer
        silence = AudioBuffer(data=np.zeros(1000), sample_rate=44100)
        assert audio.rms(silence) < 0.001

        # Full scale sine wave
        loud = audio.sine(freq=440.0, duration=1.0)
        rms_val = audio.rms(loud)

        # Sine wave RMS should be approximately 1/sqrt(2) = 0.707
        assert 0.65 < rms_val < 0.75

    def test_rms_stereo(self):
        """Test RMS on stereo buffer."""
        mono = audio.sine(freq=440.0, duration=1.0)
        stereo = audio.pan(mono, position=0.0)

        rms_val = audio.rms(stereo)
        # Panned stereo preserves energy, RMS should be similar to mono
        assert 0.4 < rms_val < 0.8

    def test_rms_db_conversion(self):
        """Test RMS to dB conversion."""
        buf = audio.sine(freq=440.0, duration=1.0)
        rms_val = audio.rms(buf)
        rms_db = audio.lin2db(rms_val)

        # Should be close to -3 dB for sine wave
        assert -4 < rms_db < -2

    def test_zero_crossings_basic(self):
        """Test zero crossing count."""
        # Sine wave at 440 Hz for 1 second should have ~880 crossings
        buf = audio.sine(freq=440.0, duration=1.0)
        zcr = audio.zero_crossings(buf)

        # Should be close to 2 * freq * duration
        expected = 2 * 440 * 1.0
        assert abs(zcr - expected) < 50

    def test_zero_crossings_dc(self):
        """Test zero crossings on DC signal."""
        # Constant signal should have no crossings
        buf = AudioBuffer(data=np.ones(1000), sample_rate=44100)
        zcr = audio.zero_crossings(buf)
        assert zcr == 0

    def test_zero_crossings_noise(self):
        """Test zero crossings on noise."""
        # Noise should have many zero crossings
        noise = audio.noise(noise_type="white", seed=42, duration=1.0)
        zcr = audio.zero_crossings(noise)

        # White noise should cross zero many times
        assert zcr > 10000


class TestSpectralProcessing:
    """Tests for spectral processing operations."""

    def test_spectral_gate_basic(self):
        """Test spectral noise gate."""
        # Create signal with noise
        tone = audio.sine(freq=440.0, duration=1.0)
        noise = audio.noise(noise_type="white", seed=42, duration=1.0)
        noise = audio.gain(noise, amount_db=-40.0)  # Very quiet noise
        noisy = audio.mix(tone, noise)

        # Apply spectral gate with higher threshold
        cleaned = audio.spectral_gate(noisy, threshold_db=-50.0)

        # Both should have similar energy (mostly preserving tone)
        rms_noisy = audio.rms(noisy)
        rms_cleaned = audio.rms(cleaned)
        # Gate should preserve most of the signal
        assert 0.8 * rms_noisy < rms_cleaned <= 1.1 * rms_noisy

    def test_spectral_filter_bandpass(self):
        """Test spectral filter as bandpass."""
        # Create broadband signal
        noise = audio.noise(noise_type="white", seed=42, duration=1.0)

        # Create bandpass mask (500-1500 Hz)
        freqs, _ = audio.fft(noise)
        mask = ((freqs >= 500) & (freqs <= 1500)).astype(float)

        # Apply filter
        filtered = audio.spectral_filter(noise, mask)

        # Check that energy is concentrated in passband
        freqs_filt, mags_filt = audio.spectrum(filtered)

        # Energy below 500 Hz should be reduced
        low_energy = np.sum(mags_filt[freqs_filt < 500] ** 2)
        mid_energy = np.sum(mags_filt[(freqs_filt >= 500) & (freqs_filt <= 1500)] ** 2)

        assert mid_energy > low_energy

    def test_spectral_filter_stereo_raises_error(self):
        """Test that spectral operations on stereo raise error."""
        mono = audio.sine(freq=440.0, duration=0.1)
        stereo = audio.pan(mono, position=0.0)

        freqs, _ = audio.fft(mono)
        mask = np.ones(len(freqs))

        with pytest.raises(ValueError):
            audio.spectral_filter(stereo, mask)

    def test_convolution_basic(self):
        """Test convolution operation."""
        # Create signal
        signal = audio.sine(freq=440.0, duration=0.5)

        # Create simple impulse response (delay)
        impulse = AudioBuffer(data=np.zeros(1000), sample_rate=44100)
        impulse.data[0] = 0.7  # Direct signal
        impulse.data[500] = 0.3  # Delayed reflection

        # Convolve
        result = audio.convolution(signal, impulse)

        # Result should be longer than input
        assert result.num_samples >= signal.num_samples

    def test_convolution_impulse_identity(self):
        """Test that convolution with unit impulse is identity."""
        signal = audio.sine(freq=440.0, duration=0.5)

        # Unit impulse at start
        impulse = AudioBuffer(data=np.zeros(10), sample_rate=44100)
        impulse.data[0] = 1.0

        result = audio.convolution(signal, impulse)

        # Should be approximately same as input (within precision)
        min_len = min(len(signal.data), len(result.data))
        np.testing.assert_array_almost_equal(
            signal.data[:min_len],
            result.data[:min_len],
            decimal=5
        )

    def test_convolution_normalization(self):
        """Test that convolution normalizes to prevent clipping."""
        signal = audio.sine(freq=440.0, duration=0.5)

        # Large impulse response
        impulse = audio.sine(freq=100.0, duration=0.1)

        result = audio.convolution(signal, impulse)

        # Result should not clip
        assert np.max(np.abs(result.data)) <= 1.0


class TestSpectralWorkflows:
    """Integration tests for spectral processing workflows."""

    def test_fft_modify_ifft(self):
        """Test workflow of FFT → modify → IFFT."""
        # Use a signal with high frequency content
        low = audio.sine(freq=200.0, duration=1.0)
        high = audio.sine(freq=8000.0, duration=1.0)
        buf = audio.mix(low, high)

        # Get FFT
        freqs, spectrum = audio.fft(buf)

        # Zero out high frequencies (lowpass at 2kHz)
        spectrum[freqs > 2000] = 0

        # Reconstruct
        filtered = audio.ifft(spectrum, buf.sample_rate)

        # Should have less high frequency content
        centroid_orig = audio.spectral_centroid(buf)
        centroid_filt = audio.spectral_centroid(filtered)
        assert centroid_filt < centroid_orig

    def test_stft_modify_istft(self):
        """Test workflow of STFT → modify → ISTFT."""
        buf = audio.sine(freq=440.0, duration=1.0)

        # Get STFT
        times, freqs, stft_matrix = audio.stft(buf)

        # Reduce magnitude by half
        magnitude = np.abs(stft_matrix)
        phase = np.angle(stft_matrix)
        stft_modified = (magnitude * 0.5) * np.exp(1j * phase)

        # Reconstruct
        quieter = audio.istft(stft_modified, sample_rate=buf.sample_rate)

        # RMS should be lower
        assert audio.rms(quieter) < audio.rms(buf)

    def test_spectral_analysis_pipeline(self):
        """Test complete spectral analysis pipeline."""
        # Create complex signal
        bass = audio.sine(freq=100.0, duration=1.0)
        mid = audio.sine(freq=1000.0, duration=1.0)
        high = audio.sine(freq=5000.0, duration=1.0)
        signal = audio.mix(bass, mid, high)

        # Analyze
        centroid = audio.spectral_centroid(signal)
        rolloff = audio.spectral_rolloff(signal)
        peak_freqs, peak_mags = audio.spectral_peaks(signal, num_peaks=5)
        rms_val = audio.rms(signal)
        zcr = audio.zero_crossings(signal)

        # Basic sanity checks
        assert 100 < centroid < 10000
        assert centroid < rolloff
        assert len(peak_freqs) >= 3
        assert 0.3 < rms_val < 0.8
        assert zcr > 0
