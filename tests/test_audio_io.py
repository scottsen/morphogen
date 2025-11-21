"""Tests for audio I/O operations (v0.6.0)."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from morphogen.stdlib.audio import AudioBuffer, AudioOperations, audio


class TestAudioIO:
    """Test audio I/O operations."""

    def test_save_wav_mono(self):
        """Test WAV export for mono audio."""
        # Generate test audio
        tone = audio.sine(freq=440.0, duration=0.1)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            path = f.name

        try:
            # Save audio
            audio.save(tone, path)

            # Verify file exists
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_save_wav_stereo(self):
        """Test WAV export for stereo audio."""
        # Generate stereo test audio
        left = audio.sine(freq=440.0, duration=0.1)
        right = audio.sine(freq=554.37, duration=0.1)  # C#5

        # Create stereo buffer
        stereo_data = np.column_stack([left.data, right.data])
        stereo = AudioBuffer(data=stereo_data, sample_rate=44100)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            path = f.name

        try:
            audio.save(stereo, path)
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_save_flac_mono(self):
        """Test FLAC export for mono audio."""
        pytest.importorskip("soundfile")

        tone = audio.sine(freq=440.0, duration=0.1)

        with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as f:
            path = f.name

        try:
            audio.save(tone, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_save_auto_format(self):
        """Test automatic format detection from extension."""
        tone = audio.sine(freq=440.0, duration=0.1)

        # Test .wav
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wav_path = f.name

        try:
            audio.save(tone, wav_path, format="auto")
            assert os.path.exists(wav_path)
        finally:
            if os.path.exists(wav_path):
                os.unlink(wav_path)

    def test_save_explicit_format(self):
        """Test explicit format specification."""
        tone = audio.sine(freq=440.0, duration=0.1)

        with tempfile.NamedTemporaryFile(suffix='.audio', delete=False) as f:
            path = f.name

        try:
            audio.save(tone, path, format="wav")
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_save_invalid_format(self):
        """Test error handling for invalid format."""
        tone = audio.sine(freq=440.0, duration=0.1)

        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported format"):
                audio.save(tone, path, format="mp3")
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_save_invalid_buffer(self):
        """Test error handling for invalid buffer type."""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            path = f.name

        try:
            with pytest.raises(TypeError, match="Expected AudioBuffer"):
                audio.save("not a buffer", path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_load_wav_mono(self):
        """Test loading mono WAV file."""
        # Create and save test audio
        original = audio.sine(freq=440.0, duration=0.1)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            path = f.name

        try:
            audio.save(original, path)

            # Load audio
            loaded = audio.load(path)

            # Verify properties
            assert isinstance(loaded, AudioBuffer)
            assert loaded.sample_rate == original.sample_rate
            assert loaded.num_samples == original.num_samples
            assert not loaded.is_stereo

            # Check similarity (allowing for codec precision)
            correlation = np.corrcoef(original.data, loaded.data)[0, 1]
            assert correlation > 0.99
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_load_wav_stereo(self):
        """Test loading stereo WAV file."""
        pytest.importorskip("soundfile")

        # Create stereo test audio
        left = audio.sine(freq=440.0, duration=0.1)
        right = audio.sine(freq=554.37, duration=0.1)
        stereo_data = np.column_stack([left.data, right.data])
        original = AudioBuffer(data=stereo_data, sample_rate=44100)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            path = f.name

        try:
            audio.save(original, path)
            loaded = audio.load(path)

            assert loaded.is_stereo
            assert loaded.sample_rate == original.sample_rate
            assert loaded.data.shape == original.data.shape
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_load_flac(self):
        """Test loading FLAC file."""
        pytest.importorskip("soundfile")

        original = audio.sine(freq=440.0, duration=0.1)

        with tempfile.NamedTemporaryFile(suffix='.flac', delete=False) as f:
            path = f.name

        try:
            audio.save(original, path)
            loaded = audio.load(path)

            assert isinstance(loaded, AudioBuffer)
            assert loaded.sample_rate == original.sample_rate

            # FLAC is lossless, should be very close
            correlation = np.corrcoef(original.data, loaded.data)[0, 1]
            assert correlation > 0.99
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_load_nonexistent_file(self):
        """Test error handling for missing file."""
        with pytest.raises((FileNotFoundError, RuntimeError)):
            audio.load("/nonexistent/path/to/audio.wav")

    def test_save_load_roundtrip(self):
        """Test save/load preserves audio data."""
        # Create complex test signal
        tone1 = audio.sine(freq=440.0, duration=0.1)
        tone2 = audio.sine(freq=554.37, duration=0.1)
        original = audio.mix(tone1, tone2)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            path = f.name

        try:
            audio.save(original, path)
            loaded = audio.load(path)

            # Check correlation (high correlation = waveform preserved)
            correlation = np.corrcoef(original.data, loaded.data)[0, 1]
            assert correlation > 0.99

            # Check energy preservation (allow 20% difference due to 16-bit quantization)
            # Note: scipy.io.wavfile uses int16, which has limited precision
            orig_energy = np.sum(original.data ** 2)
            load_energy = np.sum(loaded.data ** 2)
            energy_diff = abs(orig_energy - load_energy) / max(orig_energy, 1e-10)
            assert energy_diff < 0.2  # 20% tolerance for int16 codec
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_save_clipping(self):
        """Test that save clips values to [-1, 1]."""
        # Create buffer with values outside [-1, 1]
        data = np.array([0.5, 1.5, -1.5, 0.0], dtype=np.float32)
        buffer = AudioBuffer(data=data, sample_rate=44100)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            path = f.name

        try:
            audio.save(buffer, path)
            loaded = audio.load(path)

            # Check clipping occurred
            assert np.max(np.abs(loaded.data)) <= 1.0
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_play_type_validation(self):
        """Test play() validates input type."""
        with pytest.raises(TypeError, match="Expected AudioBuffer"):
            audio.play("not a buffer")

    def test_record_validation(self):
        """Test record() validates parameters."""
        with pytest.raises(ValueError, match="channels must be"):
            audio.record(duration=0.1, channels=3)

    def test_audio_formats_compatibility(self):
        """Test compatibility across different audio formats."""
        pytest.importorskip("soundfile")

        original = audio.sine(freq=440.0, duration=0.1)

        formats = [
            ('.wav', 'wav'),
            ('.flac', 'flac'),
        ]

        for ext, fmt in formats:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                path = f.name

            try:
                audio.save(original, path, format=fmt)
                loaded = audio.load(path)

                # All formats should preserve basic properties
                assert loaded.sample_rate == original.sample_rate
                assert abs(loaded.duration - original.duration) < 0.01

                # Check signal correlation
                correlation = np.corrcoef(original.data, loaded.data)[0, 1]
                assert correlation > 0.95  # Allow for format differences
            finally:
                if os.path.exists(path):
                    os.unlink(path)

    def test_different_sample_rates(self):
        """Test save/load with different sample rates."""
        rates = [22050, 44100, 48000]

        for rate in rates:
            original = audio.sine(freq=440.0, duration=0.1, sample_rate=rate)

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                path = f.name

            try:
                audio.save(original, path)
                loaded = audio.load(path)

                assert loaded.sample_rate == rate
                assert abs(loaded.duration - original.duration) < 0.01
            finally:
                if os.path.exists(path):
                    os.unlink(path)

    def test_save_processed_audio(self):
        """Test saving audio after processing."""
        # Create and process audio
        tone = audio.sine(freq=440.0, duration=0.2)
        filtered = audio.lowpass(tone, cutoff=2000.0)
        reverbed = audio.reverb(filtered, mix=0.3)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            path = f.name

        try:
            audio.save(reverbed, path)
            loaded = audio.load(path)

            # Verify processing was preserved
            assert loaded.sample_rate == reverbed.sample_rate
            correlation = np.corrcoef(reverbed.data, loaded.data)[0, 1]
            assert correlation > 0.99
        finally:
            if os.path.exists(path):
                os.unlink(path)
