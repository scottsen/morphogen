"""Unit tests for audio buffer operations."""

import pytest
import numpy as np
from morphogen.stdlib.audio import audio, AudioBuffer


class TestBufferOperations:
    """Tests for buffer manipulation operations."""

    def test_slice_basic(self):
        """Test basic buffer slicing."""
        # Create 1 second buffer
        buf = audio.sine(freq=440.0, duration=1.0)

        # Slice middle 0.5 seconds
        sliced = audio.slice(buf, start=0.25, end=0.75)
        assert abs(sliced.duration - 0.5) < 0.01
        assert sliced.sample_rate == buf.sample_rate

    def test_slice_start_only(self):
        """Test slicing with only start time."""
        buf = audio.sine(freq=440.0, duration=1.0)
        sliced = audio.slice(buf, start=0.5)
        assert abs(sliced.duration - 0.5) < 0.01

    def test_slice_boundary_clamping(self):
        """Test that slice clamps to valid boundaries."""
        buf = audio.sine(freq=440.0, duration=1.0)

        # Try to slice beyond buffer
        sliced = audio.slice(buf, start=-1.0, end=2.0)
        assert sliced.num_samples == buf.num_samples

    def test_concat_basic(self):
        """Test concatenating buffers."""
        buf1 = audio.sine(freq=440.0, duration=0.5)
        buf2 = audio.sine(freq=880.0, duration=0.5)

        combined = audio.concat(buf1, buf2)
        assert abs(combined.duration - 1.0) < 0.01
        assert combined.sample_rate == buf1.sample_rate

    def test_concat_multiple(self):
        """Test concatenating multiple buffers."""
        bufs = [audio.sine(freq=440.0, duration=0.2) for _ in range(5)]
        combined = audio.concat(*bufs)
        assert abs(combined.duration - 1.0) < 0.01

    def test_concat_different_sample_rates(self):
        """Test that concatenating different sample rates raises error."""
        buf1 = AudioBuffer(data=np.zeros(1000), sample_rate=44100)
        buf2 = AudioBuffer(data=np.zeros(1000), sample_rate=48000)

        with pytest.raises(ValueError):
            audio.concat(buf1, buf2)

    def test_resample_basic(self):
        """Test resampling to different sample rate."""
        buf = audio.sine(freq=440.0, duration=1.0, sample_rate=44100)

        # Resample to 48kHz
        resampled = audio.resample(buf, new_sample_rate=48000)
        assert resampled.sample_rate == 48000
        # Duration should be approximately same
        assert abs(resampled.duration - 1.0) < 0.01

    def test_resample_same_rate(self):
        """Test that resampling to same rate returns copy."""
        buf = audio.sine(freq=440.0, duration=1.0)
        resampled = audio.resample(buf, new_sample_rate=44100)

        assert resampled.sample_rate == buf.sample_rate
        assert resampled.num_samples == buf.num_samples
        # Should be a copy
        assert resampled.data is not buf.data

    def test_resample_stereo(self):
        """Test resampling stereo buffer."""
        mono = audio.sine(freq=440.0, duration=1.0)
        stereo = audio.pan(mono, position=0.0)

        resampled = audio.resample(stereo, new_sample_rate=48000)
        assert resampled.sample_rate == 48000
        assert resampled.is_stereo

    def test_reverse_basic(self):
        """Test reversing a buffer."""
        # Create impulse at start
        buf = AudioBuffer(data=np.zeros(1000), sample_rate=44100)
        buf.data[100] = 1.0

        reversed_buf = audio.reverse(buf)
        # Impulse should now be near end
        assert reversed_buf.data[899] == 1.0
        assert reversed_buf.sample_rate == buf.sample_rate

    def test_reverse_twice_is_identity(self):
        """Test that reversing twice gives original."""
        buf = audio.sine(freq=440.0, duration=0.1)
        reversed_twice = audio.reverse(audio.reverse(buf))

        np.testing.assert_array_almost_equal(buf.data, reversed_twice.data)

    def test_fade_in_basic(self):
        """Test fade-in envelope."""
        # Create constant buffer
        buf = AudioBuffer(data=np.ones(44100), sample_rate=44100)

        faded = audio.fade_in(buf, duration=0.1)
        # Start should be near zero
        assert faded.data[0] < 0.01
        # End should be at full level
        assert faded.data[-1] > 0.99

    def test_fade_out_basic(self):
        """Test fade-out envelope."""
        buf = AudioBuffer(data=np.ones(44100), sample_rate=44100)

        faded = audio.fade_out(buf, duration=0.1)
        # Start should be at full level
        assert faded.data[0] > 0.99
        # End should be near zero
        assert faded.data[-1] < 0.01

    def test_fade_stereo(self):
        """Test fading stereo buffer."""
        mono = audio.sine(freq=440.0, duration=1.0)
        stereo = audio.pan(mono, position=0.0)

        faded = audio.fade_in(stereo, duration=0.1)
        assert faded.is_stereo
        # Both channels should be faded
        assert faded.data[0, 0] < 0.01
        assert faded.data[0, 1] < 0.01

    def test_fade_long_duration(self):
        """Test that fade duration is clamped to buffer length."""
        buf = audio.sine(freq=440.0, duration=0.1)

        # Request fade longer than buffer
        faded = audio.fade_in(buf, duration=1.0)
        assert faded.num_samples == buf.num_samples


class TestBufferWorkflows:
    """Integration tests for buffer operation workflows."""

    def test_slice_and_concat(self):
        """Test slicing and concatenating workflow."""
        # Create 2 second buffer
        buf = audio.sine(freq=440.0, duration=2.0)

        # Slice into quarters
        q1 = audio.slice(buf, start=0.0, end=0.5)
        q2 = audio.slice(buf, start=0.5, end=1.0)
        q3 = audio.slice(buf, start=1.0, end=1.5)
        q4 = audio.slice(buf, start=1.5, end=2.0)

        # Concatenate in different order
        rearranged = audio.concat(q4, q2, q3, q1)
        assert abs(rearranged.duration - 2.0) < 0.01

    def test_resample_chain(self):
        """Test chain of resampling operations."""
        buf = audio.sine(freq=440.0, duration=1.0, sample_rate=44100)

        # Resample up then down
        up = audio.resample(buf, new_sample_rate=88200)
        down = audio.resample(up, new_sample_rate=44100)

        # Should be close to original duration
        assert abs(down.duration - 1.0) < 0.01
        assert down.sample_rate == 44100

    def test_reverse_with_fades(self):
        """Test combining reverse with fades."""
        # Use constant signal to avoid phase issues with sine
        buf = AudioBuffer(data=np.ones(44100), sample_rate=44100)

        # Fade in, reverse (becomes fade out)
        faded = audio.fade_in(buf, duration=0.2)
        reversed_faded = audio.reverse(faded)

        # Should now fade out
        assert reversed_faded.data[0] > 0.5
        assert reversed_faded.data[-1] < 0.1
