"""Comprehensive tests for I/O & Storage domain.

Tests cover:
- Image I/O (PNG, JPEG, BMP)
- Audio I/O (WAV, FLAC)
- JSON I/O
- HDF5 I/O
- Checkpoint/Resume functionality
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Import I/O functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from morphogen.stdlib import io_storage as io


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    tmp = tempfile.mkdtemp()
    yield Path(tmp)
    shutil.rmtree(tmp)


@pytest.fixture
def sample_image_gray():
    """Create sample grayscale image (checkerboard pattern)."""
    img = np.zeros((64, 64), dtype=np.float32)
    img[::2, ::2] = 1.0  # White squares
    img[1::2, 1::2] = 1.0
    return img


@pytest.fixture
def sample_image_rgb():
    """Create sample RGB image (gradient)."""
    img = np.zeros((64, 64, 3), dtype=np.float32)
    img[:, :, 0] = np.linspace(0, 1, 64)[:, None]  # Red gradient
    img[:, :, 1] = np.linspace(0, 1, 64)[None, :]  # Green gradient
    img[:, :, 2] = 0.5  # Constant blue
    return img


@pytest.fixture
def sample_audio_mono():
    """Create sample mono audio (440 Hz sine wave)."""
    sample_rate = 44100
    duration = 1.0  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz, amplitude 0.5
    return audio, sample_rate


@pytest.fixture
def sample_audio_stereo():
    """Create sample stereo audio (440 Hz and 880 Hz)."""
    sample_rate = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 880 * t)
    audio = np.column_stack([left, right])
    return audio, sample_rate


# ============================================================================
# IMAGE I/O TESTS
# ============================================================================

class TestImageIO:
    """Tests for image loading and saving."""

    def test_save_load_png_gray(self, temp_dir, sample_image_gray):
        """Test saving and loading grayscale PNG."""
        path = temp_dir / "test_gray.png"

        # Save
        io.save_image(path, sample_image_gray)
        assert path.exists()

        # Load
        loaded = io.load_image(path, grayscale=True)
        assert loaded.shape == sample_image_gray.shape
        assert loaded.dtype == np.float32

        # Check values (allow for JPEG compression artifacts)
        np.testing.assert_allclose(loaded, sample_image_gray, atol=0.01)

    def test_save_load_png_rgb(self, temp_dir, sample_image_rgb):
        """Test saving and loading RGB PNG."""
        path = temp_dir / "test_rgb.png"

        # Save
        io.save_image(path, sample_image_rgb)
        assert path.exists()

        # Load
        loaded = io.load_image(path)
        assert loaded.shape == sample_image_rgb.shape
        assert loaded.dtype == np.float32

        # Check values
        np.testing.assert_allclose(loaded, sample_image_rgb, atol=0.01)

    def test_save_load_jpeg(self, temp_dir, sample_image_rgb):
        """Test saving and loading JPEG (lossy compression)."""
        path = temp_dir / "test.jpg"

        # Save with high quality
        io.save_image(path, sample_image_rgb, quality=95)
        assert path.exists()

        # Load
        loaded = io.load_image(path)
        assert loaded.shape == sample_image_rgb.shape

        # JPEG is lossy, so use larger tolerance
        np.testing.assert_allclose(loaded, sample_image_rgb, atol=0.05)

    def test_save_uint8(self, temp_dir):
        """Test saving uint8 images (no denormalization)."""
        path = temp_dir / "test_uint8.png"
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

        io.save_image(path, img, denormalize=False)
        loaded = io.load_image(path, as_float=False)

        assert loaded.dtype == np.uint8
        np.testing.assert_array_equal(loaded, img)

    def test_load_nonexistent(self, temp_dir):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            io.load_image(temp_dir / "nonexistent.png")

    def test_save_invalid_shape(self, temp_dir):
        """Test saving array with invalid shape raises error."""
        with pytest.raises(ValueError):
            # 4D array not supported
            io.save_image(temp_dir / "invalid.png", np.zeros((10, 10, 10, 10)))

    def test_grayscale_conversion(self, temp_dir, sample_image_rgb):
        """Test automatic grayscale conversion."""
        path = temp_dir / "color.png"
        io.save_image(path, sample_image_rgb)

        # Load as grayscale
        gray = io.load_image(path, grayscale=True)
        assert gray.ndim == 2
        assert gray.shape == (64, 64)


# ============================================================================
# AUDIO I/O TESTS
# ============================================================================

class TestAudioIO:
    """Tests for audio loading and saving."""

    def test_save_load_wav_mono(self, temp_dir, sample_audio_mono):
        """Test saving and loading mono WAV."""
        audio, sr = sample_audio_mono
        path = temp_dir / "test_mono.wav"

        # Save
        io.save_audio(path, audio, sr)
        assert path.exists()

        # Load
        loaded, loaded_sr = io.load_audio(path)
        assert loaded_sr == sr
        assert loaded.shape == audio.shape

        # Check values
        np.testing.assert_allclose(loaded, audio, atol=1e-5)

    def test_save_load_wav_stereo(self, temp_dir, sample_audio_stereo):
        """Test saving and loading stereo WAV."""
        audio, sr = sample_audio_stereo
        path = temp_dir / "test_stereo.wav"

        # Save
        io.save_audio(path, audio, sr)
        assert path.exists()

        # Load
        loaded, loaded_sr = io.load_audio(path)
        assert loaded_sr == sr
        assert loaded.shape == audio.shape

        # Check values
        np.testing.assert_allclose(loaded, audio, atol=1e-5)

    def test_save_load_flac(self, temp_dir, sample_audio_mono):
        """Test saving and loading FLAC (lossless compression)."""
        audio, sr = sample_audio_mono
        path = temp_dir / "test.flac"

        # Save
        io.save_audio(path, audio, sr)
        assert path.exists()

        # Load
        loaded, loaded_sr = io.load_audio(path)
        assert loaded_sr == sr

        # FLAC is lossless
        np.testing.assert_allclose(loaded, audio, atol=1e-5)

    def test_mono_conversion(self, temp_dir, sample_audio_stereo):
        """Test automatic stereo-to-mono conversion."""
        audio, sr = sample_audio_stereo
        path = temp_dir / "stereo.wav"

        # Save stereo
        io.save_audio(path, audio, sr)

        # Load as mono
        loaded, loaded_sr = io.load_audio(path, mono=True)
        assert loaded.ndim == 1
        assert loaded_sr == sr

    def test_audio_clipping_warning(self, temp_dir):
        """Test that out-of-range audio triggers warning and clipping."""
        audio = np.array([0.5, 1.5, -1.5, 0.0], dtype=np.float32)  # Out of range
        path = temp_dir / "clipped.wav"

        with pytest.warns(UserWarning, match="out of range"):
            io.save_audio(path, audio, 44100)

        # Load and verify clipping
        loaded, _ = io.load_audio(path)
        assert np.max(loaded) <= 1.0
        assert np.min(loaded) >= -1.0

    def test_load_nonexistent_audio(self, temp_dir):
        """Test loading nonexistent audio file raises error."""
        with pytest.raises(FileNotFoundError):
            io.load_audio(temp_dir / "nonexistent.wav")


# ============================================================================
# JSON I/O TESTS
# ============================================================================

class TestJSONIO:
    """Tests for JSON loading and saving."""

    def test_save_load_json(self, temp_dir):
        """Test saving and loading JSON dict."""
        path = temp_dir / "test.json"
        data = {
            "name": "test",
            "value": 42,
            "pi": 3.14159,
            "enabled": True,
            "items": [1, 2, 3, 4, 5],
            "nested": {"a": 1, "b": 2}
        }

        # Save
        io.save_json(path, data)
        assert path.exists()

        # Load
        loaded = io.load_json(path)
        assert loaded == data

    def test_save_json_with_numpy(self, temp_dir):
        """Test saving JSON with NumPy types."""
        path = temp_dir / "numpy_types.json"
        data = {
            "int32": np.int32(42),
            "int64": np.int64(1000),
            "float32": np.float32(3.14),
            "float64": np.float64(2.71828),
            "bool": np.bool_(True),
            "array": np.array([1, 2, 3])
        }

        # Save (should convert NumPy types to Python types)
        io.save_json(path, data)

        # Load
        loaded = io.load_json(path)
        assert loaded["int32"] == 42
        assert loaded["float32"] == pytest.approx(3.14, rel=1e-5)
        assert loaded["array"] == [1, 2, 3]

    def test_save_json_sorted_keys(self, temp_dir):
        """Test saving JSON with sorted keys."""
        path = temp_dir / "sorted.json"
        data = {"z": 1, "a": 2, "m": 3}

        io.save_json(path, data, sort_keys=True)

        # Read raw file to check key order
        with open(path) as f:
            content = f.read()

        # Keys should be sorted alphabetically
        assert content.index('"a"') < content.index('"m"') < content.index('"z"')

    def test_load_nonexistent_json(self, temp_dir):
        """Test loading nonexistent JSON file raises error."""
        with pytest.raises(FileNotFoundError):
            io.load_json(temp_dir / "nonexistent.json")


# ============================================================================
# HDF5 I/O TESTS
# ============================================================================

class TestHDF5IO:
    """Tests for HDF5 loading and saving."""

    def test_save_load_single_array(self, temp_dir):
        """Test saving and loading single array."""
        path = temp_dir / "single.h5"
        data = np.random.rand(100, 100).astype(np.float32)

        # Save
        io.save_hdf5(path, data)
        assert path.exists()

        # Load
        loaded = io.load_hdf5(path, dataset='data')
        np.testing.assert_array_equal(loaded, data)

    def test_save_load_multiple_arrays(self, temp_dir):
        """Test saving and loading multiple arrays."""
        path = temp_dir / "multiple.h5"
        data = {
            "velocity": np.random.rand(64, 64, 2).astype(np.float32),
            "pressure": np.random.rand(64, 64).astype(np.float32),
            "temperature": np.random.rand(64, 64).astype(np.float32)
        }

        # Save
        io.save_hdf5(path, data)

        # Load all
        loaded = io.load_hdf5(path)
        assert set(loaded.keys()) == set(data.keys())
        for key in data:
            np.testing.assert_array_equal(loaded[key], data[key])

    def test_save_load_specific_dataset(self, temp_dir):
        """Test loading specific dataset from HDF5."""
        path = temp_dir / "multi.h5"
        data = {
            "field1": np.ones((10, 10)),
            "field2": np.zeros((10, 10))
        }

        io.save_hdf5(path, data)

        # Load only field1
        loaded = io.load_hdf5(path, dataset='field1')
        np.testing.assert_array_equal(loaded, data["field1"])

    def test_hdf5_compression(self, temp_dir):
        """Test HDF5 compression reduces file size."""
        data = np.zeros((1000, 1000), dtype=np.float32)  # Highly compressible

        # Save without compression
        path_uncompressed = temp_dir / "uncompressed.h5"
        io.save_hdf5(path_uncompressed, data, compression=None)

        # Save with compression
        path_compressed = temp_dir / "compressed.h5"
        io.save_hdf5(path_compressed, data, compression='gzip', compression_opts=9)

        # Compressed should be smaller
        size_uncompressed = path_uncompressed.stat().st_size
        size_compressed = path_compressed.stat().st_size
        assert size_compressed < size_uncompressed

    def test_load_nonexistent_dataset(self, temp_dir):
        """Test loading nonexistent dataset raises error."""
        path = temp_dir / "test.h5"
        io.save_hdf5(path, {"field1": np.ones((10, 10))})

        with pytest.raises(KeyError):
            io.load_hdf5(path, dataset='nonexistent')


# ============================================================================
# CHECKPOINT/RESUME TESTS
# ============================================================================

class TestCheckpointing:
    """Tests for checkpoint saving and loading."""

    def test_save_load_checkpoint(self, temp_dir):
        """Test saving and loading checkpoint."""
        path = temp_dir / "checkpoint.h5"

        # Create state
        state = {
            "velocity_field": np.random.rand(64, 64, 2).astype(np.float32),
            "pressure_field": np.random.rand(64, 64).astype(np.float32),
            "particles": np.random.rand(100, 3).astype(np.float32),
            "parameters": {
                "dt": 0.01,
                "viscosity": 0.1,
                "gravity": -9.81
            }
        }

        metadata = {
            "iteration": 1000,
            "time": 10.0,
            "version": "1.0"
        }

        # Save
        io.save_checkpoint(path, state, metadata)
        assert path.exists()

        # Load
        loaded_state, loaded_metadata = io.load_checkpoint(path)

        # Verify arrays
        np.testing.assert_array_equal(loaded_state["velocity_field"], state["velocity_field"])
        np.testing.assert_array_equal(loaded_state["pressure_field"], state["pressure_field"])
        np.testing.assert_array_equal(loaded_state["particles"], state["particles"])

        # Verify parameters
        assert loaded_state["parameters"]["dt"] == state["parameters"]["dt"]
        assert loaded_state["parameters"]["viscosity"] == state["parameters"]["viscosity"]

        # Verify metadata
        assert loaded_metadata["iteration"] == metadata["iteration"]
        assert loaded_metadata["time"] == metadata["time"]

    def test_checkpoint_no_metadata(self, temp_dir):
        """Test checkpoint without metadata."""
        path = temp_dir / "checkpoint_no_meta.h5"
        state = {"field": np.ones((10, 10))}

        # Save without metadata
        io.save_checkpoint(path, state)

        # Load
        loaded_state, loaded_metadata = io.load_checkpoint(path)
        np.testing.assert_array_equal(loaded_state["field"], state["field"])
        assert loaded_metadata == {}

    def test_checkpoint_determinism(self, temp_dir):
        """Test checkpoint save/load is deterministic."""
        path1 = temp_dir / "checkpoint1.h5"
        path2 = temp_dir / "checkpoint2.h5"

        state = {
            "field": np.random.RandomState(42).rand(100, 100).astype(np.float32)
        }

        # Save twice
        io.save_checkpoint(path1, state)
        io.save_checkpoint(path2, state)

        # Load both
        state1, _ = io.load_checkpoint(path1)
        state2, _ = io.load_checkpoint(path2)

        # Should be identical
        np.testing.assert_array_equal(state1["field"], state2["field"])


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple I/O operations."""

    def test_simulation_workflow(self, temp_dir):
        """Test complete simulation workflow with checkpointing."""
        # Initial state
        state = {
            "velocity": np.random.rand(32, 32, 2).astype(np.float32),
            "density": np.random.rand(32, 32).astype(np.float32)
        }

        # Save initial checkpoint
        checkpoint_0 = temp_dir / "sim_0000.h5"
        io.save_checkpoint(checkpoint_0, state, {"iteration": 0, "time": 0.0})

        # Simulate some steps
        for i in range(1, 4):
            state["velocity"] *= 0.99  # Damping
            state["density"] *= 0.98

            checkpoint = temp_dir / f"sim_{i:04d}.h5"
            io.save_checkpoint(checkpoint, state, {"iteration": i * 100, "time": i * 1.0})

        # Load checkpoint and verify
        loaded_state, loaded_meta = io.load_checkpoint(temp_dir / "sim_0003.h5")
        assert loaded_meta["iteration"] == 300
        assert loaded_state["velocity"].shape == (32, 32, 2)

    def test_field_visualization_pipeline(self, temp_dir):
        """Test field → visualization → save image pipeline."""
        # Generate field
        field = np.random.rand(128, 128).astype(np.float32)

        # Normalize to [0, 1]
        field = (field - field.min()) / (field.max() - field.min())

        # Convert to RGB (grayscale)
        rgb = np.stack([field, field, field], axis=-1)

        # Save as image
        img_path = temp_dir / "field_visualization.png"
        io.save_image(img_path, rgb)

        # Save raw data as HDF5
        h5_path = temp_dir / "field_data.h5"
        io.save_hdf5(h5_path, {"field": field})

        # Verify both exist
        assert img_path.exists()
        assert h5_path.exists()

        # Load and verify
        loaded_field = io.load_hdf5(h5_path, "field")
        np.testing.assert_array_equal(loaded_field, field)


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
