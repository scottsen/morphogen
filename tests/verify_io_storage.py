"""Verification script for I/O & Storage domain (no pytest required)"""

import numpy as np
import sys
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, '/home/user/morphogen')

from morphogen.stdlib.io_storage import (
    load_image, save_image,
    load_audio, save_audio,
    load_json, save_json,
    load_hdf5, save_hdf5,
    save_checkpoint, load_checkpoint
)


def test_image_io():
    """Test image I/O"""
    print("Testing Image I/O...")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create synthetic image
        img = np.random.rand(128, 128, 3).astype(np.float32)

        # Save and load
        path = tmpdir / "test_image.png"
        save_image(path, img)
        assert path.exists(), "Image file was not created"

        img_loaded = load_image(path)
        assert img_loaded.shape == img.shape, "Image shape mismatch"

        # Check values are close (PNG is lossy for normalized floats)
        error = np.mean(np.abs(img_loaded - img))
        assert error < 0.01, f"Image roundtrip error too large: {error}"

        print(f"  ✓ Image I/O test passed (error: {error:.6f})")


def test_grayscale_image():
    """Test grayscale image I/O"""
    print("Testing Grayscale Image I/O...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create grayscale image
        img = np.random.rand(128, 128).astype(np.float32)

        # Save and load
        path = tmpdir / "test_gray.png"
        save_image(path, img)
        img_loaded = load_image(path, grayscale=True)

        assert img_loaded.shape == img.shape, "Grayscale shape mismatch"

        error = np.mean(np.abs(img_loaded - img))
        assert error < 0.01, f"Grayscale roundtrip error too large: {error}"

        print(f"  ✓ Grayscale image test passed (error: {error:.6f})")


def test_audio_io():
    """Test audio I/O"""
    print("Testing Audio I/O...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create synthetic audio (440 Hz sine wave)
        sr = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Save and load
        path = tmpdir / "test_audio.wav"
        save_audio(path, audio, sr)
        assert path.exists(), "Audio file was not created"

        audio_loaded, sr_loaded = load_audio(path)
        assert sr_loaded == sr, "Sample rate mismatch"
        assert audio_loaded.shape == audio.shape, "Audio shape mismatch"

        error = np.mean(np.abs(audio_loaded - audio))
        assert error < 1e-4, f"Audio roundtrip error too large: {error}"

        print(f"  ✓ Audio I/O test passed (error: {error:.6f})")


def test_stereo_audio():
    """Test stereo audio I/O"""
    print("Testing Stereo Audio I/O...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create stereo audio
        sr = 44100
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        left = np.sin(2 * np.pi * 440 * t)
        right = np.sin(2 * np.pi * 554.37 * t)  # C#5
        audio = np.column_stack([left, right]).astype(np.float32)

        # Save and load
        path = tmpdir / "test_stereo.wav"
        save_audio(path, audio, sr)

        audio_loaded, sr_loaded = load_audio(path)
        assert audio_loaded.shape == audio.shape, "Stereo shape mismatch"

        error = np.mean(np.abs(audio_loaded - audio))
        assert error < 1e-4, f"Stereo roundtrip error too large: {error}"

        print(f"  ✓ Stereo audio test passed (error: {error:.6f})")


def test_json_io():
    """Test JSON I/O"""
    print("Testing JSON I/O...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test data
        data = {
            "learning_rate": 0.01,
            "epochs": 100,
            "layers": [64, 128, 256],
            "activation": "relu",
            "nested": {
                "param1": 1.5,
                "param2": "value"
            }
        }

        # Save and load
        path = tmpdir / "test.json"
        save_json(path, data)
        assert path.exists(), "JSON file was not created"

        data_loaded = load_json(path)
        assert data_loaded == data, "JSON data mismatch"

        print("  ✓ JSON I/O test passed")


def test_json_numpy_types():
    """Test JSON I/O with NumPy types"""
    print("Testing JSON with NumPy types...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create data with NumPy types
        data = {
            "int32": np.int32(42),
            "int64": np.int64(12345),
            "float32": np.float32(3.14),
            "float64": np.float64(2.71828),
            "bool": np.bool_(True),
            "array": np.array([1, 2, 3])
        }

        # Save and load
        path = tmpdir / "test_numpy.json"
        save_json(path, data)

        data_loaded = load_json(path)

        # Check values (types will be converted to Python natives)
        assert data_loaded["int32"] == 42
        assert data_loaded["int64"] == 12345
        assert abs(data_loaded["float32"] - 3.14) < 1e-6
        assert abs(data_loaded["float64"] - 2.71828) < 1e-6
        assert data_loaded["bool"] == True
        assert data_loaded["array"] == [1, 2, 3]

        print("  ✓ JSON NumPy types test passed")


def test_hdf5_io():
    """Test HDF5 I/O"""
    print("Testing HDF5 I/O...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create test arrays
        field1 = np.random.rand(64, 64).astype(np.float32)
        field2 = np.random.rand(128, 128, 3).astype(np.float32)

        # Save multiple datasets
        path = tmpdir / "test.h5"
        save_hdf5(path, {
            "field1": field1,
            "field2": field2
        })
        assert path.exists(), "HDF5 file was not created"

        # Load all datasets
        data_loaded = load_hdf5(path)
        assert "field1" in data_loaded, "field1 not found"
        assert "field2" in data_loaded, "field2 not found"

        assert np.allclose(data_loaded["field1"], field1), "field1 mismatch"
        assert np.allclose(data_loaded["field2"], field2), "field2 mismatch"

        # Load specific dataset
        field1_only = load_hdf5(path, "field1")
        assert np.allclose(field1_only, field1), "field1 specific load mismatch"

        print("  ✓ HDF5 I/O test passed")


def test_checkpoint():
    """Test checkpoint save/load"""
    print("Testing Checkpoint/Resume...")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create simulation state
        state = {
            "velocity_field": np.random.rand(64, 64, 2).astype(np.float32),
            "pressure_field": np.random.rand(64, 64).astype(np.float32),
            "particles": np.random.rand(100, 2).astype(np.float32),
            "parameters": {
                "dt": 0.01,
                "viscosity": 0.1,
                "n_particles": 100
            }
        }

        metadata = {
            "iteration": 1000,
            "time": 10.0,
            "version": "0.8.0"
        }

        # Save checkpoint
        path = tmpdir / "checkpoint_1000.h5"
        save_checkpoint(path, state, metadata)
        assert path.exists(), "Checkpoint file was not created"

        # Load checkpoint
        state_loaded, metadata_loaded = load_checkpoint(path)

        # Verify state
        assert np.allclose(state_loaded["velocity_field"], state["velocity_field"])
        assert np.allclose(state_loaded["pressure_field"], state["pressure_field"])
        assert np.allclose(state_loaded["particles"], state["particles"])

        # Verify parameters
        assert state_loaded["parameters"]["dt"] == state["parameters"]["dt"]
        assert state_loaded["parameters"]["viscosity"] == state["parameters"]["viscosity"]

        # Verify metadata
        assert metadata_loaded["iteration"] == metadata["iteration"]
        assert metadata_loaded["time"] == metadata["time"]
        assert metadata_loaded["version"] == metadata["version"]

        print("  ✓ Checkpoint/Resume test passed")


def test_error_handling():
    """Test error handling"""
    print("Testing Error Handling...")

    # Test loading non-existent file
    try:
        load_image("nonexistent.png")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass

    try:
        load_audio("nonexistent.wav")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass

    try:
        load_json("nonexistent.json")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass

    try:
        load_hdf5("nonexistent.h5")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass

    print("  ✓ Error handling test passed")


if __name__ == "__main__":
    print("=" * 60)
    print("I/O & STORAGE DOMAIN VERIFICATION")
    print("=" * 60)
    print()

    try:
        test_image_io()
        test_grayscale_image()
        test_audio_io()
        test_stereo_audio()
        test_json_io()
        test_json_numpy_types()
        test_hdf5_io()
        test_checkpoint()
        test_error_handling()

        print()
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)

    except AssertionError as e:
        print()
        print("=" * 60)
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)
