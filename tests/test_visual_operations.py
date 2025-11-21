"""Unit tests for visual operations."""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from morphogen.stdlib.field import field, Field2D
from morphogen.stdlib.visual import visual, Visual


class TestVisualColorize:
    """Tests for visual.colorize operation."""

    def test_colorize_basic(self):
        """Test basic colorization."""
        f = field.random((64, 64), seed=0)
        vis = visual.colorize(f, palette="grayscale")
        assert isinstance(vis, Visual)
        assert vis.shape == (64, 64)

    def test_colorize_all_palettes(self):
        """Test all available palettes."""
        f = field.random((32, 32), seed=0)

        for palette in ["grayscale", "fire", "viridis", "coolwarm"]:
            vis = visual.colorize(f, palette=palette)
            assert isinstance(vis, Visual)
            assert vis.shape == f.shape

    def test_colorize_with_vmin_vmax(self):
        """Test colorization with custom value range."""
        f = field.random((32, 32), seed=0)
        vis = visual.colorize(f, palette="fire", vmin=0.2, vmax=0.8)
        assert isinstance(vis, Visual)

    def test_colorize_invalid_palette(self):
        """Test that invalid palette raises error."""
        f = field.random((32, 32), seed=0)
        with pytest.raises(ValueError, match="Unknown palette"):
            visual.colorize(f, palette="nonexistent")

    def test_colorize_constant_field(self):
        """Test colorization of field with constant values."""
        f = field.alloc((32, 32), fill_value=0.5)
        vis = visual.colorize(f, palette="fire")
        # Should not crash, even though max == min
        assert isinstance(vis, Visual)

    def test_colorize_vector_field(self):
        """Test colorization uses magnitude for vector fields."""
        vx = field.random((32, 32), seed=1, low=-1, high=1)
        vy = field.random((32, 32), seed=2, low=-1, high=1)
        velocity = Field2D(np.stack([vx.data, vy.data], axis=-1))

        vis = visual.colorize(velocity, palette="viridis")
        assert isinstance(vis, Visual)
        assert vis.shape == (32, 32)


class TestVisualOutput:
    """Tests for visual.output operation."""

    def test_output_png(self):
        """Test PNG output."""
        f = field.random((64, 64), seed=0)
        vis = visual.colorize(f, palette="fire")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            visual.output(vis, path=tmp_path)
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 0
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_output_auto_format(self):
        """Test automatic format detection from extension."""
        f = field.random((32, 32), seed=0)
        vis = visual.colorize(f, palette="fire")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            visual.output(vis, path=tmp_path, format="auto")
            assert os.path.exists(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_output_jpg(self):
        """Test JPEG output."""
        f = field.random((64, 64), seed=0)
        vis = visual.colorize(f, palette="fire")

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            visual.output(vis, path=tmp_path, format="jpg")
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 0
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_output_preserves_dimensions(self):
        """Test that output image has correct dimensions."""
        f = field.random((128, 64), seed=0)  # Non-square
        vis = visual.colorize(f, palette="fire")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            visual.output(vis, path=tmp_path)

            # Verify dimensions with PIL
            from PIL import Image
            img = Image.open(tmp_path)
            assert img.size == (64, 128)  # PIL uses (width, height)
            img.close()  # Close file handle before deletion (Windows compatibility)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestVisualDataRange:
    """Tests for handling different data ranges."""

    def test_colorize_negative_values(self):
        """Test colorization with negative values."""
        f = field.random((32, 32), seed=0, low=-1.0, high=1.0)
        vis = visual.colorize(f, palette="coolwarm")
        assert isinstance(vis, Visual)

    def test_colorize_large_values(self):
        """Test colorization with large values."""
        f = field.random((32, 32), seed=0, low=1000, high=2000)
        vis = visual.colorize(f, palette="fire")
        assert isinstance(vis, Visual)

    def test_colorize_small_range(self):
        """Test colorization with very small value range."""
        f = field.random((32, 32), seed=0, low=0.5, high=0.5001)
        vis = visual.colorize(f, palette="viridis")
        assert isinstance(vis, Visual)


class TestVisualRGBCorrectness:
    """Tests for RGB value correctness."""

    def test_rgb_values_in_range(self):
        """Test that RGB values are in [0, 1] range."""
        f = field.random((32, 32), seed=0)
        vis = visual.colorize(f, palette="fire")

        assert np.all(vis.data >= 0.0)
        assert np.all(vis.data <= 1.0)

    def test_rgb_channels(self):
        """Test that visual has 3 RGB channels."""
        f = field.random((32, 32), seed=0)
        vis = visual.colorize(f, palette="fire")

        assert vis.data.shape == (32, 32, 3)

    def test_grayscale_palette(self):
        """Test that grayscale palette produces gray values."""
        f = field.alloc((10, 10), fill_value=0.5)
        vis = visual.colorize(f, palette="grayscale")

        # Grayscale should have R=G=B
        assert np.allclose(vis.data[:, :, 0], vis.data[:, :, 1])
        assert np.allclose(vis.data[:, :, 1], vis.data[:, :, 2])


class TestVisualEdgeCases:
    """Tests for edge cases."""

    def test_colorize_single_pixel(self):
        """Test colorization of 1x1 field."""
        f = field.alloc((1, 1), fill_value=0.5)
        vis = visual.colorize(f, palette="fire")
        assert vis.shape == (1, 1)

    def test_colorize_narrow_field(self):
        """Test colorization of very narrow field."""
        f = field.random((1, 100), seed=0)
        vis = visual.colorize(f, palette="viridis")
        assert vis.shape == (1, 100)

    def test_output_without_extension(self):
        """Test output with no file extension."""
        f = field.random((32, 32), seed=0)
        vis = visual.colorize(f, palette="fire")

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            visual.output(vis, path=tmp_path)
            # Should default to PNG
            assert os.path.exists(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestVisualDeterminism:
    """Tests for deterministic visualization."""

    def test_colorize_deterministic(self):
        """Test that colorization is deterministic."""
        f = field.random((64, 64), seed=42)

        vis1 = visual.colorize(f, palette="fire")
        vis2 = visual.colorize(f, palette="fire")

        assert np.array_equal(vis1.data, vis2.data)

    def test_output_deterministic(self):
        """Test that output files are identical."""
        f = field.random((64, 64), seed=42)
        vis = visual.colorize(f, palette="fire")

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp1:
            path1 = tmp1.name
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp2:
            path2 = tmp2.name

        try:
            visual.output(vis, path=path1)
            visual.output(vis, path=path2)

            # Files should be identical
            with open(path1, 'rb') as f1, open(path2, 'rb') as f2:
                assert f1.read() == f2.read()
        finally:
            for path in [path1, path2]:
                if os.path.exists(path):
                    os.unlink(path)


class TestVisualIntegration:
    """Integration tests for complete field-to-image pipeline."""

    def test_full_pipeline(self):
        """Test complete pipeline from field creation to output."""
        # Create field
        temp = field.random((128, 128), seed=42, low=0.0, high=1.0)

        # Process field
        temp = field.diffuse(temp, rate=0.5, dt=0.1, iterations=10)
        temp = field.boundary(temp, spec="reflect")

        # Visualize
        vis = visual.colorize(temp, palette="fire")

        # Output
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            visual.output(vis, path=tmp_path)
            assert os.path.exists(tmp_path)
            assert os.path.getsize(tmp_path) > 1000  # Should be substantial
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def test_multiple_palettes_same_field(self):
        """Test creating multiple visualizations of same field."""
        f = field.random((64, 64), seed=0)

        outputs = []
        for palette in ["grayscale", "fire", "viridis"]:
            vis = visual.colorize(f, palette=palette)
            with tempfile.NamedTemporaryFile(suffix=f"_{palette}.png", delete=False) as tmp:
                tmp_path = tmp.name
                outputs.append(tmp_path)
                visual.output(vis, path=tmp_path)

        try:
            # All files should exist
            for path in outputs:
                assert os.path.exists(path)
                assert os.path.getsize(path) > 0
        finally:
            for path in outputs:
                if os.path.exists(path):
                    os.unlink(path)
