"""Unit tests for field operations."""

import pytest
import numpy as np
from morphogen.stdlib.field import field, Field2D


class TestFieldAllocation:
    """Tests for field.alloc operation."""

    def test_alloc_basic(self):
        """Test basic field allocation."""
        f = field.alloc((64, 64), fill_value=0.0)
        assert f.shape == (64, 64)
        assert np.all(f.data == 0.0)

    def test_alloc_with_value(self):
        """Test allocation with non-zero fill value."""
        f = field.alloc((32, 32), fill_value=5.0)
        assert np.all(f.data == 5.0)

    def test_alloc_dtype(self):
        """Test allocation with specific dtype."""
        f = field.alloc((10, 10), dtype=np.float64)
        assert f.data.dtype == np.float64


class TestFieldRandom:
    """Tests for field.random operation."""

    def test_random_deterministic(self):
        """Test that same seed produces same field."""
        f1 = field.random((50, 50), seed=42)
        f2 = field.random((50, 50), seed=42)
        assert np.allclose(f1.data, f2.data)

    def test_random_different_seeds(self):
        """Test that different seeds produce different fields."""
        f1 = field.random((50, 50), seed=1)
        f2 = field.random((50, 50), seed=2)
        assert not np.allclose(f1.data, f2.data)

    def test_random_range(self):
        """Test that random values are in specified range."""
        f = field.random((100, 100), seed=0, low=0.0, high=1.0)
        assert np.all(f.data >= 0.0)
        assert np.all(f.data <= 1.0)


class TestFieldDiffusion:
    """Tests for field.diffuse operation."""

    def test_diffuse_smooths_field(self):
        """Test that diffusion smooths a field."""
        # Create field with sharp peak
        f = field.alloc((32, 32), fill_value=0.0)
        f.data[16, 16] = 1.0

        # Diffuse
        f_smooth = field.diffuse(f, rate=0.5, dt=0.1, iterations=20)

        # Check that peak is lower and neighbors are higher
        assert f_smooth.data[16, 16] < 1.0
        assert f_smooth.data[15, 16] > 0.0
        assert f_smooth.data[17, 16] > 0.0

    def test_diffuse_preserves_shape(self):
        """Test that diffusion preserves field shape."""
        f = field.random((64, 64), seed=0)
        f_diffused = field.diffuse(f, rate=0.1, dt=0.01, iterations=10)
        assert f_diffused.shape == f.shape

    def test_diffuse_zero_rate(self):
        """Test that zero diffusion rate doesn't change field much."""
        f = field.random((32, 32), seed=0)
        f_diffused = field.diffuse(f, rate=0.0, dt=0.01, iterations=10)
        # With zero rate, field should be nearly unchanged
        assert np.allclose(f.data, f_diffused.data, rtol=0.1)


class TestFieldBoundary:
    """Tests for field.boundary operation."""

    def test_boundary_reflect(self):
        """Test reflect boundary conditions."""
        f = field.alloc((10, 10), fill_value=1.0)
        f.data[5, 5] = 5.0

        f_bound = field.boundary(f, spec="reflect")

        # Edges should be mirrored from interior
        assert f_bound.data[0, 5] == f_bound.data[1, 5]
        assert f_bound.data[-1, 5] == f_bound.data[-2, 5]

    def test_boundary_periodic(self):
        """Test periodic boundary conditions."""
        f = field.alloc((10, 10), fill_value=1.0)
        f.data[1, 5] = 2.0
        f.data[-2, 5] = 3.0

        f_bound = field.boundary(f, spec="periodic")

        # Edges should wrap around
        assert f_bound.data[0, 5] == f.data[-2, 5]
        assert f_bound.data[-1, 5] == f.data[1, 5]


class TestFieldCombine:
    """Tests for field.combine operation."""

    def test_combine_add(self):
        """Test field addition."""
        f1 = field.alloc((10, 10), fill_value=2.0)
        f2 = field.alloc((10, 10), fill_value=3.0)

        result = field.combine(f1, f2, operation="add")
        assert np.allclose(result.data, 5.0)

    def test_combine_mul(self):
        """Test field multiplication."""
        f1 = field.alloc((10, 10), fill_value=2.0)
        f2 = field.alloc((10, 10), fill_value=3.0)

        result = field.combine(f1, f2, operation="mul")
        assert np.allclose(result.data, 6.0)

    def test_combine_shape_mismatch(self):
        """Test that combining different sizes raises error."""
        f1 = field.alloc((10, 10))
        f2 = field.alloc((20, 20))

        with pytest.raises(ValueError, match="shape"):
            field.combine(f1, f2, operation="add")


class TestFieldMap:
    """Tests for field.map operation."""

    def test_map_abs(self):
        """Test mapping absolute value."""
        f = field.alloc((10, 10), fill_value=-5.0)
        result = field.map(f, func="abs")
        assert np.allclose(result.data, 5.0)

    def test_map_square(self):
        """Test mapping square function."""
        f = field.alloc((10, 10), fill_value=3.0)
        result = field.map(f, func="square")
        assert np.allclose(result.data, 9.0)

    def test_map_custom_function(self):
        """Test mapping custom callable."""
        f = field.alloc((10, 10), fill_value=2.0)
        result = field.map(f, func=lambda x: x * 2 + 1)
        assert np.allclose(result.data, 5.0)


class TestFieldAdvection:
    """Tests for field.advect operation."""

    def test_advect_preserves_shape(self):
        """Test that advection preserves field shape."""
        scalar = field.random((32, 32), seed=0)

        # Create velocity field
        vx = field.alloc((32, 32), fill_value=1.0)
        vy = field.alloc((32, 32), fill_value=0.0)
        velocity = Field2D(np.stack([vx.data, vy.data], axis=-1))

        advected = field.advect(scalar, velocity, dt=0.01)
        assert advected.shape == scalar.shape

    def test_advect_with_zero_velocity(self):
        """Test that zero velocity doesn't change field much."""
        scalar = field.random((32, 32), seed=0)

        # Zero velocity
        velocity = Field2D(np.zeros((32, 32, 2), dtype=np.float32))

        advected = field.advect(scalar, velocity, dt=0.01)
        assert np.allclose(scalar.data, advected.data, atol=0.01)


class TestFieldProjection:
    """Tests for field.project operation."""

    def test_project_reduces_divergence(self):
        """Test that projection reduces velocity divergence."""
        # Create divergent velocity field
        vx = field.random((32, 32), seed=1, low=-1, high=1)
        vy = field.random((32, 32), seed=2, low=-1, high=1)
        velocity = Field2D(np.stack([vx.data, vy.data], axis=-1))

        # Compute initial divergence
        div_before = np.zeros((32, 32))
        div_before[1:-1, 1:-1] = (
            (velocity.data[1:-1, 2:, 0] - velocity.data[1:-1, :-2, 0]) / 2 +
            (velocity.data[2:, 1:-1, 1] - velocity.data[:-2, 1:-1, 1]) / 2
        )
        div_rms_before = np.sqrt(np.mean(div_before**2))

        # Project
        projected = field.project(velocity, iterations=30)

        # Compute final divergence
        div_after = np.zeros((32, 32))
        div_after[1:-1, 1:-1] = (
            (projected.data[1:-1, 2:, 0] - projected.data[1:-1, :-2, 0]) / 2 +
            (projected.data[2:, 1:-1, 1] - projected.data[:-2, 1:-1, 1]) / 2
        )
        div_rms_after = np.sqrt(np.mean(div_after**2))

        # Divergence should be reduced
        assert div_rms_after < div_rms_before

    def test_project_preserves_shape(self):
        """Test that projection preserves velocity field shape."""
        vx = field.alloc((32, 32), fill_value=1.0)
        vy = field.alloc((32, 32), fill_value=0.0)
        velocity = Field2D(np.stack([vx.data, vy.data], axis=-1))

        projected = field.project(velocity, iterations=20)
        assert projected.shape == velocity.shape
        assert projected.data.shape == velocity.data.shape


class TestFieldDeterminism:
    """Tests for deterministic behavior."""

    def test_random_deterministic(self):
        """Test that random fields are deterministic with seed."""
        f1 = field.random((100, 100), seed=12345)
        f2 = field.random((100, 100), seed=12345)
        assert np.array_equal(f1.data, f2.data)

    def test_diffuse_deterministic(self):
        """Test that diffusion is deterministic."""
        f = field.random((64, 64), seed=42)

        f1 = field.diffuse(f, rate=0.3, dt=0.1, iterations=20)
        f2 = field.diffuse(f, rate=0.3, dt=0.1, iterations=20)

        assert np.allclose(f1.data, f2.data)

    def test_advect_deterministic(self):
        """Test that advection is deterministic."""
        scalar = field.random((32, 32), seed=1)
        vx = field.alloc((32, 32), fill_value=0.5)
        vy = field.alloc((32, 32), fill_value=0.5)
        velocity = Field2D(np.stack([vx.data, vy.data], axis=-1))

        adv1 = field.advect(scalar, velocity, dt=0.01)
        adv2 = field.advect(scalar, velocity, dt=0.01)

        assert np.allclose(adv1.data, adv2.data)


class TestFieldEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_size_field(self):
        """Test that zero-size field is handled."""
        # Should work or raise clear error
        try:
            f = field.alloc((0, 0))
            assert f.shape == (0, 0)
        except ValueError:
            pass  # Acceptable to reject

    def test_large_diffusion_rate(self):
        """Test stability with large diffusion rate."""
        f = field.random((32, 32), seed=0)
        # Large rate might be unstable but shouldn't crash
        f_diff = field.diffuse(f, rate=10.0, dt=0.1, iterations=5)
        assert f_diff.shape == f.shape
        assert not np.any(np.isnan(f_diff.data))

    def test_negative_values(self):
        """Test operations with negative field values."""
        f = field.random((32, 32), seed=0, low=-1.0, high=1.0)
        assert np.any(f.data < 0)

        # Should handle negative values fine
        f_diff = field.diffuse(f, rate=0.1, dt=0.01, iterations=10)
        assert f_diff.shape == f.shape
