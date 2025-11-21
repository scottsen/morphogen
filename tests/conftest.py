"""Shared pytest fixtures for all tests."""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from morphogen.stdlib.field import field, Field2D
from morphogen.runtime.runtime import ExecutionContext, Runtime


@pytest.fixture
def temp_dir():
    """Provide temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def execution_context():
    """Provide fresh execution context with deterministic seed."""
    return ExecutionContext(global_seed=42)


@pytest.fixture
def runtime(execution_context):
    """Provide runtime with fresh context."""
    return Runtime(execution_context)


@pytest.fixture
def simple_field():
    """Provide simple 32x32 random field."""
    return field.random((32, 32), seed=42)


@pytest.fixture
def small_field():
    """Provide small 16x16 field for quick tests."""
    return field.random((16, 16), seed=1)


@pytest.fixture
def large_field():
    """Provide large 128x128 field for performance tests."""
    return field.random((128, 128), seed=42)


@pytest.fixture
def constant_field():
    """Provide constant field with known value."""
    return field.alloc((32, 32), fill_value=1.0)


@pytest.fixture
def velocity_field():
    """Provide 2D velocity field for testing."""
    vx = field.alloc((32, 32), fill_value=0.5)
    vy = field.alloc((32, 32), fill_value=0.5)
    return Field2D(np.stack([vx.data, vy.data], axis=-1))


@pytest.fixture
def divergent_velocity():
    """Provide divergent velocity field for projection tests."""
    vx = field.random((32, 32), seed=1, low=-1, high=1)
    vy = field.random((32, 32), seed=2, low=-1, high=1)
    return Field2D(np.stack([vx.data, vy.data], axis=-1))


@pytest.fixture
def peak_field():
    """Provide field with single peak in center."""
    f = field.alloc((32, 32), fill_value=0.0)
    f.data[16, 16] = 1.0
    return f


@pytest.fixture(params=["grayscale", "fire", "viridis", "coolwarm"])
def palette_name(request):
    """Parametrized fixture for all available palettes."""
    return request.param


@pytest.fixture(params=[16, 32, 64])
def field_size(request):
    """Parametrized fixture for different field sizes."""
    return request.param


@pytest.fixture
def sample_dsl_program():
    """Provide sample DSL program for testing."""
    return """
    # Simple heat diffusion
    temp = field.random((64, 64), seed=42, low=0.0, high=1.0)
    temp = field.diffuse(temp, rate=0.5, dt=0.1, iterations=20)
    temp = field.boundary(temp, spec="reflect")
    vis = visual.colorize(temp, palette="fire")
    """


@pytest.fixture
def complex_dsl_program():
    """Provide complex DSL program with multiple operations."""
    return """
    # Reaction-diffusion simulation
    temp = field.random((128, 128), seed=42)
    temp = field.diffuse(temp, rate=0.2, dt=0.1, iterations=30)

    # Create velocity field
    vx = field.random((128, 128), seed=1, low=-1.0, high=1.0)
    vy = field.random((128, 128), seed=2, low=-1.0, high=1.0)

    # Combine fields
    result = field.combine(temp, vx, operation="mul")

    # Visualize
    vis = visual.colorize(result, palette="viridis")
    """


# Mark configuration
def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "determinism: marks tests that verify deterministic behavior"
    )


# Helper functions available to all tests
@pytest.fixture
def assert_fields_equal():
    """Helper to assert two fields are equal."""
    def _assert_equal(f1, f2, rtol=1e-5, atol=1e-8):
        assert f1.shape == f2.shape, f"Shapes differ: {f1.shape} vs {f2.shape}"
        assert np.allclose(f1.data, f2.data, rtol=rtol, atol=atol), \
            f"Field values differ (max diff: {np.max(np.abs(f1.data - f2.data))})"
    return _assert_equal


@pytest.fixture
def assert_deterministic():
    """Helper to assert operation is deterministic."""
    def _assert_deterministic(operation, *args, **kwargs):
        """Run operation twice and verify results are identical."""
        result1 = operation(*args, **kwargs)
        result2 = operation(*args, **kwargs)

        if hasattr(result1, 'data'):
            assert np.array_equal(result1.data, result2.data), \
                "Operation is not deterministic"
        else:
            assert result1 == result2, "Operation is not deterministic"

    return _assert_deterministic


@pytest.fixture
def measure_performance():
    """Helper to measure operation performance."""
    import time

    def _measure(operation, *args, **kwargs):
        """Measure operation execution time."""
        start = time.perf_counter()
        result = operation(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed

    return _measure
