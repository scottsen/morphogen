# Testing Quick Start Guide

Get your testing infrastructure up and running in 5 minutes.

---

## Step 1: Install Test Dependencies

```bash
# Install dev dependencies including pytest
pip install -e ".[dev]"

# Or install individually
pip install pytest pytest-cov pytest-xdist hypothesis
```

---

## Step 2: Run Your First Tests

```bash
# Run all tests
python -m pytest

# Run with verbose output
python -m pytest -v

# Run with coverage
python -m pytest --cov=creative_computation --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

---

## Step 3: Verify Test Status

You should see output like:

```
============================== test session starts ===============================
collected 66 items

tests/test_field_operations.py ........................... [ 40%]
tests/test_visual_operations.py ......................... [ 78%]
tests/test_lexer.py ............ [ 96%]
tests/test_parser.py ...... [100%]

============================== 66 passed in 2.34s ================================
```

---

## Step 4: Add Your First New Test

**File:** `tests/test_my_feature.py`

```python
"""Tests for my new feature."""

import pytest
from creative_computation.stdlib.field import field


def test_my_feature():
    """Test that my feature works."""
    f = field.random((32, 32), seed=42)
    assert f.shape == (32, 32)
```

Run it:
```bash
python -m pytest tests/test_my_feature.py -v
```

---

## Common Test Commands

```bash
# Run specific test file
python -m pytest tests/test_field_operations.py

# Run specific test class
python -m pytest tests/test_field_operations.py::TestFieldDiffusion

# Run specific test
python -m pytest tests/test_field_operations.py::TestFieldDiffusion::test_diffuse_smooths_field

# Run tests matching pattern
python -m pytest -k "diffusion"

# Run tests in parallel (4 workers)
python -m pytest -n 4

# Stop on first failure
python -m pytest -x

# Show print statements
python -m pytest -s

# Show slowest 10 tests
python -m pytest --durations=10

# Run only fast tests (skip slow ones)
python -m pytest -m "not slow"
```

---

## Understanding Test Output

### ✅ Success
```
tests/test_field_operations.py::TestFieldDiffusion::test_diffuse_smooths_field PASSED
```

### ❌ Failure
```
tests/test_field_operations.py::TestFieldDiffusion::test_diffuse_smooths_field FAILED

================================== FAILURES ======================================
_____________________ TestFieldDiffusion.test_diffuse_smooths_field ______________

    def test_diffuse_smooths_field(self):
        f = field.alloc((32, 32), fill_value=0.0)
        f.data[16, 16] = 1.0
        f_smooth = field.diffuse(f, rate=0.5, dt=0.1, iterations=20)
>       assert f_smooth.data[16, 16] < 1.0
E       assert 1.0 < 1.0

tests/test_field_operations.py:63: AssertionError
```

### ⏩ Skipped
```
tests/test_integration.py::test_example_execution SKIPPED (need pytest-asyncio)
```

---

## Coverage Report

After running with `--cov`, check coverage:

```bash
# Terminal summary
python -m pytest --cov=creative_computation --cov-report=term-missing

# HTML report (more detailed)
python -m pytest --cov=creative_computation --cov-report=html
open htmlcov/index.html
```

Coverage report shows:
- Which files are tested
- Which lines are covered
- Which lines are NOT covered (need tests!)

**Goal:** Aim for 80%+ coverage

---

## Writing Good Tests

### Test Structure (AAA Pattern)

```python
def test_my_operation():
    # Arrange - Set up test data
    f = field.random((32, 32), seed=42)

    # Act - Perform operation
    result = field.diffuse(f, rate=0.5, dt=0.1, iterations=20)

    # Assert - Verify results
    assert result.shape == (32, 32)
    assert np.all(result.data >= 0.0)
```

### Use Fixtures

```python
def test_with_fixture(simple_field):
    """Use fixture for common test data."""
    # simple_field is provided by conftest.py
    result = field.diffuse(simple_field, rate=0.1, dt=0.01, iterations=5)
    assert result.shape == simple_field.shape
```

### Test Edge Cases

```python
def test_edge_cases():
    """Test boundary conditions."""
    # Empty field
    f_empty = field.alloc((0, 0))

    # Single pixel
    f_single = field.alloc((1, 1), fill_value=1.0)

    # Very large
    f_large = field.alloc((1024, 1024))

    # Negative values
    f_neg = field.random((32, 32), seed=1, low=-10, high=10)
```

### Test Errors

```python
def test_invalid_input():
    """Test that invalid inputs are caught."""
    f1 = field.alloc((10, 10))
    f2 = field.alloc((20, 20))

    with pytest.raises(ValueError, match="shape"):
        field.combine(f1, f2, operation="add")
```

---

## Debugging Failed Tests

### 1. Run with verbose output
```bash
python -m pytest tests/test_field_operations.py::test_my_test -vv
```

### 2. Add print statements
```python
def test_my_operation():
    f = field.random((32, 32), seed=42)
    print(f"Field shape: {f.shape}")
    print(f"Field min/max: {f.data.min()}, {f.data.max()}")
    result = field.diffuse(f, rate=0.5, dt=0.1, iterations=20)
    print(f"Result shape: {result.shape}")
    assert result.shape == (32, 32)
```

Run with `-s` to see prints:
```bash
python -m pytest tests/test_my_test.py -s
```

### 3. Use pytest's debugging
```bash
# Drop into debugger on failure
python -m pytest --pdb tests/test_my_test.py
```

---

## Continuous Integration

Add to `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -e ".[dev]"
      - run: pytest --cov --cov-report=xml
      - uses: codecov/codecov-action@v3
```

---

## Test-Driven Development (TDD)

1. **Write test first** (it will fail)
```python
def test_new_feature():
    result = my_new_function(42)
    assert result == 84
```

2. **Implement feature** (make it pass)
```python
def my_new_function(x):
    return x * 2
```

3. **Refactor** (improve code quality)
```python
def my_new_function(x: int) -> int:
    """Double the input value."""
    return x * 2
```

---

## Next Steps

1. ✅ **Run all tests** and make sure they pass
2. ✅ **Check coverage** - aim for 80%+
3. ✅ **Add tests for new features** before implementing
4. ✅ **Run tests before committing** to catch issues early
5. ✅ **Set up CI/CD** to run tests automatically

---

## Resources

- **Full Testing Guide:** `docs/TESTING_STRATEGY.md`
- **pytest Documentation:** https://docs.pytest.org/
- **Coverage.py:** https://coverage.readthedocs.io/
- **Hypothesis (property testing):** https://hypothesis.readthedocs.io/

---

## Questions?

- Check existing tests in `tests/` for examples
- Read `docs/TESTING_STRATEGY.md` for comprehensive guide
- Ask in GitHub Discussions
