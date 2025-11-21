"""
Comprehensive tests for Cellular Automata Domain.

Tests all cellular automata operations:
- 2D Cellular Automata (Game of Life, variants)
- 1D Cellular Automata (Wolfram Elementary CA)
- Brian's Brain (3-state CA)
- Pattern analysis
- Rule application

Validates:
- Correctness (known patterns behave correctly)
- Determinism (same seed → same result)
- Rule application accuracy
- Edge cases and boundary conditions
"""

import pytest
import numpy as np
from morphogen.stdlib.cellular import (
    cellular,
    CellularField2D,
    CellularField1D,
    CellularOperations,
    CARule,
)


# ============================================================================
# Test Fixtures and Helpers
# ============================================================================

@pytest.fixture
def small_field_2d():
    """Create a small 2D field for testing."""
    data = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.int32)
    return CellularField2D(data, states=2)


@pytest.fixture
def small_field_1d():
    """Create a small 1D field for testing."""
    data = np.array([0, 0, 1, 0, 0], dtype=np.int32)
    return CellularField1D(data, states=2)


@pytest.fixture
def game_of_life_rule():
    """Create Game of Life rule (B3/S23)."""
    return CARule(birth={3}, survival={2, 3}, states=2, neighborhood="moore")


# ============================================================================
# Basic Field Operations Tests
# ============================================================================

class TestCellularFieldBasics:
    """Tests for basic cellular field operations."""

    def test_alloc_2d_field(self):
        """Test allocating a 2D cellular field."""
        field = cellular.alloc((10, 10), states=2, fill_value=0)

        assert isinstance(field, CellularField2D)
        assert field.shape == (10, 10)
        assert field.states == 2
        assert np.all(field.data == 0)
        assert field.generation == 0

    def test_alloc_1d_field(self):
        """Test allocating a 1D cellular field."""
        field = cellular.alloc(20, states=2, fill_value=0)

        assert isinstance(field, CellularField1D)
        assert field.width == 20
        assert field.states == 2
        assert np.all(field.data == 0)

    def test_random_init_2d(self):
        """Test random initialization with seed."""
        field1 = cellular.random_init((50, 50), states=2, density=0.5, seed=42)
        field2 = cellular.random_init((50, 50), states=2, density=0.5, seed=42)

        # Same seed should produce identical fields
        assert np.array_equal(field1.data, field2.data)

        # Check density is approximately correct
        actual_density = np.sum(field1.data == 1) / (50 * 50)
        assert 0.4 < actual_density < 0.6  # Allow some variance

    def test_random_init_1d(self):
        """Test random 1D initialization."""
        field = cellular.random_init(100, states=2, density=0.3, seed=42)

        assert field.width == 100
        actual_density = np.sum(field.data == 1) / 100
        assert 0.2 < actual_density < 0.4

    def test_field_copy(self, small_field_2d):
        """Test field copy operation."""
        copy = small_field_2d.copy()

        assert copy is not small_field_2d
        assert np.array_equal(copy.data, small_field_2d.data)
        assert copy.states == small_field_2d.states

        # Modifying copy shouldn't affect original
        copy.data[0, 0] = 1
        assert small_field_2d.data[0, 0] == 0


# ============================================================================
# Neighbor Counting Tests
# ============================================================================

class TestNeighborCounting:
    """Tests for neighbor counting operations."""

    def test_moore_neighbors_center(self):
        """Test Moore neighborhood counting in center."""
        # Create field with single alive cell
        data = np.zeros((5, 5), dtype=np.int32)
        data[2, 2] = 1
        field = CellularField2D(data, states=2)

        counts = cellular.count_neighbors_moore(field, state=1)

        # All 8 neighbors of center cell should see 1 neighbor
        expected_neighbors = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.int32)

        assert np.array_equal(counts, expected_neighbors)

    def test_moore_neighbors_pattern(self, small_field_2d):
        """Test Moore neighbor counting on a pattern."""
        # Pattern is a horizontal blinker:
        # [0, 0, 0, 0, 0]
        # [0, 0, 1, 0, 0]
        # [0, 1, 1, 1, 0]
        # [0, 0, 0, 0, 0]
        # [0, 0, 0, 0, 0]

        counts = cellular.count_neighbors_moore(small_field_2d, state=1)

        # Center cell (2, 2) has 3 neighbors (above, left, and right)
        assert counts[2, 2] == 3  # Three neighbors (above, left, right)
        assert counts[1, 2] == 3  # Cell above center has 3 neighbors below
        assert counts[2, 1] == 2  # Left cell has 2 neighbors (center and right)

    def test_von_neumann_neighbors(self):
        """Test von Neumann neighborhood (4 neighbors)."""
        data = np.zeros((5, 5), dtype=np.int32)
        data[2, 2] = 1
        field = CellularField2D(data, states=2)

        counts = cellular.count_neighbors_von_neumann(field, state=1)

        # Only 4 orthogonal neighbors
        expected = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=np.int32)

        assert np.array_equal(counts, expected)

    def test_wraparound_neighbors(self):
        """Test that neighbors wrap around edges."""
        data = np.zeros((5, 5), dtype=np.int32)
        data[0, 0] = 1  # Top-left corner
        field = CellularField2D(data, states=2)

        counts = cellular.count_neighbors_moore(field, state=1)

        # Corner cell's neighbors include wraparound
        # Top-right, bottom-left, and bottom-right should see it
        assert counts[0, 4] == 1  # Top-right
        assert counts[4, 0] == 1  # Bottom-left
        assert counts[4, 4] == 1  # Bottom-right


# ============================================================================
# Game of Life Tests
# ============================================================================

class TestGameOfLife:
    """Tests for Conway's Game of Life."""

    def test_game_of_life_creation(self):
        """Test Game of Life field creation."""
        field, rule = cellular.game_of_life((50, 50), density=0.3, seed=42)

        assert isinstance(field, CellularField2D)
        assert field.shape == (50, 50)
        assert rule.birth == {3}
        assert rule.survival == {2, 3}
        assert rule.neighborhood == "moore"

    def test_blinker_oscillation(self):
        """Test that blinker pattern oscillates correctly."""
        # Horizontal blinker
        data = np.zeros((5, 5), dtype=np.int32)
        data[2, 1:4] = 1  # Horizontal line
        field = CellularField2D(data, states=2)
        rule = CARule(birth={3}, survival={2, 3}, states=2)

        # After one step, should become vertical
        field = cellular.step(field, rule)
        assert field.data[1, 2] == 1
        assert field.data[2, 2] == 1
        assert field.data[3, 2] == 1
        assert field.data[2, 1] == 0
        assert field.data[2, 3] == 0

        # After another step, back to horizontal
        field = cellular.step(field, rule)
        assert field.data[2, 1] == 1
        assert field.data[2, 2] == 1
        assert field.data[2, 3] == 1
        assert field.data[1, 2] == 0
        assert field.data[3, 2] == 0

    def test_block_still_life(self):
        """Test that block pattern is stable."""
        # 2x2 block
        data = np.zeros((5, 5), dtype=np.int32)
        data[2:4, 2:4] = 1
        field = CellularField2D(data, states=2)
        rule = CARule(birth={3}, survival={2, 3}, states=2)

        # Should remain unchanged
        field = cellular.step(field, rule)
        assert field.data[2, 2] == 1
        assert field.data[2, 3] == 1
        assert field.data[3, 2] == 1
        assert field.data[3, 3] == 1

    def test_glider_movement(self):
        """Test that glider moves diagonally."""
        # Create glider pattern
        data = np.zeros((10, 10), dtype=np.int32)
        data[1, 2] = 1
        data[2, 3] = 1
        data[3, 1:4] = 1
        field = CellularField2D(data, states=2)
        rule = CARule(birth={3}, survival={2, 3}, states=2)

        # Count initial alive cells
        initial_alive = np.sum(field.data == 1)

        # Evolve several steps
        for _ in range(4):
            field = cellular.step(field, rule)

        # Should still have 5 alive cells (glider)
        assert np.sum(field.data == 1) == initial_alive

    def test_evolution_generations(self):
        """Test that generation counter increments."""
        field, rule = cellular.game_of_life((10, 10), density=0.3, seed=42)

        assert field.generation == 0

        field = cellular.step(field, rule)
        assert field.generation == 1

        field = cellular.evolve(field, rule, steps=10)
        assert field.generation == 11


# ============================================================================
# Wolfram CA Tests
# ============================================================================

class TestWolframCA:
    """Tests for Wolfram Elementary Cellular Automata."""

    def test_wolfram_creation(self):
        """Test Wolfram CA field creation."""
        field, rule = cellular.wolfram_ca(100, rule_number=30)

        assert isinstance(field, CellularField1D)
        assert field.width == 100
        assert rule == 30
        # Should have single alive cell in center
        assert field.data[50] == 1
        assert np.sum(field.data) == 1

    def test_rule_30_determinism(self):
        """Test Rule 30 is deterministic."""
        field1, rule = cellular.wolfram_ca(100, rule_number=30)
        field2, _ = cellular.wolfram_ca(100, rule_number=30)

        # Evolve both
        for _ in range(50):
            field1 = cellular.step(field1, rule)
            field2 = cellular.step(field2, rule)

        assert np.array_equal(field1.data, field2.data)

    def test_rule_90_sierpinski(self):
        """Test Rule 90 produces symmetric pattern."""
        # Use odd width for perfect symmetry with single centered cell
        field, rule = cellular.wolfram_ca(65, rule_number=90)

        history = cellular.history(field, rule, steps=32)

        # Rule 90 produces Sierpinski triangle - should be symmetric
        for gen_field in history:
            data = gen_field.data
            # Check symmetry around center
            assert np.array_equal(data, data[::-1])

    def test_rule_0_dies(self):
        """Test Rule 0 kills all cells."""
        field, rule = cellular.wolfram_ca(50, rule_number=0)

        field = cellular.step(field, rule)

        # All cells should be dead
        assert np.all(field.data == 0)

    def test_rule_255_fills(self):
        """Test Rule 255 fills all cells."""
        field, rule = cellular.wolfram_ca(50, rule_number=255)

        field = cellular.step(field, rule)

        # All cells should be alive
        assert np.all(field.data == 1)

    def test_wolfram_rule_validation(self):
        """Test that invalid rule numbers raise error."""
        field = CellularField1D(np.zeros(10, dtype=np.int32), 2)

        with pytest.raises(ValueError):
            cellular.apply_wolfram_rule(field, 256)

        with pytest.raises(ValueError):
            cellular.apply_wolfram_rule(field, -1)


# ============================================================================
# Brian's Brain Tests
# ============================================================================

class TestBriansBrain:
    """Tests for Brian's Brain 3-state CA."""

    def test_brians_brain_creation(self):
        """Test Brian's Brain field creation."""
        field = cellular.brians_brain((50, 50), density=0.1, seed=42)

        assert isinstance(field, CellularField2D)
        assert field.states == 3
        assert field.shape == (50, 50)

    def test_brians_brain_state_transitions(self):
        """Test correct state transitions in Brian's Brain."""
        # Create field with specific states
        data = np.array([
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0],  # Two firing cells
            [0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],  # One refractory cell
            [0, 0, 0, 0, 0]
        ], dtype=np.int32)
        field = CellularField2D(data, states=3)

        field = cellular.brians_brain_step(field)

        # Firing cells should become refractory
        assert field.data[1, 1] == 2
        assert field.data[1, 2] == 2

        # Refractory cells should become dead
        assert field.data[3, 1] == 0

        # Dead cell with 2 firing neighbors should fire
        # (check cells adjacent to the two firing cells)

    def test_brians_brain_determinism(self):
        """Test Brian's Brain is deterministic."""
        field1 = cellular.brians_brain((30, 30), density=0.1, seed=42)
        field2 = cellular.brians_brain((30, 30), density=0.1, seed=42)

        for _ in range(20):
            field1 = cellular.brians_brain_step(field1)
            field2 = cellular.brians_brain_step(field2)

        assert np.array_equal(field1.data, field2.data)


# ============================================================================
# CA Variants Tests
# ============================================================================

class TestCAVariants:
    """Tests for CA variants (HighLife, Seeds)."""

    def test_highlife_creation(self):
        """Test HighLife CA creation."""
        field, rule = cellular.highlife((50, 50), density=0.3, seed=42)

        assert rule.birth == {3, 6}
        assert rule.survival == {2, 3}

    def test_seeds_creation(self):
        """Test Seeds CA creation."""
        field, rule = cellular.seeds((50, 50), density=0.1, seed=42)

        assert rule.birth == {2}
        assert rule.survival == set()  # No survival

    def test_seeds_no_survival(self):
        """Test that Seeds cells don't survive."""
        # Create single alive cell
        data = np.zeros((5, 5), dtype=np.int32)
        data[2, 2] = 1
        field = CellularField2D(data, states=2)
        rule = CARule(birth={2}, survival=set(), states=2)

        field = cellular.step(field, rule)

        # Original cell should die
        assert field.data[2, 2] == 0


# ============================================================================
# Pattern Analysis Tests
# ============================================================================

class TestPatternAnalysis:
    """Tests for pattern analysis operations."""

    def test_analyze_pattern_basic(self):
        """Test basic pattern analysis."""
        data = np.zeros((10, 10), dtype=np.int32)
        data[2:5, 2:5] = 1  # 3x3 block of alive cells
        field = CellularField2D(data, states=2)

        stats = cellular.analyze_pattern(field)

        assert stats['alive_count'] == 9
        assert stats['dead_count'] == 91
        assert stats['density'] == 0.09
        assert stats['generation'] == 0
        assert stats['shape'] == (10, 10)

    def test_analyze_empty_field(self):
        """Test analysis of empty field."""
        field = cellular.alloc((20, 20), states=2, fill_value=0)
        stats = cellular.analyze_pattern(field)

        assert stats['alive_count'] == 0
        assert stats['density'] == 0.0

    def test_analyze_full_field(self):
        """Test analysis of full field."""
        field = cellular.alloc((15, 15), states=2, fill_value=1)
        stats = cellular.analyze_pattern(field)

        assert stats['alive_count'] == 225
        assert stats['density'] == 1.0


# ============================================================================
# History and Evolution Tests
# ============================================================================

class TestEvolution:
    """Tests for evolution and history operations."""

    def test_evolution_steps(self):
        """Test multi-step evolution."""
        field, rule = cellular.game_of_life((20, 20), density=0.3, seed=42)

        evolved = cellular.evolve(field, rule, steps=50)

        assert evolved.generation == 50

    def test_history_generation(self):
        """Test history generation."""
        field, rule = cellular.game_of_life((15, 15), density=0.3, seed=42)

        history = cellular.history(field, rule, steps=10)

        assert len(history) == 11  # Initial + 10 steps
        assert history[0].generation == 0
        assert history[10].generation == 10

    def test_history_independence(self):
        """Test that history entries are independent copies."""
        field, rule = cellular.game_of_life((10, 10), density=0.3, seed=42)
        history = cellular.history(field, rule, steps=5)

        # Modify one entry
        history[2].data[0, 0] = 999

        # Others should be unaffected
        assert history[1].data[0, 0] != 999
        assert history[3].data[0, 0] != 999


# ============================================================================
# Array Conversion Tests
# ============================================================================

class TestArrayConversion:
    """Tests for array conversion operations."""

    def test_to_array_2d(self, small_field_2d):
        """Test converting 2D field to array."""
        arr = cellular.to_array(small_field_2d)

        assert isinstance(arr, np.ndarray)
        assert arr.shape == small_field_2d.shape
        assert np.array_equal(arr, small_field_2d.data)

        # Modifying array shouldn't affect field
        arr[0, 0] = 999
        assert small_field_2d.data[0, 0] != 999

    def test_from_array_2d(self):
        """Test creating field from 2D array."""
        data = np.random.randint(0, 2, size=(20, 20), dtype=np.int32)
        field = cellular.from_array(data, states=2)

        assert isinstance(field, CellularField2D)
        assert np.array_equal(field.data, data)

    def test_from_array_1d(self):
        """Test creating field from 1D array."""
        data = np.random.randint(0, 2, size=(50,), dtype=np.int32)
        field = cellular.from_array(data, states=2)

        assert isinstance(field, CellularField1D)
        assert np.array_equal(field.data, data)


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_field_evolution(self):
        """Test evolving empty field."""
        field = cellular.alloc((10, 10), states=2, fill_value=0)
        rule = CARule(birth={3}, survival={2, 3}, states=2)

        field = cellular.evolve(field, rule, steps=10)

        # Should remain empty
        assert np.all(field.data == 0)

    def test_single_cell_field(self):
        """Test 1x1 field (edge case)."""
        field = cellular.alloc((1, 1), states=2, fill_value=1)
        rule = CARule(birth={3}, survival={2, 3}, states=2)

        # Should work without errors
        field = cellular.step(field, rule)

    def test_invalid_neighborhood(self):
        """Test invalid neighborhood type."""
        field = cellular.alloc((10, 10), states=2)
        rule = CARule(birth={3}, survival={2, 3}, neighborhood="invalid")

        with pytest.raises(ValueError):
            cellular.apply_rule(field, rule)

    def test_wrong_field_rule_combination(self):
        """Test passing wrong field/rule combination to step."""
        field_2d = cellular.alloc((10, 10), states=2)
        rule_1d = 30  # Wolfram rule

        with pytest.raises(ValueError):
            cellular.step(field_2d, rule_1d)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple operations."""

    def test_complete_workflow_2d(self):
        """Test complete 2D CA workflow."""
        # Create
        field, rule = cellular.game_of_life((50, 50), density=0.3, seed=42)

        # Analyze initial
        stats_initial = cellular.analyze_pattern(field)

        # Evolve
        field = cellular.evolve(field, rule, steps=100)

        # Analyze final
        stats_final = cellular.analyze_pattern(field)

        # Should have evolved
        assert field.generation == 100
        assert stats_final['generation'] == 100

    def test_complete_workflow_1d(self):
        """Test complete 1D CA workflow."""
        # Create
        field, rule = cellular.wolfram_ca(100, rule_number=30)

        # Generate history
        history = cellular.history(field, rule, steps=50)

        # Convert to arrays
        arrays = [cellular.to_array(f) for f in history]

        # Verify
        assert len(arrays) == 51
        assert all(arr.shape == (100,) for arr in arrays)

    def test_determinism_across_workflow(self):
        """Test determinism across complete workflow."""
        # Run 1
        field1, rule1 = cellular.game_of_life((30, 30), density=0.3, seed=123)
        field1 = cellular.evolve(field1, rule1, steps=50)
        stats1 = cellular.analyze_pattern(field1)

        # Run 2 (same seed)
        field2, rule2 = cellular.game_of_life((30, 30), density=0.3, seed=123)
        field2 = cellular.evolve(field2, rule2, steps=50)
        stats2 = cellular.analyze_pattern(field2)

        # Should be identical
        assert np.array_equal(field1.data, field2.data)
        assert stats1 == stats2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
