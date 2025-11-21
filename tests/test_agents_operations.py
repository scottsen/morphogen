"""Unit tests for agent operations (map, filter, reduce)."""

import pytest
import numpy as np
from morphogen.stdlib.agents import agents, Agents


class TestAgentMap:
    """Tests for agents.map operation."""

    def test_map_add_constant(self):
        """Test mapping addition of constant to property."""
        a = agents.alloc(
            count=100,
            properties={'pos': np.zeros((100, 2))}
        )

        # Add 1.0 to all positions (vectorized)
        new_pos = agents.map(a, 'pos', lambda p: p + 1.0)
        assert new_pos.shape == (100, 2)
        assert np.all(new_pos == 1.0)

    def test_map_scale(self):
        """Test mapping scalar multiplication."""
        a = agents.alloc(
            count=50,
            properties={'vel': np.ones((50, 2))}
        )

        # Scale velocities by 2.0
        new_vel = agents.map(a, 'vel', lambda v: v * 2.0)
        assert np.all(new_vel == 2.0)

    def test_map_normalize_vectors(self):
        """Test mapping vector normalization."""
        # Create random non-zero vectors
        rng = np.random.RandomState(42)
        vecs = rng.rand(100, 2) + 0.1  # Add 0.1 to avoid zeros

        a = agents.alloc(count=100, properties={'dir': vecs})

        # Normalize
        def normalize(v):
            norms = np.linalg.norm(v, axis=1, keepdims=True)
            return v / (norms + 1e-10)

        normalized = agents.map(a, 'dir', normalize)

        # Check all have unit length
        lengths = np.linalg.norm(normalized, axis=1)
        assert np.allclose(lengths, 1.0, atol=1e-5)

    def test_map_apply_function(self):
        """Test mapping arbitrary function."""
        a = agents.alloc(
            count=100,
            properties={'x': np.linspace(0, 2*np.pi, 100)}
        )

        # Apply sin
        result = agents.map(a, 'x', np.sin)
        expected = np.sin(np.linspace(0, 2*np.pi, 100))
        assert np.allclose(result, expected)

    def test_map_respects_alive_mask(self):
        """Test that map only operates on alive agents."""
        a = agents.alloc(
            count=100,
            properties={'x': np.ones(100)}
        )

        # Kill half the agents
        a.alive_mask[50:] = False

        # Map should only return 50 values
        result = agents.map(a, 'x', lambda x: x * 2)
        assert len(result) == 50
        assert np.all(result == 2.0)

    def test_map_vector_operations(self):
        """Test mapping on vector properties."""
        a = agents.alloc(
            count=100,
            properties={'pos': np.ones((100, 2))}
        )

        # Translate by vector
        offset = np.array([10.0, 20.0])
        new_pos = agents.map(a, 'pos', lambda p: p + offset)

        assert new_pos.shape == (100, 2)
        assert np.allclose(new_pos[:, 0], 11.0)
        assert np.allclose(new_pos[:, 1], 21.0)


class TestAgentFilter:
    """Tests for agents.filter operation."""

    def test_filter_simple_condition(self):
        """Test filtering with simple condition."""
        a = agents.alloc(
            count=100,
            properties={'id': np.arange(100)}
        )

        # Keep only agents with id < 50
        b = agents.filter(a, 'id', lambda x: x < 50)

        assert b.alive_count == 50
        assert b.count == 100  # Total capacity unchanged
        ids = b.get('id')
        assert np.all(ids < 50)

    def test_filter_position_based(self):
        """Test filtering based on position."""
        rng = np.random.RandomState(42)
        positions = rng.rand(100, 2) * 10.0  # Positions in [0, 10)

        a = agents.alloc(count=100, properties={'pos': positions})

        # Keep only agents in left half (x < 5)
        b = agents.filter(a, 'pos', lambda p: p[:, 0] < 5.0)

        remaining_pos = b.get('pos')
        assert np.all(remaining_pos[:, 0] < 5.0)

    def test_filter_multiple_times(self):
        """Test applying filter multiple times."""
        a = agents.alloc(
            count=100,
            properties={'id': np.arange(100)}
        )

        # First filter: keep id < 75
        b = agents.filter(a, 'id', lambda x: x < 75)
        assert b.alive_count == 75

        # Second filter: keep id >= 25
        c = agents.filter(b, 'id', lambda x: x >= 25)
        assert c.alive_count == 50  # Only [25, 75) remain

        ids = c.get('id')
        assert np.all(ids >= 25)
        assert np.all(ids < 75)

    def test_filter_no_matches(self):
        """Test filter that matches no agents."""
        a = agents.alloc(
            count=100,
            properties={'id': np.arange(100)}
        )

        # Filter that matches nothing
        b = agents.filter(a, 'id', lambda x: x > 1000)
        assert b.alive_count == 0

    def test_filter_all_match(self):
        """Test filter that matches all agents."""
        a = agents.alloc(
            count=100,
            properties={'id': np.arange(100)}
        )

        # Filter that matches everything
        b = agents.filter(a, 'id', lambda x: x >= 0)
        assert b.alive_count == 100

    def test_filter_even_odd(self):
        """Test filtering even/odd values."""
        a = agents.alloc(
            count=100,
            properties={'id': np.arange(100)}
        )

        # Keep only even IDs
        b = agents.filter(a, 'id', lambda x: x % 2 == 0)
        assert b.alive_count == 50

        ids = b.get('id')
        assert np.all(ids % 2 == 0)

    def test_filter_preserves_other_properties(self):
        """Test that filter preserves all properties."""
        a = agents.alloc(
            count=100,
            properties={
                'id': np.arange(100),
                'pos': np.random.rand(100, 2),
                'mass': np.ones(100)
            }
        )

        b = agents.filter(a, 'id', lambda x: x < 50)

        # All properties still exist
        assert 'pos' in b.properties
        assert 'mass' in b.properties
        assert 'id' in b.properties

        # And have correct alive count
        assert len(b.get('pos')) == 50
        assert len(b.get('mass')) == 50


class TestAgentReduce:
    """Tests for agents.reduce operation."""

    def test_reduce_sum(self):
        """Test sum reduction."""
        a = agents.alloc(
            count=100,
            properties={'mass': np.ones(100) * 2.0}
        )

        total_mass = agents.reduce(a, 'mass', operation='sum')
        assert np.isclose(total_mass, 200.0)

    def test_reduce_mean(self):
        """Test mean reduction."""
        a = agents.alloc(
            count=100,
            properties={'temp': np.ones(100) * 300.0}
        )

        avg_temp = agents.reduce(a, 'temp', operation='mean')
        assert np.isclose(avg_temp, 300.0)

    def test_reduce_min_max(self):
        """Test min and max reductions."""
        a = agents.alloc(
            count=100,
            properties={'id': np.arange(100)}
        )

        min_id = agents.reduce(a, 'id', operation='min')
        max_id = agents.reduce(a, 'id', operation='max')

        assert min_id == 0
        assert max_id == 99

    def test_reduce_product(self):
        """Test product reduction."""
        a = agents.alloc(
            count=5,
            properties={'x': np.array([1.0, 2.0, 3.0, 4.0, 5.0])}
        )

        product = agents.reduce(a, 'x', operation='prod')
        assert np.isclose(product, 120.0)  # 1*2*3*4*5 = 120

    def test_reduce_respects_alive_mask(self):
        """Test that reduce only considers alive agents."""
        a = agents.alloc(
            count=100,
            properties={'x': np.ones(100)}
        )

        # Kill half
        a.alive_mask[50:] = False

        total = agents.reduce(a, 'x', operation='sum')
        assert np.isclose(total, 50.0)  # Only 50 alive agents

    def test_reduce_empty_agents(self):
        """Test reduce on empty agent collection."""
        a = agents.alloc(count=0, properties={'x': np.array([])})

        # Sum of empty is 0
        total = agents.reduce(a, 'x', operation='sum')
        assert total == 0.0

    def test_reduce_single_agent(self):
        """Test reduce on single agent."""
        a = agents.alloc(
            count=1,
            properties={'x': np.array([42.0])}
        )

        result = agents.reduce(a, 'x', operation='sum')
        assert result == 42.0

    def test_reduce_vector_property(self):
        """Test reducing vector properties."""
        a = agents.alloc(
            count=100,
            properties={'pos': np.ones((100, 2))}
        )

        # Sum of positions - NumPy sum flattens by default
        total_pos = agents.reduce(a, 'pos', operation='sum')
        # NumPy sum() on 2D array returns scalar (sums all elements)
        assert np.isclose(total_pos, 200.0)  # 100 agents * 2 components

    def test_reduce_unknown_operation(self):
        """Test that unknown operation raises error."""
        a = agents.alloc(count=10, properties={'x': np.ones(10)})

        with pytest.raises(ValueError, match="Unknown reduction"):
            agents.reduce(a, 'x', operation='unknown')


class TestAgentComposition:
    """Tests for composing multiple agent operations."""

    def test_map_then_reduce(self):
        """Test mapping then reducing."""
        a = agents.alloc(
            count=100,
            properties={'x': np.arange(100, dtype=np.float32)}
        )

        # Square all values
        squared = agents.map(a, 'x', lambda x: x ** 2)

        # Create new agents with squared values
        b = a.copy()
        b.set('x', squared)

        # Sum of squares
        sum_of_squares = agents.reduce(b, 'x', operation='sum')

        # Verify: sum(i^2 for i in range(100)) = 328350
        expected = sum(i**2 for i in range(100))
        assert np.isclose(sum_of_squares, expected)

    def test_filter_then_map(self):
        """Test filtering then mapping."""
        a = agents.alloc(
            count=100,
            properties={'id': np.arange(100, dtype=np.float32)}
        )

        # Filter to first 50
        b = agents.filter(a, 'id', lambda x: x < 50)
        assert b.alive_count == 50

        # Map: add 100
        result = agents.map(b, 'id', lambda x: x + 100)

        assert len(result) == 50
        assert np.allclose(result, np.arange(100, 150))

    def test_filter_then_reduce(self):
        """Test filtering then reducing."""
        a = agents.alloc(
            count=100,
            properties={'mass': np.ones(100)}
        )

        # Keep only half
        b = agents.filter(a, 'mass', lambda m: np.arange(len(m)) < 50)

        # Sum remaining masses
        total = agents.reduce(b, 'mass', operation='sum')
        assert np.isclose(total, 50.0)

    def test_multiple_filters(self):
        """Test chaining multiple filters."""
        a = agents.alloc(
            count=1000,
            properties={'id': np.arange(1000)}
        )

        # Keep id >= 100
        b = agents.filter(a, 'id', lambda x: x >= 100)
        assert b.alive_count == 900

        # Keep id < 900
        c = agents.filter(b, 'id', lambda x: x < 900)
        assert c.alive_count == 800

        # Keep even IDs
        d = agents.filter(c, 'id', lambda x: x % 2 == 0)
        assert d.alive_count == 400  # [100, 900) step 2


class TestAgentDeterminism:
    """Tests for deterministic agent operations."""

    def test_map_deterministic(self):
        """Test that map produces deterministic results."""
        rng = np.random.RandomState(42)
        data1 = rng.rand(100, 2)

        rng = np.random.RandomState(42)
        data2 = rng.rand(100, 2)

        a1 = agents.alloc(count=100, properties={'pos': data1})
        a2 = agents.alloc(count=100, properties={'pos': data2})

        result1 = agents.map(a1, 'pos', lambda p: p * 2.0)
        result2 = agents.map(a2, 'pos', lambda p: p * 2.0)

        assert np.allclose(result1, result2)

    def test_filter_deterministic(self):
        """Test that filter produces deterministic results."""
        rng = np.random.RandomState(42)
        data1 = rng.rand(100)

        rng = np.random.RandomState(42)
        data2 = rng.rand(100)

        a1 = agents.alloc(count=100, properties={'x': data1})
        a2 = agents.alloc(count=100, properties={'x': data2})

        b1 = agents.filter(a1, 'x', lambda x: x > 0.5)
        b2 = agents.filter(a2, 'x', lambda x: x > 0.5)

        assert b1.alive_count == b2.alive_count
        assert np.array_equal(b1.alive_mask, b2.alive_mask)

    def test_reduce_deterministic(self):
        """Test that reduce produces deterministic results."""
        rng = np.random.RandomState(42)
        data1 = rng.rand(100)

        rng = np.random.RandomState(42)
        data2 = rng.rand(100)

        a1 = agents.alloc(count=100, properties={'x': data1})
        a2 = agents.alloc(count=100, properties={'x': data2})

        sum1 = agents.reduce(a1, 'x', operation='sum')
        sum2 = agents.reduce(a2, 'x', operation='sum')

        assert np.isclose(sum1, sum2)
