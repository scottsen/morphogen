"""Unit tests for basic agent operations (allocation, properties, update)."""

import pytest
import numpy as np
from morphogen.stdlib.agents import agents, Agents


class TestAgentAllocation:
    """Tests for agents.alloc operation."""

    def test_alloc_basic(self):
        """Test basic agent allocation with arrays."""
        a = agents.alloc(
            count=100,
            properties={
                'pos': np.zeros((100, 2)),
                'vel': np.ones((100, 2))
            }
        )
        assert a.count == 100
        assert a.alive_count == 100
        assert 'pos' in a.properties
        assert 'vel' in a.properties

    def test_alloc_with_scalar_broadcast(self):
        """Test allocation with scalar values (broadcast to all agents)."""
        a = agents.alloc(
            count=50,
            properties={
                'mass': 1.5,
                'temperature': 300.0
            }
        )
        assert a.count == 50
        assert np.all(a.get('mass') == 1.5)
        assert np.all(a.get('temperature') == 300.0)

    def test_alloc_mixed_types(self):
        """Test allocation with mixed scalar and array properties."""
        a = agents.alloc(
            count=100,
            properties={
                'pos': np.random.rand(100, 2),
                'vel': np.zeros((100, 2)),
                'mass': 1.0,  # Scalar
                'id': np.arange(100)  # Array
            }
        )
        assert a.count == 100
        assert a.get('pos').shape == (100, 2)
        assert np.all(a.get('mass') == 1.0)
        assert len(a.get('id')) == 100

    def test_alloc_empty(self):
        """Test allocation with zero agents."""
        a = agents.alloc(count=0, properties={})
        assert a.count == 0
        assert a.alive_count == 0

    def test_alloc_shape_mismatch_error(self):
        """Test that mismatched array shapes raise error."""
        with pytest.raises(ValueError, match="expected 100"):
            agents.alloc(
                count=100,
                properties={
                    'pos': np.zeros((50, 2))  # Wrong size!
                }
            )


class TestAgentProperties:
    """Tests for agent property access and modification."""

    def test_get_property(self):
        """Test getting property values."""
        a = agents.alloc(
            count=10,
            properties={
                'pos': np.arange(20).reshape(10, 2).astype(np.float32)
            }
        )
        pos = a.get('pos')
        assert pos.shape == (10, 2)
        assert pos[0, 0] == 0.0
        assert pos[9, 1] == 19.0

    def test_get_nonexistent_property(self):
        """Test that getting unknown property raises error."""
        a = agents.alloc(count=10, properties={'pos': np.zeros((10, 2))})
        with pytest.raises(KeyError, match="Unknown property"):
            a.get('velocity')

    def test_set_property(self):
        """Test setting property values."""
        a = agents.alloc(
            count=10,
            properties={'pos': np.zeros((10, 2))}
        )
        new_pos = np.ones((10, 2))
        a.set('pos', new_pos)
        assert np.all(a.get('pos') == 1.0)

    def test_update_property_immutability(self):
        """Test that update returns new instance."""
        a = agents.alloc(
            count=10,
            properties={'pos': np.zeros((10, 2))}
        )
        new_pos = np.ones((10, 2))
        b = a.update('pos', new_pos)

        # Original unchanged
        assert np.all(a.get('pos') == 0.0)
        # New instance updated
        assert np.all(b.get('pos') == 1.0)

    def test_copy(self):
        """Test that copy creates independent instance."""
        a = agents.alloc(
            count=10,
            properties={'pos': np.zeros((10, 2))}
        )
        b = a.copy()

        # Modify copy
        b.set('pos', np.ones((10, 2)))

        # Original unchanged
        assert np.all(a.get('pos') == 0.0)
        assert np.all(b.get('pos') == 1.0)


class TestAgentAliveMask:
    """Tests for agent alive mask functionality."""

    def test_all_alive_initially(self):
        """Test that all agents start alive."""
        a = agents.alloc(count=100, properties={'pos': np.zeros((100, 2))})
        assert a.alive_count == 100
        assert np.all(a.alive_mask)

    def test_filter_updates_alive_mask(self):
        """Test that filter updates the alive mask."""
        a = agents.alloc(
            count=100,
            properties={'id': np.arange(100)}
        )
        # Keep only even IDs
        b = agents.filter(a, 'id', lambda x: x % 2 == 0)

        assert b.alive_count == 50
        assert b.count == 100  # Total count unchanged

    def test_get_respects_alive_mask(self):
        """Test that get only returns alive agents."""
        a = agents.alloc(
            count=10,
            properties={'id': np.arange(10)}
        )
        # Mark some as dead
        a.alive_mask[5:] = False

        ids = a.get('id')
        assert len(ids) == 5  # Only first 5 alive
        assert np.all(ids == np.arange(5))


class TestAgentCount:
    """Tests for agent count tracking."""

    def test_count_vs_alive_count(self):
        """Test difference between total count and alive count."""
        a = agents.alloc(count=100, properties={'id': np.arange(100)})
        assert a.count == 100
        assert a.alive_count == 100

        # Filter to 50 agents
        b = agents.filter(a, 'id', lambda x: x < 50)
        assert b.count == 100  # Total capacity unchanged
        assert b.alive_count == 50  # Only 50 alive

    def test_empty_agents(self):
        """Test empty agent collection."""
        a = agents.alloc(count=0, properties={})
        assert a.count == 0
        assert a.alive_count == 0

    def test_single_agent(self):
        """Test single agent collection."""
        a = agents.alloc(
            count=1,
            properties={'pos': np.array([[1.0, 2.0]])}
        )
        assert a.count == 1
        assert a.alive_count == 1
        assert a.get('pos').shape == (1, 2)


class TestAgentRepr:
    """Tests for agent string representation."""

    def test_repr_shows_count(self):
        """Test that repr shows agent count."""
        a = agents.alloc(
            count=100,
            properties={'pos': np.zeros((100, 2)), 'vel': np.zeros((100, 2))}
        )
        repr_str = repr(a)
        assert '100' in repr_str
        assert 'pos' in repr_str
        assert 'vel' in repr_str

    def test_repr_shows_alive_count(self):
        """Test that repr shows alive vs total count."""
        a = agents.alloc(count=100, properties={'id': np.arange(100)})
        b = agents.filter(a, 'id', lambda x: x < 50)
        repr_str = repr(b)
        assert '50/100' in repr_str  # 50 alive out of 100 total


class TestAgentDeterminism:
    """Tests for deterministic agent operations."""

    def test_deterministic_allocation(self):
        """Test that allocation with same data is deterministic."""
        rng = np.random.RandomState(42)
        pos1 = rng.rand(100, 2)

        rng = np.random.RandomState(42)
        pos2 = rng.rand(100, 2)

        a1 = agents.alloc(count=100, properties={'pos': pos1})
        a2 = agents.alloc(count=100, properties={'pos': pos2})

        assert np.allclose(a1.get('pos'), a2.get('pos'))

    def test_copy_preserves_data(self):
        """Test that copy preserves exact data."""
        rng = np.random.RandomState(42)
        a = agents.alloc(
            count=100,
            properties={'pos': rng.rand(100, 2)}
        )
        b = a.copy()

        assert np.array_equal(a.get('pos'), b.get('pos'))


class TestAgentEdgeCases:
    """Tests for edge cases and error handling."""

    def test_property_name_validation(self):
        """Test that property names are validated."""
        a = agents.alloc(count=10, properties={'pos': np.zeros((10, 2))})

        # Getting non-existent property
        with pytest.raises(KeyError):
            a.get('nonexistent')

        # Setting non-existent property
        with pytest.raises(KeyError):
            a.set('nonexistent', np.zeros((10, 2)))

    def test_non_array_property_error(self):
        """Test that non-array properties are rejected."""
        with pytest.raises(TypeError):
            Agents(
                count=10,
                properties={'pos': "not an array"}  # String, not array
            )

    def test_vector_properties(self):
        """Test agents with vector properties of different dimensions."""
        a = agents.alloc(
            count=100,
            properties={
                'pos2d': np.zeros((100, 2)),
                'pos3d': np.zeros((100, 3)),
                'color': np.zeros((100, 4))  # RGBA
            }
        )
        assert a.get('pos2d').shape == (100, 2)
        assert a.get('pos3d').shape == (100, 3)
        assert a.get('color').shape == (100, 4)

    def test_scalar_properties(self):
        """Test agents with scalar properties."""
        a = agents.alloc(
            count=100,
            properties={
                'mass': 1.0,
                'charge': -1.6e-19,
                'id': np.arange(100)
            }
        )
        assert a.get('mass').shape == (100,)
        assert a.get('charge').shape == (100,)
        assert a.get('id').shape == (100,)

    def test_large_agent_count(self):
        """Test allocation of large number of agents."""
        # 10,000 agents should work fine
        a = agents.alloc(
            count=10000,
            properties={
                'pos': np.random.rand(10000, 2),
                'vel': np.zeros((10000, 2))
            }
        )
        assert a.count == 10000
        assert a.alive_count == 10000
        assert a.get('pos').shape == (10000, 2)
