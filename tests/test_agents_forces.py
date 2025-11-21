"""Unit tests for agent force calculations and field coupling."""

import pytest
import numpy as np
from morphogen.stdlib.agents import agents, Agents
from morphogen.stdlib.field import field, Field2D


class TestPairwiseForcesBrute:
    """Tests for brute-force pairwise force calculations."""

    def test_simple_repulsion_2d(self):
        """Test simple repulsive forces in 2D."""
        # Two agents close together
        a = agents.alloc(
            count=2,
            properties={
                'pos': np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
            }
        )

        # Simple repulsion: force points away from other agent
        def repulsion(pi, pj):
            delta = pi - pj
            dist = np.linalg.norm(delta)
            if dist < 0.1:
                return np.zeros(2, dtype=np.float32)
            return delta / dist  # Unit vector away

        forces = agents.compute_pairwise_forces(
            a,
            radius=2.0,
            force_func=repulsion,
            use_spatial_hashing=False  # Use brute force
        )

        # Agent 0 should be pushed left (negative x)
        assert forces[0, 0] < 0
        # Agent 1 should be pushed right (positive x)
        assert forces[1, 0] > 0

    def test_gravity_with_mass(self):
        """Test gravitational forces with mass."""
        # Two agents with different masses
        a = agents.alloc(
            count=2,
            properties={
                'pos': np.array([[0.0, 0.0], [10.0, 0.0]], dtype=np.float32),
                'mass': np.array([1.0, 2.0], dtype=np.float32)
            }
        )

        # Gravitational force
        def gravity(pi, pj, mi, mj):
            delta = pj - pi
            dist = np.linalg.norm(delta)
            if dist < 0.1:
                return np.zeros(2, dtype=np.float32)
            force_mag = mi * mj / (dist ** 2)
            return (delta / dist) * force_mag

        forces = agents.compute_pairwise_forces(
            a,
            radius=100.0,
            force_func=gravity,
            mass_property='mass',
            use_spatial_hashing=False
        )

        # Both agents should be pulled toward each other
        assert forces[0, 0] > 0  # Agent 0 pulled right
        assert forces[1, 0] < 0  # Agent 1 pulled left

    def test_forces_beyond_radius(self):
        """Test that forces beyond radius are zero."""
        # Two agents far apart
        a = agents.alloc(
            count=2,
            properties={
                'pos': np.array([[0.0, 0.0], [100.0, 0.0]], dtype=np.float32)
            }
        )

        def constant_force(pi, pj):
            return np.array([1.0, 0.0], dtype=np.float32)

        forces = agents.compute_pairwise_forces(
            a,
            radius=10.0,  # Too small to reach
            force_func=constant_force,
            use_spatial_hashing=False
        )

        # No forces should be applied
        assert np.allclose(forces, 0.0)

    def test_multiple_agents_forces(self):
        """Test forces with multiple agents."""
        # 4 agents in a square
        a = agents.alloc(
            count=4,
            properties={
                'pos': np.array([
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0]
                ], dtype=np.float32)
            }
        )

        # Repulsion
        def repulsion(pi, pj):
            delta = pi - pj
            dist = np.linalg.norm(delta)
            if dist < 0.1:
                return np.zeros(2, dtype=np.float32)
            return delta / (dist ** 2)

        forces = agents.compute_pairwise_forces(
            a,
            radius=2.0,
            force_func=repulsion,
            use_spatial_hashing=False
        )

        # Corner agents should be pushed diagonally away from center
        # Agent 0 (bottom-left) should have negative x and y components
        assert forces[0, 0] < 0
        assert forces[0, 1] < 0

        # Agent 3 (top-right) should have positive x and y components
        assert forces[3, 0] > 0
        assert forces[3, 1] > 0


class TestPairwiseForcesSpatialHashing:
    """Tests for spatial hashing pairwise force calculations."""

    def test_spatial_hashing_matches_brute(self):
        """Test that spatial hashing gives same results as brute force."""
        rng = np.random.RandomState(42)
        positions = rng.rand(50, 2) * 10.0

        a = agents.alloc(count=50, properties={'pos': positions})

        def simple_force(pi, pj):
            delta = pi - pj
            dist = np.linalg.norm(delta)
            if dist < 0.1:
                return np.zeros(2, dtype=np.float32)
            return delta / dist

        # Brute force
        forces_brute = agents.compute_pairwise_forces(
            a,
            radius=2.0,
            force_func=simple_force,
            use_spatial_hashing=False
        )

        # Spatial hashing
        forces_hash = agents.compute_pairwise_forces(
            a,
            radius=2.0,
            force_func=simple_force,
            use_spatial_hashing=True
        )

        # Should match closely
        assert np.allclose(forces_brute, forces_hash, atol=1e-5)

    def test_spatial_hashing_large_count(self):
        """Test spatial hashing with larger agent count."""
        rng = np.random.RandomState(42)
        positions = rng.rand(200, 2) * 50.0

        a = agents.alloc(count=200, properties={'pos': positions})

        def repulsion(pi, pj):
            delta = pi - pj
            dist = np.linalg.norm(delta)
            if dist < 0.1:
                return np.zeros(2, dtype=np.float32)
            return delta / (dist ** 2)

        # Should complete without error
        forces = agents.compute_pairwise_forces(
            a,
            radius=5.0,
            force_func=repulsion,
            use_spatial_hashing=True
        )

        assert forces.shape == (200, 2)
        # Forces should be non-zero for at least some agents
        assert np.any(forces != 0.0)


class TestPairwiseForcesEdgeCases:
    """Tests for edge cases in pairwise force calculations."""

    def test_single_agent(self):
        """Test pairwise forces with single agent."""
        a = agents.alloc(
            count=1,
            properties={'pos': np.array([[0.0, 0.0]])}
        )

        def dummy_force(pi, pj):
            return np.array([1.0, 0.0])

        forces = agents.compute_pairwise_forces(
            a,
            radius=10.0,
            force_func=dummy_force
        )

        # Single agent has no pairs, so no forces
        assert np.allclose(forces, 0.0)

    def test_zero_agents(self):
        """Test pairwise forces with no agents."""
        a = agents.alloc(count=0, properties={'pos': np.empty((0, 2))})

        def dummy_force(pi, pj):
            return np.array([1.0, 0.0])

        forces = agents.compute_pairwise_forces(
            a,
            radius=10.0,
            force_func=dummy_force
        )

        assert forces.shape == (0, 2)

    def test_agents_at_same_position(self):
        """Test forces when agents are at exactly same position."""
        a = agents.alloc(
            count=2,
            properties={
                'pos': np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
            }
        )

        # Force function that handles zero distance
        def safe_repulsion(pi, pj):
            delta = pi - pj
            dist = np.linalg.norm(delta)
            if dist < 0.01:  # Threshold for "same position"
                return np.zeros(2, dtype=np.float32)
            return delta / dist

        forces = agents.compute_pairwise_forces(
            a,
            radius=1.0,
            force_func=safe_repulsion
        )

        # Should be zero (agents at same position)
        assert np.allclose(forces, 0.0)


class TestFieldSampling:
    """Tests for sampling field values at agent positions."""

    def test_sample_constant_field(self):
        """Test sampling from constant field."""
        # Create constant field
        f = field.alloc((64, 64), fill_value=5.0)

        # Create agents at random positions
        rng = np.random.RandomState(42)
        positions = rng.rand(100, 2) * 63.0  # Positions in [0, 63)

        a = agents.alloc(count=100, properties={'pos': positions})

        # Sample field
        sampled = agents.sample_field(a, f, 'pos')

        # All values should be 5.0
        assert np.allclose(sampled, 5.0)

    def test_sample_gradient_field(self):
        """Test sampling from field with gradient."""
        # Create field with x-gradient (increases left to right)
        f = field.alloc((64, 64), fill_value=0.0)
        for i in range(64):
            f.data[:, i] = float(i)

        # Agents at known positions
        positions = np.array([
            [10.0, 32.0],  # x=10
            [30.0, 32.0],  # x=30
            [50.0, 32.0]   # x=50
        ], dtype=np.float32)

        a = agents.alloc(count=3, properties={'pos': positions})

        sampled = agents.sample_field(a, f, 'pos')

        # Values should match x-coordinates
        assert np.allclose(sampled[0], 10.0, atol=0.1)
        assert np.allclose(sampled[1], 30.0, atol=0.1)
        assert np.allclose(sampled[2], 50.0, atol=0.1)

    def test_sample_interpolation(self):
        """Test bilinear interpolation in sampling."""
        # Create simple 2x2 field
        f = Field2D(np.array([[0.0, 1.0],
                              [2.0, 3.0]], dtype=np.float32))

        # Sample at center (0.5, 0.5) - should be average of corners
        positions = np.array([[0.5, 0.5]], dtype=np.float32)
        a = agents.alloc(count=1, properties={'pos': positions})

        sampled = agents.sample_field(a, f, 'pos')

        # Average of 0, 1, 2, 3 is 1.5
        assert np.isclose(sampled[0], 1.5, atol=0.01)

    def test_sample_vector_field(self):
        """Test sampling from vector field."""
        # Create 2-channel velocity field
        f = field.alloc((64, 64), fill_value=0.0)
        f.data = np.zeros((64, 64, 2), dtype=np.float32)
        f.data[:, :, 0] = 1.0  # vx = 1.0
        f.data[:, :, 1] = 2.0  # vy = 2.0

        positions = np.random.rand(10, 2) * 63.0
        a = agents.alloc(count=10, properties={'pos': positions})

        sampled = agents.sample_field(a, f, 'pos')

        # All should be [1.0, 2.0]
        assert sampled.shape == (10, 2)
        assert np.allclose(sampled[:, 0], 1.0)
        assert np.allclose(sampled[:, 1], 2.0)

    def test_sample_boundary_clamping(self):
        """Test that positions outside field are clamped."""
        f = field.alloc((64, 64), fill_value=5.0)

        # Positions outside field bounds
        positions = np.array([
            [-10.0, 32.0],  # Left of field
            [100.0, 32.0],  # Right of field
            [32.0, -10.0],  # Below field
            [32.0, 100.0]   # Above field
        ], dtype=np.float32)

        a = agents.alloc(count=4, properties={'pos': positions})

        sampled = agents.sample_field(a, f, 'pos')

        # All should clamp to field value
        assert np.allclose(sampled, 5.0)

    def test_sample_respects_alive_mask(self):
        """Test that sampling only returns values for alive agents."""
        f = field.alloc((64, 64), fill_value=5.0)

        positions = np.random.rand(100, 2) * 63.0
        a = agents.alloc(count=100, properties={'pos': positions})

        # Kill half
        a.alive_mask[50:] = False

        sampled = agents.sample_field(a, f, 'pos')

        # Should only have 50 values
        assert len(sampled) == 50


class TestFieldAgentCoupling:
    """Tests for combined field-agent interactions."""

    def test_advect_agents_by_field(self):
        """Test moving agents by velocity field."""
        # Create simple velocity field (constant flow right)
        vel_field = field.alloc((64, 64), fill_value=0.0)
        vel_field.data = np.zeros((64, 64, 2), dtype=np.float32)
        vel_field.data[:, :, 0] = 2.0  # vx = 2.0
        vel_field.data[:, :, 1] = 0.0  # vy = 0.0

        # Create agents
        positions = np.array([[10.0, 32.0]], dtype=np.float32)
        a = agents.alloc(count=1, properties={'pos': positions})

        # Sample velocity
        vel = agents.sample_field(a, vel_field, 'pos')

        # Update position
        dt = 1.0
        new_pos = positions + vel * dt

        # Should move right by 2.0
        assert np.isclose(new_pos[0, 0], 12.0)
        assert np.isclose(new_pos[0, 1], 32.0)

    def test_temperature_affects_agent_speed(self):
        """Test agents reacting to field values."""
        # Create temperature field
        temp_field = field.alloc((64, 64), fill_value=0.0)

        # Hot on right, cold on left
        for i in range(64):
            temp_field.data[:, i] = float(i)

        # Agents at different x positions
        positions = np.array([
            [10.0, 32.0],
            [50.0, 32.0]
        ], dtype=np.float32)

        a = agents.alloc(count=2, properties={'pos': positions})

        # Sample temperature
        temps = agents.sample_field(a, temp_field, 'pos')

        # Temperature should increase with x
        assert temps[0] < temps[1]

        # Use temperature to scale velocity
        base_vel = np.array([[1.0, 0.0], [1.0, 0.0]])
        scaled_vel = base_vel * temps[:, np.newaxis]

        # Agent in hotter region should move faster
        assert np.linalg.norm(scaled_vel[1]) > np.linalg.norm(scaled_vel[0])


class TestDeterminism:
    """Tests for deterministic force calculations."""

    def test_pairwise_forces_deterministic(self):
        """Test that pairwise forces are deterministic."""
        rng = np.random.RandomState(42)
        pos1 = rng.rand(50, 2) * 10.0

        rng = np.random.RandomState(42)
        pos2 = rng.rand(50, 2) * 10.0

        a1 = agents.alloc(count=50, properties={'pos': pos1})
        a2 = agents.alloc(count=50, properties={'pos': pos2})

        def force(pi, pj):
            delta = pi - pj
            dist = np.linalg.norm(delta)
            if dist < 0.1:
                return np.zeros(2)
            return delta / dist

        f1 = agents.compute_pairwise_forces(a1, radius=2.0, force_func=force)
        f2 = agents.compute_pairwise_forces(a2, radius=2.0, force_func=force)

        assert np.allclose(f1, f2)

    def test_field_sampling_deterministic(self):
        """Test that field sampling is deterministic."""
        rng1 = np.random.RandomState(42)
        f1 = field.random((64, 64), seed=42)
        pos1 = rng1.rand(100, 2) * 63.0

        rng2 = np.random.RandomState(42)
        f2 = field.random((64, 64), seed=42)
        pos2 = rng2.rand(100, 2) * 63.0

        a1 = agents.alloc(count=100, properties={'pos': pos1})
        a2 = agents.alloc(count=100, properties={'pos': pos2})

        s1 = agents.sample_field(a1, f1, 'pos')
        s2 = agents.sample_field(a2, f2, 'pos')

        assert np.allclose(s1, s2)
