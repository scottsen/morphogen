"""Integration tests for agent operations with runtime."""

import pytest
import numpy as np
from morphogen.runtime.runtime import Runtime
from morphogen.stdlib.agents import agents
from morphogen.stdlib.field import field


class TestAgentRuntimeIntegration:
    """Tests for agent integration with runtime."""

    def test_agents_namespace_available(self):
        """Test that agents namespace is available in runtime."""
        runtime = Runtime()
        assert runtime.context.has_variable('agents')
        agents_obj = runtime.context.get_variable('agents')
        assert hasattr(agents_obj, 'alloc')
        assert hasattr(agents_obj, 'map')
        assert hasattr(agents_obj, 'filter')
        assert hasattr(agents_obj, 'reduce')

    def test_agents_alloc_from_runtime(self):
        """Test allocating agents through runtime."""
        runtime = Runtime()
        agents_ns = runtime.context.get_variable('agents')

        # Allocate agents
        a = agents_ns.alloc(
            count=100,
            properties={
                'pos': np.zeros((100, 2)),
                'vel': np.ones((100, 2))
            }
        )

        assert a.count == 100
        assert 'pos' in a.properties
        assert 'vel' in a.properties

    def test_agents_operations_from_runtime(self):
        """Test agent operations through runtime."""
        runtime = Runtime()
        agents_ns = runtime.context.get_variable('agents')

        # Create agents
        a = agents_ns.alloc(
            count=50,
            properties={'id': np.arange(50, dtype=np.float32)}
        )

        # Map operation
        doubled = agents_ns.map(a, 'id', lambda x: x * 2)
        assert np.allclose(doubled, np.arange(50) * 2)

        # Filter operation
        b = agents_ns.filter(a, 'id', lambda x: x < 25)
        assert b.alive_count == 25

        # Reduce operation
        total = agents_ns.reduce(a, 'id', operation='sum')
        expected_sum = sum(range(50))
        assert np.isclose(total, expected_sum)

    def test_agents_and_fields_together(self):
        """Test using agents with fields together."""
        runtime = Runtime()
        agents_ns = runtime.context.get_variable('agents')
        field_ns = runtime.context.get_variable('field')

        # Create field
        f = field_ns.alloc((64, 64), fill_value=5.0)

        # Create agents
        positions = np.random.rand(100, 2) * 63.0
        a = agents_ns.alloc(count=100, properties={'pos': positions})

        # Sample field
        sampled = agents_ns.sample_field(a, f, 'pos')

        assert len(sampled) == 100
        assert np.allclose(sampled, 5.0)

    def test_pairwise_forces_from_runtime(self):
        """Test pairwise force calculation through runtime."""
        runtime = Runtime()
        agents_ns = runtime.context.get_variable('agents')

        # Create agents
        a = agents_ns.alloc(
            count=2,
            properties={
                'pos': np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
            }
        )

        # Simple repulsion
        def repulsion(pi, pj):
            delta = pi - pj
            dist = np.linalg.norm(delta)
            if dist < 0.1:
                return np.zeros(2, dtype=np.float32)
            return delta / dist

        forces = agents_ns.compute_pairwise_forces(
            a,
            radius=2.0,
            force_func=repulsion,
            use_spatial_hashing=False
        )

        assert forces.shape == (2, 2)
        # Agent 0 should be pushed left
        assert forces[0, 0] < 0
        # Agent 1 should be pushed right
        assert forces[1, 0] > 0


class TestAgentSimulation:
    """Tests for complete agent-based simulations."""

    def test_simple_particle_simulation(self):
        """Test simple particle simulation with position update."""
        runtime = Runtime()
        agents_ns = runtime.context.get_variable('agents')

        # Create particles
        rng = np.random.RandomState(42)
        a = agents_ns.alloc(
            count=100,
            properties={
                'pos': rng.rand(100, 2) * 10.0,
                'vel': rng.rand(100, 2) - 0.5
            }
        )

        # Simulate one timestep
        dt = 0.1
        new_pos = a.get('pos') + a.get('vel') * dt
        a = a.update('pos', new_pos)

        # Verify positions changed
        assert not np.allclose(a.get('pos'), rng.rand(100, 2) * 10.0)

    def test_filter_out_of_bounds_particles(self):
        """Test filtering particles that leave bounds."""
        runtime = Runtime()
        agents_ns = runtime.context.get_variable('agents')

        # Create particles
        a = agents_ns.alloc(
            count=100,
            properties={
                'pos': np.random.rand(100, 2) * 20.0 - 10.0  # Range [-10, 10]
            }
        )

        # Filter to keep only particles in [0, 10]
        def in_bounds(p):
            return (p[:, 0] >= 0) & (p[:, 0] <= 10) & (p[:, 1] >= 0) & (p[:, 1] <= 10)

        b = agents_ns.filter(a, 'pos', in_bounds)

        # Some should be filtered out
        assert b.alive_count < 100
        assert b.alive_count > 0

        # All remaining should be in bounds
        pos = b.get('pos')
        assert np.all(pos[:, 0] >= 0)
        assert np.all(pos[:, 0] <= 10)
        assert np.all(pos[:, 1] >= 0)
        assert np.all(pos[:, 1] <= 10)

    def test_center_of_mass_calculation(self):
        """Test calculating center of mass of agents."""
        runtime = Runtime()
        agents_ns = runtime.context.get_variable('agents')

        # Create symmetric particle distribution
        a = agents_ns.alloc(
            count=4,
            properties={
                'pos': np.array([
                    [0.0, 0.0],
                    [2.0, 0.0],
                    [0.0, 2.0],
                    [2.0, 2.0]
                ], dtype=np.float32),
                'mass': np.ones(4, dtype=np.float32)
            }
        )

        # Calculate center of mass
        positions = a.get('pos')
        masses = a.get('mass')
        total_mass = np.sum(masses)
        com = np.sum(positions * masses[:, np.newaxis], axis=0) / total_mass

        # Should be at (1, 1)
        assert np.allclose(com, [1.0, 1.0])


class TestAgentWithFieldInteraction:
    """Tests for agent-field coupling scenarios."""

    def test_particles_sample_temperature_field(self):
        """Test particles sampling from temperature field."""
        runtime = Runtime()
        agents_ns = runtime.context.get_variable('agents')
        field_ns = runtime.context.get_variable('field')

        # Create temperature gradient field
        temp = field_ns.alloc((64, 64), fill_value=0.0)
        for i in range(64):
            temp.data[:, i] = float(i) * 2.0  # Temperature increases with x

        # Create particles
        positions = np.array([
            [10.0, 32.0],
            [30.0, 32.0],
            [50.0, 32.0]
        ], dtype=np.float32)

        a = agents_ns.alloc(count=3, properties={'pos': positions})

        # Sample temperature
        temps = agents_ns.sample_field(a, temp, 'pos')

        # Verify temperature increases with x position
        assert temps[0] < temps[1] < temps[2]
        assert np.isclose(temps[0], 20.0, atol=1.0)
        assert np.isclose(temps[1], 60.0, atol=1.0)
        assert np.isclose(temps[2], 100.0, atol=1.0)

    def test_particles_advected_by_velocity_field(self):
        """Test particles being moved by velocity field."""
        runtime = Runtime()
        agents_ns = runtime.context.get_variable('agents')
        field_ns = runtime.context.get_variable('field')

        # Create constant velocity field (flow to the right)
        vel_field = field_ns.alloc((64, 64), fill_value=0.0)
        vel_field.data = np.zeros((64, 64, 2), dtype=np.float32)
        vel_field.data[:, :, 0] = 2.0  # vx = 2.0
        vel_field.data[:, :, 1] = 0.0  # vy = 0.0

        # Create particles
        start_pos = np.array([[10.0, 32.0]], dtype=np.float32)
        a = agents_ns.alloc(count=1, properties={'pos': start_pos.copy()})

        # Sample velocity and update position
        vel = agents_ns.sample_field(a, vel_field, 'pos')
        dt = 1.0
        new_pos = start_pos + vel * dt

        # Should move right by 2.0
        assert np.isclose(new_pos[0, 0], 12.0)
        assert np.isclose(new_pos[0, 1], 32.0)


class TestAgentPerformance:
    """Tests for agent performance characteristics."""

    def test_large_agent_allocation(self):
        """Test that we can allocate many agents."""
        runtime = Runtime()
        agents_ns = runtime.context.get_variable('agents')

        # Allocate 10,000 agents
        a = agents_ns.alloc(
            count=10000,
            properties={
                'pos': np.random.rand(10000, 2),
                'vel': np.zeros((10000, 2))
            }
        )

        assert a.count == 10000
        assert a.alive_count == 10000

    def test_spatial_hashing_performance(self):
        """Test spatial hashing with moderate agent count."""
        runtime = Runtime()
        agents_ns = runtime.context.get_variable('agents')

        # Create 500 agents
        rng = np.random.RandomState(42)
        a = agents_ns.alloc(
            count=500,
            properties={'pos': rng.rand(500, 2) * 50.0}
        )

        # Compute forces with spatial hashing
        def simple_force(pi, pj):
            delta = pi - pj
            dist = np.linalg.norm(delta)
            if dist < 0.1:
                return np.zeros(2, dtype=np.float32)
            return delta / dist

        forces = agents_ns.compute_pairwise_forces(
            a,
            radius=5.0,
            force_func=simple_force,
            use_spatial_hashing=True
        )

        assert forces.shape == (500, 2)
