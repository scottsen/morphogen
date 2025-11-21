"""Unit tests for particle effects / VFX extensions."""

import pytest
import numpy as np
from morphogen.stdlib.agents import agents, particle_behaviors, Agents
from morphogen.stdlib.visual import visual, Visual


class TestParticleEmission:
    """Tests for particle emission system."""

    def test_emit_basic(self):
        """Test basic particle emission."""
        particles = agents.emit(
            count=100,
            position=np.array([64.0, 64.0]),
            lifetime=50.0
        )

        assert particles.count == 100
        assert particles.alive_count == 100
        assert 'pos' in particles.properties
        assert 'vel' in particles.properties
        assert 'age' in particles.properties
        assert 'lifetime' in particles.properties

    def test_emit_with_velocity(self):
        """Test emission with explicit velocity."""
        vel = np.array([1.0, 2.0])
        particles = agents.emit(
            count=50,
            position=np.array([32.0, 32.0]),
            velocity=vel
        )

        velocities = particles.get('vel')
        assert np.allclose(velocities[:, 0], 1.0)
        assert np.allclose(velocities[:, 1], 2.0)

    def test_emit_circle_shape(self):
        """Test circular emission pattern."""
        center = np.array([64.0, 64.0])
        radius = 10.0

        particles = agents.emit(
            count=100,
            position=center,
            emission_shape="circle",
            emission_radius=radius,
            seed=42
        )

        positions = particles.get('pos')

        # All particles should be within radius of center
        distances = np.linalg.norm(positions - center, axis=1)
        assert np.all(distances <= radius)

    def test_emit_with_lifetime_range(self):
        """Test emission with lifetime range."""
        particles = agents.emit(
            count=100,
            position=np.array([64.0, 64.0]),
            lifetime=(10.0, 50.0),
            seed=42
        )

        lifetimes = particles.get('lifetime')
        assert np.all(lifetimes >= 10.0)
        assert np.all(lifetimes <= 50.0)

    def test_emit_with_properties(self):
        """Test emission with custom properties."""
        particles = agents.emit(
            count=50,
            position=np.array([32.0, 32.0]),
            properties={
                'color': (1.0, 0.5, 0.0),
                'temperature': np.random.rand(50)
            }
        )

        assert 'color' in particles.properties
        assert 'temperature' in particles.properties

        colors = particles.get('color')
        assert colors.shape == (50, 3)  # 3-element RGB color per particle
        assert np.all(colors[:, 0] == 1.0)  # Red channel
        assert np.all(colors[:, 1] == 0.5)  # Green channel
        assert np.all(colors[:, 2] == 0.0)  # Blue channel

    def test_emit_callable_position(self):
        """Test emission with position callback."""
        def position_func(count):
            return np.random.rand(count, 2) * 100.0

        particles = agents.emit(
            count=100,
            position=position_func
        )

        positions = particles.get('pos')
        assert positions.shape == (100, 2)

    def test_emit_callable_velocity(self):
        """Test emission with velocity callback."""
        def velocity_func(count):
            return np.random.randn(count, 2) * 5.0

        particles = agents.emit(
            count=100,
            position=np.array([64.0, 64.0]),
            velocity=velocity_func
        )

        velocities = particles.get('vel')
        assert velocities.shape == (100, 2)


class TestParticleLifetime:
    """Tests for particle lifetime and aging."""

    def test_age_particles(self):
        """Test particle aging."""
        particles = agents.emit(
            count=100,
            position=np.array([64.0, 64.0]),
            lifetime=50.0
        )

        # Age by 10 steps
        particles = agents.age_particles(particles, dt=10.0)

        ages = particles.get('age')
        assert np.allclose(ages, 10.0)
        assert particles.alive_count == 100  # All still alive

    def test_particles_die_after_lifetime(self):
        """Test that particles die when age exceeds lifetime."""
        particles = agents.emit(
            count=100,
            position=np.array([64.0, 64.0]),
            lifetime=20.0
        )

        # Age beyond lifetime
        particles = agents.age_particles(particles, dt=25.0)

        assert particles.alive_count == 0  # All dead

    def test_partial_particle_death(self):
        """Test partial particle death with variable lifetimes."""
        lifetimes = np.array([10.0] * 50 + [30.0] * 50)

        particles = agents.alloc(
            count=100,
            properties={
                'pos': np.random.rand(100, 2),
                'vel': np.zeros((100, 2)),
                'age': np.zeros(100),
                'lifetime': lifetimes
            }
        )

        # Age to 20 (first 50 should die)
        particles = agents.age_particles(particles, dt=20.0)

        assert particles.alive_count == 50

    def test_get_particle_alpha_fade_out(self):
        """Test alpha calculation with fade out."""
        particles = agents.emit(
            count=100,
            position=np.array([64.0, 64.0]),
            lifetime=100.0
        )

        # Age to 90% of lifetime
        particles = agents.age_particles(particles, dt=90.0)

        # Get alpha with 20% fade out
        alphas = agents.get_particle_alpha(particles, fade_out=0.2)

        # Should be fading (alpha < 1.0)
        assert np.all(alphas < 1.0)
        assert np.all(alphas > 0.0)

    def test_get_particle_alpha_fade_in(self):
        """Test alpha calculation with fade in."""
        particles = agents.emit(
            count=100,
            position=np.array([64.0, 64.0]),
            lifetime=100.0
        )

        # Age to 5% of lifetime
        particles = agents.age_particles(particles, dt=5.0)

        # Get alpha with 10% fade in
        alphas = agents.get_particle_alpha(particles, fade_in=0.1)

        # Should be fading in (alpha < 1.0)
        assert np.all(alphas < 1.0)
        assert np.all(alphas > 0.0)


class TestParticleForces:
    """Tests for particle force application."""

    def test_apply_force_uniform(self):
        """Test applying uniform force."""
        particles = agents.emit(
            count=100,
            position=np.array([64.0, 64.0]),
            velocity=np.zeros((100, 2))
        )

        # Apply gravity
        particles = agents.apply_force(
            particles,
            force=np.array([0.0, -9.8]),
            dt=1.0
        )

        velocities = particles.get('vel')
        assert np.allclose(velocities[:, 0], 0.0)
        assert np.allclose(velocities[:, 1], -9.8)

    def test_apply_force_callable(self):
        """Test applying force from callable."""
        particles = agents.emit(
            count=100,
            position=np.random.rand(100, 2) * 100.0,
            velocity=np.zeros((100, 2))
        )

        # Drag force
        def drag_force(a):
            return -0.1 * a.get('vel')

        particles = agents.apply_force(
            particles,
            force=drag_force,
            dt=1.0
        )

        # Velocities should remain zero (no initial velocity)
        velocities = particles.get('vel')
        assert np.allclose(velocities, 0.0)

    def test_integrate_position(self):
        """Test position integration."""
        particles = agents.alloc(
            count=100,
            properties={
                'pos': np.zeros((100, 2)),
                'vel': np.ones((100, 2)) * 5.0
            }
        )

        # Integrate for 2 time steps
        particles = agents.integrate(particles, dt=2.0)

        positions = particles.get('pos')
        assert np.allclose(positions, 10.0)


class TestParticleBehaviors:
    """Tests for pre-built particle behaviors."""

    def test_vortex_force(self):
        """Test vortex force field."""
        center = np.array([50.0, 50.0])
        vortex = particle_behaviors.vortex(center, strength=10.0)

        particles = agents.emit(
            count=100,
            position=np.random.rand(100, 2) * 100.0
        )

        force = vortex(particles)

        assert force.shape == (100, 2)
        assert force.dtype == np.float32

    def test_attractor_force(self):
        """Test attractor force."""
        center = np.array([50.0, 50.0])
        attractor = particle_behaviors.attractor(center, strength=5.0)

        particles = agents.emit(
            count=100,
            position=np.random.rand(100, 2) * 100.0
        )

        force = attractor(particles)

        # All forces should point toward center
        positions = particles.get('pos')
        to_center = center - positions

        # Check that force and to_center have same direction (dot product > 0)
        dots = np.sum(force * to_center, axis=1)
        assert np.all(dots > 0)

    def test_repulsor_force(self):
        """Test repulsor force."""
        center = np.array([50.0, 50.0])
        repulsor = particle_behaviors.repulsor(center, strength=10.0, radius=30.0)

        particles = agents.emit(
            count=100,
            position=np.random.rand(100, 2) * 100.0
        )

        force = repulsor(particles)

        # Particles far from center should have zero force
        positions = particles.get('pos')
        distances = np.linalg.norm(positions - center, axis=1)
        far_mask = distances > 30.0

        assert np.allclose(force[far_mask], 0.0)

    def test_drag_force(self):
        """Test drag force."""
        drag = particle_behaviors.drag(coefficient=0.1)

        particles = agents.alloc(
            count=100,
            properties={
                'pos': np.zeros((100, 2)),
                'vel': np.ones((100, 2)) * 10.0
            }
        )

        force = drag(particles)

        # Drag should oppose velocity
        velocities = particles.get('vel')
        assert np.all(force * velocities < 0)

    def test_turbulence_force(self):
        """Test turbulence force."""
        turbulence = particle_behaviors.turbulence(scale=5.0, seed=42)

        particles = agents.emit(
            count=100,
            position=np.array([50.0, 50.0])
        )

        force = turbulence(particles)

        assert force.shape == (100, 2)
        # Should be random, not all zeros
        assert not np.allclose(force, 0.0)


class TestTrailManagement:
    """Tests for particle trail history."""

    def test_update_trail_initialization(self):
        """Test trail history initialization."""
        particles = agents.emit(
            count=50,
            position=np.array([64.0, 64.0])
        )

        particles = agents.update_trail(particles, trail_length=10)

        assert 'trail_history' in particles.properties
        trail = particles.get_all('trail_history')
        assert trail.shape == (50, 10, 2)

    def test_update_trail_records_positions(self):
        """Test that trail records positions correctly."""
        particles = agents.alloc(
            count=10,
            properties={
                'pos': np.array([[float(i), float(i)] for i in range(10)]),
                'vel': np.zeros((10, 2))
            }
        )

        # Update trail
        particles = agents.update_trail(particles, trail_length=3)

        # Move particles
        particles = particles.update('pos', particles.get('pos') + 10.0)

        # Update trail again
        particles = agents.update_trail(particles, trail_length=3)

        trail = particles.get_all('trail_history')

        # Last position in trail should be current position
        current_pos = particles.get('pos')
        last_trail_pos = trail[:, -1, :]

        assert np.allclose(last_trail_pos, current_pos)


class TestParticleMerging:
    """Tests for merging particle collections."""

    def test_merge_two_collections(self):
        """Test merging two particle collections."""
        p1 = agents.emit(count=50, position=np.array([10.0, 10.0]))
        p2 = agents.emit(count=30, position=np.array([90.0, 90.0]))

        merged = agents.merge([p1, p2])

        assert merged.count == 80
        assert merged.alive_count == 80

    def test_merge_preserves_properties(self):
        """Test that merge preserves all properties."""
        p1 = agents.emit(
            count=50,
            position=np.array([10.0, 10.0]),
            properties={'color': 1.0}
        )
        p2 = agents.emit(
            count=30,
            position=np.array([90.0, 90.0]),
            properties={'color': 2.0}
        )

        merged = agents.merge([p1, p2])

        assert 'color' in merged.properties

        colors = merged.get('color')
        # First 50 should be 1.0, next 30 should be 2.0
        assert np.allclose(colors[:50], 1.0)
        assert np.allclose(colors[50:], 2.0)

    def test_merge_empty_list_error(self):
        """Test that merging empty list raises error."""
        with pytest.raises(ValueError):
            agents.merge([])

    def test_merge_single_collection(self):
        """Test merging single collection returns copy."""
        p1 = agents.emit(count=50, position=np.array([10.0, 10.0]))
        merged = agents.merge([p1])

        assert merged.count == 50
        assert merged.alive_count == 50


class TestVisualParticleRendering:
    """Tests for particle visual rendering."""

    def test_render_with_alpha(self):
        """Test rendering particles with alpha."""
        particles = agents.emit(count=100, position=np.array([256.0, 256.0]))

        # Add alpha property
        particles.properties['alpha'] = np.ones(100, dtype=np.float32) * 0.5

        vis = visual.agents(
            particles,
            width=512,
            height=512,
            alpha_property='alpha'
        )

        assert isinstance(vis, Visual)
        assert vis.width == 512
        assert vis.height == 512

    def test_render_with_rotation(self):
        """Test rendering particles with rotation."""
        particles = agents.emit(count=100, position=np.array([256.0, 256.0]))

        # Add velocity for rotation visualization
        velocities = np.random.randn(100, 2)
        particles.properties['vel'] = velocities.astype(np.float32)

        vis = visual.agents(
            particles,
            width=512,
            height=512,
            rotation_property='vel'
        )

        assert isinstance(vis, Visual)

    def test_render_additive_blending(self):
        """Test rendering with additive blending."""
        particles = agents.emit(count=100, position=np.array([256.0, 256.0]))

        vis = visual.agents(
            particles,
            width=512,
            height=512,
            blend_mode='additive'
        )

        assert isinstance(vis, Visual)


class TestDeterminism:
    """Tests for deterministic particle operations."""

    def test_emit_deterministic(self):
        """Test that emission with seed is deterministic."""
        p1 = agents.emit(
            count=100,
            position=np.array([64.0, 64.0]),
            emission_shape="circle",
            emission_radius=10.0,
            seed=42
        )

        p2 = agents.emit(
            count=100,
            position=np.array([64.0, 64.0]),
            emission_shape="circle",
            emission_radius=10.0,
            seed=42
        )

        assert np.allclose(p1.get('pos'), p2.get('pos'))
        assert np.allclose(p1.get('vel'), p2.get('vel'))

    def test_turbulence_deterministic(self):
        """Test that turbulence with seed is deterministic."""
        particles = agents.emit(count=100, position=np.array([64.0, 64.0]))

        turb1 = particle_behaviors.turbulence(scale=5.0, seed=42)
        turb2 = particle_behaviors.turbulence(scale=5.0, seed=42)

        f1 = turb1(particles)
        f2 = turb2(particles)

        assert np.allclose(f1, f2)
