"""Vortex and magic spell particle effects.

Demonstrates vortex forces, attractors, rotation visualization, and trail effects.
"""

import numpy as np
from morphogen.stdlib.agents import agents, particle_behaviors
from morphogen.stdlib.visual import visual


def vortex_magic_demo():
    """Run vortex magic particle effects demo."""

    print("Vortex Magic Particle Effects Demo")
    print("=" * 50)

    # Simulation parameters
    width, height = 512, 512
    dt = 0.3

    # Vortex centers
    vortex1_center = np.array([width * 0.3, height * 0.5])
    vortex2_center = np.array([width * 0.7, height * 0.5])

    # Create initial particles in circle around each vortex
    count_per_vortex = 150

    # Vortex 1 particles (magenta)
    angles1 = np.linspace(0, 2 * np.pi, count_per_vortex, endpoint=False)
    radius1 = 100.0
    positions1 = vortex1_center + np.stack([
        radius1 * np.cos(angles1),
        radius1 * np.sin(angles1)
    ], axis=1)

    particles1 = agents.alloc(
        count=count_per_vortex,
        properties={
            'pos': positions1.astype(np.float32),
            'vel': np.zeros((count_per_vortex, 2), dtype=np.float32),
            'age': np.zeros(count_per_vortex, dtype=np.float32),
            'lifetime': np.full(count_per_vortex, np.inf, dtype=np.float32),
            'vortex_id': np.zeros(count_per_vortex, dtype=np.float32),  # 0 for vortex 1
            'size': np.ones(count_per_vortex, dtype=np.float32) * 2.0
        }
    )

    # Vortex 2 particles (cyan)
    angles2 = np.linspace(0, 2 * np.pi, count_per_vortex, endpoint=False)
    radius2 = 100.0
    positions2 = vortex2_center + np.stack([
        radius2 * np.cos(angles2),
        radius2 * np.sin(angles2)
    ], axis=1)

    particles2 = agents.alloc(
        count=count_per_vortex,
        properties={
            'pos': positions2.astype(np.float32),
            'vel': np.zeros((count_per_vortex, 2), dtype=np.float32),
            'age': np.zeros(count_per_vortex, dtype=np.float32),
            'lifetime': np.full(count_per_vortex, np.inf, dtype=np.float32),
            'vortex_id': np.ones(count_per_vortex, dtype=np.float32),  # 1 for vortex 2
            'size': np.ones(count_per_vortex, dtype=np.float32) * 2.0
        }
    )

    # Merge both particle sets
    all_particles = agents.merge([particles1, particles2])

    # Create vortex forces
    vortex1_force = particle_behaviors.vortex(vortex1_center, strength=15.0)
    vortex2_force = particle_behaviors.vortex(vortex2_center, strength=-15.0)  # Counter-rotating

    def generate_frame():
        nonlocal all_particles

        frame_count = 0
        max_frames = 400

        # Move vortex centers in circle (orbiting effect)
        orbit_radius = 50.0
        orbit_speed = 0.02

        while frame_count < max_frames:
            # Update vortex positions (orbit around center)
            angle = frame_count * orbit_speed
            v1_center = np.array([width / 2, height / 2]) + np.array([
                orbit_radius * np.cos(angle),
                orbit_radius * np.sin(angle)
            ])
            v2_center = np.array([width / 2, height / 2]) + np.array([
                orbit_radius * np.cos(angle + np.pi),
                orbit_radius * np.sin(angle + np.pi)
            ])

            # Update vortex forces
            vortex1 = particle_behaviors.vortex(v1_center, strength=15.0)
            vortex2 = particle_behaviors.vortex(v2_center, strength=-15.0)

            # Apply vortex forces
            if all_particles.alive_count > 0:
                # Apply both vortices
                all_particles = agents.apply_force(all_particles, force=vortex1, dt=dt)
                all_particles = agents.apply_force(all_particles, force=vortex2, dt=dt)

                # Add gentle attraction to opposite vortex
                vortex_ids = all_particles.get('vortex_id')

                # Particles from vortex 1 attracted to vortex 2
                mask1 = vortex_ids < 0.5
                if np.any(mask1):
                    attractor = particle_behaviors.attractor(v2_center, strength=1.0)
                    force = attractor(all_particles)
                    force[~mask1] = 0.0  # Zero out force for other particles
                    all_particles = agents.apply_force(
                        all_particles,
                        force=force,
                        dt=dt
                    )

                # Particles from vortex 2 attracted to vortex 1
                mask2 = vortex_ids >= 0.5
                if np.any(mask2):
                    attractor = particle_behaviors.attractor(v1_center, strength=1.0)
                    force = attractor(all_particles)
                    force[~mask2] = 0.0
                    all_particles = agents.apply_force(
                        all_particles,
                        force=force,
                        dt=dt
                    )

                # Slight drag
                all_particles = agents.apply_force(
                    all_particles,
                    force=particle_behaviors.drag(coefficient=0.01),
                    dt=dt
                )

                # Update positions
                all_particles = agents.integrate(all_particles, dt=dt)

                # Update trails
                if frame_count % 2 == 0:
                    all_particles = agents.update_trail(all_particles, trail_length=25)

            # Render
            if all_particles.alive_count > 0:
                # Set colors based on vortex_id
                vis = visual.agents(
                    all_particles,
                    width=width,
                    height=height,
                    color_property='vortex_id',
                    size_property='size',
                    rotation_property='vel',  # Show velocity direction
                    palette='coolwarm',  # Magenta to cyan
                    background=(0.0, 0.0, 0.0),
                    alpha=0.8,
                    blend_mode='additive',
                    trail=True,
                    trail_length=25,
                    trail_alpha=0.3
                )
            else:
                vis = visual.layer(width=width, height=height, background=(0.0, 0.0, 0.0))

            frame_count += 1

            if frame_count % 30 == 0:
                print(f"Frame {frame_count}, Particles alive: {all_particles.alive_count}")

            yield vis

    # Create generator
    gen = generate_frame()

    # Export to video
    print("\nExporting vortex magic animation...")
    visual.video(
        lambda: next(gen),
        path="/tmp/vortex_magic.mp4",
        fps=30,
        max_frames=400
    )

    print(f"\nVortex magic animation saved to /tmp/vortex_magic.mp4")


if __name__ == "__main__":
    vortex_magic_demo()
