"""Fire and smoke particle simulation.

Demonstrates particle emission from a source, upward motion with turbulence,
and color/alpha changes over particle lifetime.
"""

import numpy as np
from morphogen.stdlib.agents import agents, particle_behaviors
from morphogen.stdlib.visual import visual


def fire_particles_demo():
    """Run fire particle effects demo."""

    print("Fire Particle Effects Demo")
    print("=" * 50)

    # Simulation parameters
    width, height = 512, 512
    dt = 0.5

    # Fire source position (bottom center)
    fire_source = np.array([width / 2, 50.0])

    # Create initial empty particle system
    all_particles = agents.alloc(count=0, properties={
        'pos': np.empty((0, 2)),
        'vel': np.empty((0, 2)),
        'age': np.empty(0),
        'lifetime': np.empty(0)
    })

    def generate_frame():
        nonlocal all_particles

        frame_count = 0
        max_frames = 300

        while frame_count < max_frames:
            # Emit new fire particles every frame
            new_particles = agents.emit(
                count=20,  # Emit 20 particles per frame
                position=fire_source,
                emission_shape="circle",
                emission_radius=15.0,
                velocity=lambda n: np.stack([
                    np.random.randn(n) * 0.5,  # Slight horizontal spread
                    np.random.uniform(2.0, 4.0, n)  # Upward velocity
                ], axis=1),
                lifetime=(40.0, 80.0),  # Variable lifetime
                properties={
                    'temperature': np.random.uniform(0.7, 1.0, 20),  # For color mapping
                    'size': np.random.uniform(1.5, 3.0, 20)
                },
                seed=None
            )

            # Merge with existing particles
            if all_particles.alive_count > 0:
                all_particles = agents.merge([all_particles, new_particles])
            else:
                all_particles = new_particles

            # Apply forces
            if all_particles.alive_count > 0:
                # Buoyancy (upward force)
                all_particles = agents.apply_force(
                    all_particles,
                    force=np.array([0.0, 3.0]),
                    dt=dt
                )

                # Turbulence
                all_particles = agents.apply_force(
                    all_particles,
                    force=particle_behaviors.turbulence(scale=1.5),
                    dt=dt
                )

                # Air resistance
                all_particles = agents.apply_force(
                    all_particles,
                    force=particle_behaviors.drag(coefficient=0.03),
                    dt=dt
                )

                # Update positions
                all_particles = agents.integrate(all_particles, dt=dt)

                # Age particles
                all_particles = agents.age_particles(all_particles, dt=dt)

                # Reduce temperature over time (cooling)
                ages = all_particles.get('age')
                lifetimes = all_particles.get('lifetime')
                age_ratio = np.clip(ages / lifetimes, 0.0, 1.0)

                temps = all_particles.get('temperature')
                cooled_temps = temps * (1.0 - age_ratio * 0.7)  # Cool down over lifetime
                all_particles.set('temperature', cooled_temps)

            # Render
            if all_particles.alive_count > 0:
                # Calculate alpha (fade out as particles age)
                alphas = agents.get_particle_alpha(
                    all_particles,
                    fade_in=0.05,
                    fade_out=0.4
                )

                # Store alpha in particles for rendering
                all_particles.properties['alpha'] = np.zeros(all_particles.count, dtype=np.float32)
                all_particles.properties['alpha'][all_particles.alive_mask] = alphas

                # Render with fire palette (temperature-based coloring)
                vis = visual.agents(
                    all_particles,
                    width=width,
                    height=height,
                    color_property='temperature',
                    size_property='size',
                    alpha_property='alpha',
                    palette='fire',
                    background=(0.0, 0.0, 0.0),
                    blend_mode='additive'
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
    print("\nExporting fire animation...")
    visual.video(
        lambda: next(gen),
        path="/tmp/fire_particles.mp4",
        fps=30,
        max_frames=300
    )

    print(f"\nFire animation saved to /tmp/fire_particles.mp4")


if __name__ == "__main__":
    fire_particles_demo()
