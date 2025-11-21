"""
Interactive smoke simulation using Navier-Stokes equations.

This demonstrates:
- Vector field advection (velocity self-advection)
- Scalar field advection (density advected by velocity)
- Diffusion with iterative Jacobi solver
- Pressure projection for incompressible flow
- Real-time interactive visualization

Controls:
- SPACE: Pause/Resume
- RIGHT ARROW: Step forward (when paused)
- UP/DOWN ARROWS: Speed control
- Q or ESC: Quit
"""

import numpy as np
from morphogen.stdlib.field import field, Field2D
from morphogen.stdlib.visual import visual


def create_smoke_simulation():
    """Generate frames for interactive smoke simulation."""
    # Grid parameters
    grid_size = (128, 128)
    dt = 0.1

    # Physics parameters
    viscosity = 0.0001  # Very low viscosity for fluid-like behavior
    density_diffusion = 0.00001
    advection_iterations = 20
    projection_iterations = 40

    # Initialize fields
    # Velocity is a 2D vector field (each point has (vx, vy))
    velocity_data = np.zeros((*grid_size, 2), dtype=np.float32)

    # Add initial swirl in the center
    h, w = grid_size
    y, x = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx**2 + dy**2) + 1e-6

    # Circular velocity field (vortex)
    velocity_data[:, :, 0] = -dy / r * np.exp(-r / 20) * 5.0  # vx
    velocity_data[:, :, 1] = dx / r * np.exp(-r / 20) * 5.0   # vy

    velocity = Field2D(velocity_data, dx=1.0, dy=1.0)

    # Density field (smoke concentration)
    density = field.random(grid_size, seed=42, low=0.0, high=1.0)

    # Add concentrated smoke in patches
    for i in range(5):
        y = np.random.randint(h // 4, 3 * h // 4)
        x = np.random.randint(w // 4, 3 * w // 4)
        radius = 10
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - x)**2 + (yy - y)**2 <= radius**2
        density.data[mask] = 1.0

    frame_count = 0

    while True:
        # STEP 1: Advection (smoke and velocity move with flow)
        velocity = field.advect(velocity, velocity, dt)
        density = field.advect(density, velocity, dt)

        # STEP 2: Diffusion (smoke spreads, velocity dissipates)
        if viscosity > 0:
            velocity = field.diffuse(velocity, rate=viscosity, dt=dt,
                                    iterations=advection_iterations)

        density = field.diffuse(density, rate=density_diffusion, dt=dt,
                               iterations=advection_iterations)

        # STEP 3: Projection (enforce incompressibility)
        velocity = field.project(velocity, iterations=projection_iterations)

        # STEP 4: Boundary conditions
        velocity = field.boundary(velocity, spec="periodic")
        density = field.boundary(density, spec="periodic")

        # STEP 5: Add continuous smoke source in center-bottom
        # (Simple addition - in DSL this would be cleaner)
        source_region = density.data.copy()
        source_y = int(h * 0.8)  # Bottom 20%
        source_region[source_y:, :] += 0.01
        density = Field2D(np.clip(source_region, 0, 1), dx=density.dx, dy=density.dy)

        # Visualize density field
        vis = visual.colorize(density, palette="viridis", vmin=0.0, vmax=1.0)

        frame_count += 1
        yield vis


if __name__ == "__main__":
    print("Starting interactive smoke simulation...")
    print("This demonstrates fluid dynamics using Navier-Stokes equations")
    print("Watch the smoke swirl and diffuse!")
    print()
    print("Controls:")
    print("  SPACE: Pause/Resume")
    print("  →: Step forward (when paused)")
    print("  ↑↓: Speed control")
    print("  Q/ESC: Quit")
    print()
    print("Close window to exit.")

    # Create generator
    gen = create_smoke_simulation()

    # Display interactively
    visual.display(
        frame_generator=lambda: next(gen),
        title="Smoke Simulation - Navier-Stokes - CCDSL",
        target_fps=30,
        scale=4
    )

    print("\nSimulation ended.")
