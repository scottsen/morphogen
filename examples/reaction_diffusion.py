"""
Interactive Gray-Scott reaction-diffusion simulation.

This system creates amazing organic patterns through simple local rules:
- Two chemicals (U and V) diffuse at different rates
- They react: U + 2V → 3V (V consumes U to create more V)
- U is "fed" at a constant rate
- V "dies" at a constant rate

Different parameters create wildly different patterns:
- Coral-like structures
- Maze patterns
- Spots and stripes
- Wandering worms
- Pulsating solitons

Try modifying feed_rate and kill_rate to explore!

Controls:
- SPACE: Pause/Resume
- RIGHT ARROW: Step forward (when paused)
- UP/DOWN ARROWS: Speed control
- Q or ESC: Quit
"""

import numpy as np
from morphogen.stdlib.field import field, Field2D
from morphogen.stdlib.visual import visual


def create_reaction_diffusion():
    """Generate frames for Gray-Scott reaction-diffusion."""
    # Grid parameters
    grid_size = (256, 256)

    # Gray-Scott parameters
    # These create coral/maze-like patterns
    diffusion_u = 0.16    # Diffusion rate for U
    diffusion_v = 0.08    # Diffusion rate for V (slower)
    feed_rate = 0.055     # Feed rate (adds U)
    kill_rate = 0.062     # Kill rate (removes V)

    # Try these for different patterns:
    # Spots: feed=0.0545, kill=0.062
    # Stripes: feed=0.035, kill=0.060
    # Worms: feed=0.058, kill=0.065
    # Chaos: feed=0.026, kill=0.051

    dt = 1.0

    # Initialize fields
    # Start with U=1 everywhere (chemical U fills space)
    u = field.alloc(grid_size, fill_value=1.0)

    # Start with V=0 everywhere except a small seed
    v = field.alloc(grid_size, fill_value=0.0)

    # Add random seeds of chemical V in center
    h, w = grid_size
    seed_data = v.data.copy()

    # Add several random circular seeds
    np.random.seed(42)
    for i in range(10):
        cy = h // 2 + np.random.randint(-20, 20)
        cx = w // 2 + np.random.randint(-20, 20)
        radius = np.random.randint(3, 8)

        yy, xx = np.ogrid[:h, :w]
        mask = (xx - cx)**2 + (yy - cy)**2 <= radius**2
        seed_data[mask] = 1.0

    v = Field2D(seed_data, dx=1.0, dy=1.0)

    print(f"Pattern parameters: feed={feed_rate:.4f}, kill={kill_rate:.4f}")
    print("Watch for patterns to emerge (may take 100-500 frames)...")

    frame_count = 0

    while True:
        # Compute Laplacians (diffusion term)
        lap_u = field.laplacian(u)
        lap_v = field.laplacian(v)

        # Get raw data for reaction computation
        u_data = u.data.copy()
        v_data = v.data.copy()
        lap_u_data = lap_u.data
        lap_v_data = lap_v.data

        # Gray-Scott reaction-diffusion equations:
        # dU/dt = Du*∇²U - U*V² + f*(1-U)
        # dV/dt = Dv*∇²V + U*V² - (f+k)*V

        uvv = u_data * v_data * v_data  # U * V²

        du_dt = (
            diffusion_u * lap_u_data
            - uvv
            + feed_rate * (1.0 - u_data)
        )

        dv_dt = (
            diffusion_v * lap_v_data
            + uvv
            - (feed_rate + kill_rate) * v_data
        )

        # Update (Euler integration)
        u_data += du_dt * dt
        v_data += dv_dt * dt

        # Clamp values
        u_data = np.clip(u_data, 0.0, 1.0)
        v_data = np.clip(v_data, 0.0, 1.0)

        # Update fields
        u = Field2D(u_data, dx=1.0, dy=1.0)
        v = Field2D(v_data, dx=1.0, dy=1.0)

        # Apply boundary conditions
        u = field.boundary(u, spec="periodic")
        v = field.boundary(v, spec="periodic")

        # Visualize V field (where the patterns appear)
        # Use coolwarm palette for beautiful blue-red patterns
        vis = visual.colorize(v, palette="coolwarm", vmin=0.0, vmax=0.5)

        frame_count += 1

        # Print progress occasionally
        if frame_count % 50 == 0:
            print(f"Frame {frame_count}: min(V)={v_data.min():.4f}, max(V)={v_data.max():.4f}")

        yield vis


if __name__ == "__main__":
    print("=" * 60)
    print("  Gray-Scott Reaction-Diffusion Pattern Generator")
    print("=" * 60)
    print()
    print("This simulation creates organic patterns through chemical")
    print("reactions between two substances (U and V).")
    print()
    print("Be patient - interesting patterns emerge after ~100 frames!")
    print()
    print("Controls:")
    print("  SPACE: Pause/Resume")
    print("  →: Step forward (when paused)")
    print("  ↑↓: Speed control")
    print("  Q/ESC: Quit")
    print()
    print("Close window to exit.")
    print()

    # Create generator
    gen = create_reaction_diffusion()

    # Display interactively
    visual.display(
        frame_generator=lambda: next(gen),
        title="Reaction-Diffusion (Gray-Scott) - CCDSL",
        target_fps=30,
        scale=2  # 2x for 256x256
    )

    print("\nSimulation ended.")
