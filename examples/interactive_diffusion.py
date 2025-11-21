"""
Interactive heat diffusion demo with real-time visualization.

This demonstrates the new interactive display capabilities.
Watch the heat spread and smooth out in real-time!

Controls:
- SPACE: Pause/Resume
- RIGHT ARROW: Step forward one frame (when paused)
- UP/DOWN ARROWS: Increase/decrease simulation speed
- Q or ESC: Quit
"""

from morphogen.stdlib.field import field
from morphogen.stdlib.visual import visual


def simulate_heat_diffusion():
    """Generate frames showing heat diffusion."""
    # Start with random temperature distribution
    temp = field.random((128, 128), seed=42, low=0.0, high=1.0)

    # Simulation parameters
    diffusion_rate = 0.2
    dt = 0.1

    while True:
        # Apply diffusion (heat spreads)
        temp = field.diffuse(temp, rate=diffusion_rate, dt=dt, iterations=20)

        # Apply boundary conditions (heat reflects at edges)
        temp = field.boundary(temp, spec="reflect")

        # Colorize and yield frame
        yield visual.colorize(temp, palette="fire")


if __name__ == "__main__":
    print("Starting interactive heat diffusion simulation...")
    print("Close the window or press Q to quit")

    # Create generator
    gen = simulate_heat_diffusion()

    # Display interactively
    visual.display(
        frame_generator=lambda: next(gen),
        title="Heat Diffusion - Creative Computation DSL",
        target_fps=30,
        scale=4  # 4x scaling for better visibility
    )

    print("Simulation ended.")
