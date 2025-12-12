"""Generate visual outputs for all portfolio examples.

This script runs each portfolio example and generates output images
to verify they work correctly and provide visual references.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path to import morphogen
sys.path.insert(0, str(Path(__file__).parent.parent))

from morphogen.stdlib.field import field, Field2D
from morphogen.stdlib.visual import visual


def generate_hello_heat():
    """Generate output for 01_hello_heat.morph."""
    print("Generating 01_hello_heat.morph output...")

    # Allocate temperature field
    temp = field.alloc((128, 128), fill_value=0.0)

    # Initialize with hot spot in center
    cx, cy = 64, 64
    radius = 10.0

    for y in range(128):
        for x in range(128):
            dx = x - cx
            dy = y - cy
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radius:
                temp.data[y, x] = 100.0

    # Simulate diffusion for 200 steps
    DIFFUSION_RATE = 0.1
    dt = 0.1

    for step in range(200):
        temp = field.diffuse(temp, rate=DIFFUSION_RATE, dt=dt, iterations=10)

        # Save output at key frames
        if step in [0, 50, 100, 199]:
            vis = visual.colorize(temp, palette="fire", vmin=0.0, vmax=100.0)
            output_path = f"examples/output_01_hello_heat_step{step:03d}.png"
            visual.output(vis, path=output_path)

    print(f"  ✓ Generated output_01_hello_heat_step*.png")


def generate_pulsing_circle():
    """Generate output for 02_pulsing_circle.morph."""
    print("Generating 02_pulsing_circle.morph output...")

    BASE_RADIUS = 20.0
    PULSE_AMPLITUDE = 10.0
    PULSE_SPEED = 2.0
    GRID_SIZE = 128

    time = 0.0
    dt = 0.05

    for step in range(200):
        # Calculate current radius
        radius = BASE_RADIUS + PULSE_AMPLITUDE * np.sin(time * PULSE_SPEED)

        # Create circular field
        cx, cy = GRID_SIZE / 2, GRID_SIZE / 2
        field_data = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                dx = x - cx
                dy = y - cy
                dist = np.sqrt(dx * dx + dy * dy)

                edge_width = 2.0
                if dist < radius - edge_width:
                    field_data[y, x] = 1.0
                elif dist < radius + edge_width:
                    t = (radius + edge_width - dist) / (2.0 * edge_width)
                    field_data[y, x] = t
                else:
                    field_data[y, x] = 0.0

        # Save output at key frames
        if step in [0, 50, 100, 150, 199]:
            f = Field2D(field_data)
            vis = visual.colorize(f, palette="viridis", vmin=0.0, vmax=1.0)
            output_path = f"examples/output_02_pulsing_circle_step{step:03d}.png"
            visual.output(vis, path=output_path)

        time += dt

    print(f"  ✓ Generated output_02_pulsing_circle_step*.png")


def generate_wave_ripples():
    """Generate output for 03_wave_ripples.morph."""
    print("Generating 03_wave_ripples.morph output...")

    # Wave equation parameters
    WAVE_SPEED = 0.5
    DAMPING = 0.995
    INIT_AMPLITUDE = 1.0

    # Initialize fields
    u = field.alloc((128, 128), fill_value=0.0)
    v = field.alloc((128, 128), fill_value=0.0)

    # Initialize with Gaussian bump in center
    cx, cy = 64, 64
    sigma = 5.0

    for y in range(128):
        for x in range(128):
            dx = x - cx
            dy = y - cy
            dist_sq = dx * dx + dy * dy
            u.data[y, x] = INIT_AMPLITUDE * np.exp(-dist_sq / (2.0 * sigma * sigma))

    # Simulate wave propagation
    dt = 0.1
    c_squared = WAVE_SPEED * WAVE_SPEED

    for step in range(300):
        # Wave equation
        lap = field.laplacian(u)
        v.data = v.data + lap.data * c_squared * dt
        u.data = u.data + v.data * dt

        # Damping
        v.data = v.data * DAMPING

        # Save output at key frames
        if step in [0, 30, 60, 120, 200, 299]:
            vis = visual.colorize(u, palette="coolwarm", vmin=-1.0, vmax=1.0)
            output_path = f"examples/output_03_wave_ripples_step{step:03d}.png"
            visual.output(vis, path=output_path)

    print(f"  ✓ Generated output_03_wave_ripples_step*.png")


def generate_heat_equation():
    """Generate output for 10_heat_equation.morph."""
    print("Generating 10_heat_equation.morph output...")

    KAPPA = 0.1
    HOT_TEMP = 350.0
    COLD_TEMP = 250.0
    AMBIENT_TEMP = 300.0

    # Initialize with ambient temperature
    temp = field.alloc((256, 256), fill_value=AMBIENT_TEMP)

    dt = 0.01

    for step in range(1000):
        # Apply heat source at top
        for x in range(256):
            for y in range(20):
                temp.data[y, x] = HOT_TEMP

        # Apply cold sink at bottom
        for x in range(256):
            for y in range(235, 256):
                temp.data[y, x] = COLD_TEMP

        # Diffuse
        temp = field.diffuse(temp, rate=KAPPA, dt=dt, iterations=20)

        # Save output at key frames
        if step in [0, 100, 250, 500, 999]:
            vis = visual.colorize(temp, palette="fire", vmin=COLD_TEMP, vmax=HOT_TEMP)
            output_path = f"examples/output_10_heat_equation_step{step:04d}.png"
            visual.output(vis, path=output_path)

    print(f"  ✓ Generated output_10_heat_equation_step*.png")


def generate_gray_scott():
    """Generate output for 11_gray_scott.morph."""
    print("Generating 11_gray_scott.morph output...")

    Du = 0.16
    Dv = 0.08
    F = 0.060
    K = 0.062

    # Initialize fields
    u = field.alloc((256, 256), fill_value=1.0)
    v = field.alloc((256, 256), fill_value=0.0)

    # Initialize with perturbation in center
    cx, cy = 128, 128
    radius = 20.0

    for y in range(256):
        for x in range(256):
            dx = x - cx
            dy = y - cy
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radius:
                u.data[y, x] = 0.5
                # Add some pseudo-random variation
                rand_offset = ((x * 13 + y * 17) % 100) / 400.0
                v.data[y, x] = 0.25 + rand_offset

    # Simulate Gray-Scott reaction-diffusion
    dt = 1.0

    for step in range(10000):
        # Gray-Scott reaction
        uvv = u.data * v.data * v.data
        du_dt = Du * field.laplacian(u).data - uvv + F * (1.0 - u.data)
        dv_dt = Dv * field.laplacian(v).data + uvv - (F + K) * v.data

        # Euler integration
        u.data = u.data + du_dt * dt
        v.data = v.data + dv_dt * dt

        # Save output at key frames
        if step in [0, 100, 500, 1000, 2500, 5000, 9999]:
            vis = visual.colorize(v, palette="viridis", vmin=0.0, vmax=1.0)
            output_path = f"examples/output_11_gray_scott_step{step:05d}.png"
            visual.output(vis, path=output_path)

            if step % 1000 == 0:
                print(f"    Step {step}/10000")

    print(f"  ✓ Generated output_11_gray_scott_step*.png")


def main():
    """Generate all portfolio example outputs."""
    print("=" * 60)
    print("Portfolio Examples Visual Output Generator")
    print("=" * 60)
    print()

    # Generate outputs for each example
    generate_hello_heat()
    print()
    generate_pulsing_circle()
    print()
    generate_wave_ripples()
    print()
    generate_heat_equation()
    print()
    generate_gray_scott()
    print()

    print("=" * 60)
    print("All portfolio example outputs generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
