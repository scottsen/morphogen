"""Brian's Brain - Beautiful 3-State Cellular Automaton

Brian's Brain is a mesmerizing 3-state cellular automaton that creates
wave-like propagation patterns and beautiful emergent structures.

States:
- Dead (0): Black - Inactive cell
- Firing (1): Bright - Active cell
- Refractory (2): Dim - Recently fired, cooling down

Rules:
- Dead cells with exactly 2 firing neighbors become firing
- Firing cells become refractory
- Refractory cells become dead

This creates stunning wave propagation and spiral patterns!

Features:
- Multiple initialization modes
- Beautiful color gradients showing state transitions
- Analysis of pattern dynamics
- Cross-domain visualization
"""

import numpy as np
from morphogen.stdlib import cellular, visual, palette, image, color

def create_custom_colormap():
    """Create a custom 3-color gradient for Brian's Brain states.

    - Dead: Dark blue/black
    - Firing: Bright yellow/white
    - Refractory: Red/orange
    """
    # Create a 3-color palette
    colors = np.array([
        [0.05, 0.05, 0.15],  # Dead: Dark blue
        [1.0, 0.9, 0.3],     # Firing: Bright yellow
        [0.8, 0.2, 0.1],     # Refractory: Red/orange
    ], dtype=np.float32)

    return colors

def visualize_brians_brain(field, use_custom_colors=True):
    """Visualize Brian's Brain with custom colors.

    Args:
        field: Cellular field with 3 states
        use_custom_colors: Use custom color scheme

    Returns:
        RGB image
    """
    if use_custom_colors:
        # Custom 3-state color mapping
        colormap = create_custom_colormap()

        # Create RGB image
        h, w = field.shape
        img = np.zeros((h, w, 3), dtype=np.float32)

        for state in range(3):
            mask = (field.data == state)
            img[mask] = colormap[state]

    else:
        # Use standard palette
        data_norm = field.data.astype(np.float32) / 2.0  # Normalize to [0, 1]
        pal = palette.create_gradient('hot', 256)
        img = palette.apply(pal, data_norm)

    return img

def run_brians_brain_random(width=400, height=400, steps=500,
                           density=0.1, seed=42):
    """Run Brian's Brain with random initialization.

    Args:
        width: Grid width
        height: Grid height
        steps: Number of generations
        density: Initial firing cell density
        seed: Random seed
    """
    print(f"Running Brian's Brain: {width}x{height}, {steps} generations")
    print(f"Initial density: {density}")

    # Create field
    field = cellular.brians_brain((height, width), density=density, seed=seed)

    # Track statistics
    frames_to_save = [0, 50, 100, 200, 300, 400, 499]

    for step in range(steps):
        if step in frames_to_save:
            # Visualize
            img = visualize_brians_brain(field, use_custom_colors=True)

            # Save
            output_path = f"output_brians_brain_step{step:04d}.png"
            image.save(img, output_path)

            # Count states
            dead = np.sum(field.data == 0)
            firing = np.sum(field.data == 1)
            refractory = np.sum(field.data == 2)
            total = dead + firing + refractory

            print(f"Step {step}: Dead={dead}, Firing={firing}, "
                  f"Refractory={refractory} ({firing/total:.2%} firing)")

        # Advance
        field = cellular.brians_brain_step(field)

    print(f"Evolution complete! Final generation: {field.generation}")

def run_brians_brain_patterns(width=400, height=400, steps=500):
    """Run Brian's Brain with specific initial patterns.

    Creates localized patterns that spread as waves.
    """
    print(f"Running Brian's Brain with pattern initialization")

    # Create empty field
    field = cellular.brians_brain((height, width), density=0.0, seed=None)

    # Create initial patterns - place firing cells
    # Central cluster
    field.data[height//2-5:height//2+5, width//2-5:width//2+5] = 1

    # Corner clusters
    field.data[20:30, 20:30] = 1
    field.data[20:30, width-30:width-20] = 1
    field.data[height-30:height-20, 20:30] = 1
    field.data[height-30:height-20, width-30:width-20] = 1

    # Line patterns
    field.data[height//4, width//4:3*width//4] = 1
    field.data[3*height//4, width//4:3*width//4] = 1

    print("Initial patterns placed: clusters, lines")

    frames_to_save = [0, 20, 50, 100, 150, 200, 300, 400, 499]

    for step in range(steps):
        if step in frames_to_save:
            img = visualize_brians_brain(field, use_custom_colors=True)
            output_path = f"output_brians_brain_patterns_step{step:04d}.png"
            image.save(img, output_path)

            firing = np.sum(field.data == 1)
            refractory = np.sum(field.data == 2)
            print(f"Step {step}: Firing={firing}, Refractory={refractory}")

        field = cellular.brians_brain_step(field)

    print("Pattern evolution complete!")

def create_spiral_pattern(field):
    """Create a spiral pattern that produces rotating waves."""
    h, w = field.shape
    cx, cy = w // 2, h // 2

    # Create spiral arms
    angles = np.linspace(0, 4*np.pi, 100)
    for i, angle in enumerate(angles):
        r = i * 0.5
        x = int(cx + r * np.cos(angle))
        y = int(cy + r * np.sin(angle))

        if 0 <= x < w and 0 <= y < h:
            field.data[y, x] = 1

    return field

def run_brians_brain_spirals(width=400, height=400, steps=500):
    """Run Brian's Brain with spiral initialization.

    Creates beautiful rotating spiral patterns.
    """
    print(f"Running Brian's Brain with spiral patterns")

    field = cellular.brians_brain((height, width), density=0.0, seed=None)
    field = create_spiral_pattern(field)

    print("Spiral pattern initialized")

    frames_to_save = [0, 25, 50, 100, 150, 200, 250, 300, 400, 499]

    for step in range(steps):
        if step in frames_to_save:
            img = visualize_brians_brain(field, use_custom_colors=True)
            output_path = f"output_brians_brain_spirals_step{step:04d}.png"
            image.save(img, output_path)
            print(f"Step {step}: Saved frame")

        field = cellular.brians_brain_step(field)

    print("Spiral evolution complete!")

def analyze_wave_dynamics(width=300, height=300, steps=300, density=0.15):
    """Analyze wave propagation dynamics in Brian's Brain.

    Tracks the evolution of firing and refractory states over time.
    """
    print(f"Analyzing Brian's Brain dynamics...")

    field = cellular.brians_brain((height, width), density=density, seed=42)

    # Track state counts over time
    firing_history = []
    refractory_history = []
    total_cells = width * height

    for step in range(steps):
        firing = np.sum(field.data == 1)
        refractory = np.sum(field.data == 2)

        firing_history.append(firing)
        refractory_history.append(refractory)

        field = cellular.brians_brain_step(field)

    # Compute statistics
    avg_firing = np.mean(firing_history)
    avg_refractory = np.mean(refractory_history)
    std_firing = np.std(firing_history)

    print(f"  Average firing cells: {avg_firing:.1f} ({avg_firing/total_cells:.2%})")
    print(f"  Average refractory: {avg_refractory:.1f} ({avg_refractory/total_cells:.2%})")
    print(f"  Std dev firing: {std_firing:.1f}")
    print(f"  Max firing: {max(firing_history)}")
    print(f"  Min firing: {min(firing_history)}")

    # Determine if pattern stabilized
    late_std = np.std(firing_history[-100:])
    if late_std < 10:
        print(f"  Pattern: Stabilized")
    else:
        print(f"  Pattern: Dynamic oscillations")

    print()

def create_comparison_visualization(width=400, height=300, steps=300):
    """Create side-by-side comparison of different densities."""
    print("Creating density comparison...")

    densities = [0.05, 0.10, 0.15, 0.20]
    final_images = []

    for density in densities:
        print(f"  Running density={density}...")
        field = cellular.brians_brain((height, width), density=density, seed=42)

        # Evolve to interesting state
        for _ in range(steps):
            field = cellular.brians_brain_step(field)

        # Visualize
        img = visualize_brians_brain(field, use_custom_colors=True)
        final_images.append(img)

    # Stack images horizontally
    comparison = np.hstack(final_images)

    # Save
    output_path = "output_brians_brain_density_comparison.png"
    image.save(comparison, output_path)

    print(f"Comparison saved: {output_path}")
    print(f"Densities: {densities}")

def main():
    """Run all Brian's Brain demonstrations."""
    print("=" * 60)
    print("BRIAN'S BRAIN - 3-STATE CELLULAR AUTOMATON DEMO")
    print("=" * 60)
    print()

    # Demo 1: Random initialization
    print("Demo 1: Random initialization")
    print("-" * 60)
    run_brians_brain_random(width=400, height=400, steps=500,
                           density=0.1, seed=42)
    print()

    # Demo 2: Pattern initialization
    print("Demo 2: Structured patterns (clusters, lines)")
    print("-" * 60)
    run_brians_brain_patterns(width=400, height=400, steps=500)
    print()

    # Demo 3: Spiral patterns
    print("Demo 3: Spiral initialization (rotating waves)")
    print("-" * 60)
    run_brians_brain_spirals(width=400, height=400, steps=500)
    print()

    # Demo 4: Wave dynamics analysis
    print("Demo 4: Wave dynamics analysis")
    print("-" * 60)
    for density in [0.05, 0.10, 0.15, 0.20]:
        print(f"Density: {density}")
        analyze_wave_dynamics(width=300, height=300, steps=300, density=density)
    print()

    # Demo 5: Density comparison
    print("Demo 5: Density comparison visualization")
    print("-" * 60)
    create_comparison_visualization(width=400, height=300, steps=300)
    print()

    print("=" * 60)
    print("ALL BRIAN'S BRAIN DEMOS COMPLETE!")
    print("=" * 60)
    print()
    print("Brian's Brain creates beautiful wave propagation patterns!")
    print("The 3-state system (dead/firing/refractory) produces")
    print("organic-looking structures and rotating spirals.")

if __name__ == "__main__":
    main()
