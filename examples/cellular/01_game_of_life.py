"""Game of Life - Classic Cellular Automaton Demo

Conway's Game of Life is the most famous cellular automaton, demonstrating
how simple rules can create complex emergent behavior.

This demo showcases:
- Classic Game of Life (B3/S23 rule)
- Beautiful visualization with color palettes
- Evolution over time
- Cross-domain integration with visual and palette domains

Features:
- Random initialization or classic patterns
- Smooth color gradients showing cell density
- Multiple visualization modes
- Export to image/video
"""

import numpy as np
from morphogen.stdlib import cellular, visual, palette, image, color

def create_glider(field, x, y):
    """Place a glider pattern at position (x, y)."""
    pattern = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ], dtype=np.int32)

    h, w = pattern.shape
    field.data[y:y+h, x:x+w] = pattern
    return field

def create_gosper_glider_gun(field, x, y):
    """Place the famous Gosper Glider Gun pattern."""
    pattern_str = """
........................O...........
......................O.O...........
............OO......OO............OO
...........O...O....OO............OO
OO........O.....O...OO..............
OO........O...O.OO....O.O...........
..........O.....O.......O...........
...........O...O....................
............OO......................
"""
    lines = [line.strip() for line in pattern_str.strip().split('\n')]
    h = len(lines)
    w = max(len(line) for line in lines)

    for i, line in enumerate(lines):
        for j, char in enumerate(line):
            if char == 'O':
                if y+i < field.height and x+j < field.width:
                    field.data[y+i, x+j] = 1

    return field

def create_pulsar(field, x, y):
    """Place a pulsar oscillator pattern."""
    # Pulsar is a period-3 oscillator
    pattern = np.array([
        [0,0,1,1,1,0,0,0,1,1,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,0,0,0,0,1,0,1,0,0,0,0,1],
        [1,0,0,0,0,1,0,1,0,0,0,0,1],
        [1,0,0,0,0,1,0,1,0,0,0,0,1],
        [0,0,1,1,1,0,0,0,1,1,1,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,1,1,1,0,0,0,1,1,1,0,0],
        [1,0,0,0,0,1,0,1,0,0,0,0,1],
        [1,0,0,0,0,1,0,1,0,0,0,0,1],
        [1,0,0,0,0,1,0,1,0,0,0,0,1],
        [0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,1,1,1,0,0,0,1,1,1,0,0],
    ], dtype=np.int32)

    h, w = pattern.shape
    field.data[y:y+h, x:x+w] = pattern
    return field

def visualize_life(field, colormap='plasma'):
    """Create beautiful visualization of Game of Life state.

    Args:
        field: Cellular field
        colormap: Color palette name

    Returns:
        RGB image array
    """
    # Create color palette
    pal = palette.create_gradient(colormap, 256)

    # Apply palette to field
    data_norm = field.data.astype(np.float32)
    img = palette.apply(pal, data_norm)

    return img

def visualize_density(field, history_length=10, colormap='viridis'):
    """Visualize cell density over time (shows activity patterns).

    Args:
        field: Current cellular field
        colormap: Color palette name

    Returns:
        RGB image showing density
    """
    # For single frame, just show current state with smooth gradient
    density = field.data.astype(np.float32)

    # Create smooth visualization
    pal = palette.create_gradient(colormap, 256)
    img = palette.apply(pal, density)

    return img

def run_game_of_life_random(width=200, height=200, steps=500,
                            density=0.3, seed=42, save_frames=True):
    """Run Game of Life with random initialization.

    Args:
        width: Grid width
        height: Grid height
        steps: Number of generations
        density: Initial alive cell density
        seed: Random seed
        save_frames: Whether to save output frames
    """
    print(f"Running Game of Life: {width}x{height}, {steps} generations")
    print(f"Initial density: {density}")

    # Create field and rule
    field, rule = cellular.game_of_life((height, width), density=density, seed=seed)

    # Evolve and save key frames
    frames_to_save = [0, 50, 100, 200, 300, 400, 499]

    for step in range(steps):
        if step in frames_to_save and save_frames:
            # Visualize
            img = visualize_life(field, colormap='plasma')

            # Save
            output_path = f"output_game_of_life_step{step:04d}.png"
            image.save(img, output_path)

            # Analyze
            stats = cellular.analyze_pattern(field)
            print(f"Step {step}: {stats['alive_count']} alive cells "
                  f"({stats['density']:.2%} density)")

        # Advance one generation
        field = cellular.step(field, rule)

    print(f"Final generation {field.generation}: Complete")

def run_game_of_life_patterns(width=400, height=300, steps=500):
    """Run Game of Life with classic patterns.

    Demonstrates gliders, oscillators, and the Gosper Glider Gun.
    """
    print(f"Running Game of Life with classic patterns")

    # Create empty field
    field, rule = cellular.game_of_life((height, width), density=0.0, seed=None)

    # Place patterns
    field = create_glider(field, 10, 10)
    field = create_glider(field, 50, 50)
    field = create_pulsar(field, 150, 100)
    field = create_gosper_glider_gun(field, 200, 150)

    print("Patterns placed: gliders, pulsar, Gosper glider gun")

    # Evolve and save key frames
    frames_to_save = [0, 50, 100, 200, 300, 400, 499]

    for step in range(steps):
        if step in frames_to_save:
            # Visualize with different colormaps for variety
            colormap = ['plasma', 'viridis', 'magma', 'inferno'][step % 4]
            img = visualize_life(field, colormap=colormap)

            # Save
            output_path = f"output_game_of_life_patterns_step{step:04d}.png"
            image.save(img, output_path)

            stats = cellular.analyze_pattern(field)
            print(f"Step {step}: {stats['alive_count']} cells alive")

        # Advance
        field = cellular.step(field, rule)

    print("Evolution complete!")

def run_highlife(width=200, height=200, steps=300, density=0.3, seed=42):
    """Run HighLife cellular automaton (B36/S23).

    HighLife is notable for its self-replicating patterns.
    """
    print(f"Running HighLife: {width}x{height}, {steps} generations")

    field, rule = cellular.highlife((height, width), density=density, seed=seed)

    frames_to_save = [0, 50, 100, 150, 200, 250, 299]

    for step in range(steps):
        if step in frames_to_save:
            img = visualize_life(field, colormap='cool')
            output_path = f"output_highlife_step{step:04d}.png"
            image.save(img, output_path)

            stats = cellular.analyze_pattern(field)
            print(f"Step {step}: {stats['alive_count']} cells "
                  f"({stats['density']:.2%} density)")

        field = cellular.step(field, rule)

    print("HighLife evolution complete!")

def run_seeds(width=200, height=200, steps=100, density=0.1, seed=42):
    """Run Seeds cellular automaton (B2/S).

    Creates explosive growth patterns - cells live only one generation.
    """
    print(f"Running Seeds: {width}x{height}, {steps} generations")

    field, rule = cellular.seeds((height, width), density=density, seed=seed)

    frames_to_save = [0, 10, 20, 30, 40, 50, 75, 99]

    for step in range(steps):
        if step in frames_to_save:
            img = visualize_life(field, colormap='hot')
            output_path = f"output_seeds_step{step:04d}.png"
            image.save(img, output_path)

            stats = cellular.analyze_pattern(field)
            print(f"Step {step}: {stats['alive_count']} cells")

        field = cellular.step(field, rule)

    print("Seeds evolution complete!")

def main():
    """Run all Game of Life demonstrations."""
    print("=" * 60)
    print("GAME OF LIFE - CELLULAR AUTOMATON DEMO")
    print("=" * 60)
    print()

    # Demo 1: Random initialization
    print("Demo 1: Random initialization")
    print("-" * 60)
    run_game_of_life_random(width=200, height=200, steps=500, density=0.3, seed=42)
    print()

    # Demo 2: Classic patterns
    print("Demo 2: Classic patterns (gliders, oscillators, guns)")
    print("-" * 60)
    run_game_of_life_patterns(width=400, height=300, steps=500)
    print()

    # Demo 3: HighLife variant
    print("Demo 3: HighLife (B36/S23 - self-replicating patterns)")
    print("-" * 60)
    run_highlife(width=200, height=200, steps=300, density=0.3, seed=42)
    print()

    # Demo 4: Seeds variant
    print("Demo 4: Seeds (B2/S - explosive growth)")
    print("-" * 60)
    run_seeds(width=200, height=200, steps=100, density=0.1, seed=42)
    print()

    print("=" * 60)
    print("ALL DEMOS COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
