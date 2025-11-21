"""Wolfram Elementary Cellular Automata - 1D Pattern Generator

Stephen Wolfram's Elementary Cellular Automata are the simplest class of
cellular automata, yet they produce incredibly diverse and complex patterns.

This demo showcases:
- All 256 Wolfram rules
- Famous rules (30, 90, 110, 184)
- Pattern classification (uniform, periodic, chaotic, complex)
- Beautiful visualizations
- Cross-domain integration with palette and image domains

Key Rules:
- Rule 30: Chaotic, used in random number generation
- Rule 90: Sierpinski triangle pattern
- Rule 110: Turing complete, complex computation
- Rule 184: Traffic flow simulation
"""

import numpy as np
from morphogen.stdlib import cellular, visual, palette, image, color

def visualize_wolfram_history(history, colormap='binary'):
    """Visualize Wolfram CA evolution history as a 2D image.

    Args:
        history: List of 1D cellular fields
        colormap: Color palette name

    Returns:
        RGB image showing spacetime diagram
    """
    # Stack history into 2D array
    height = len(history)
    width = history[0].width

    spacetime = np.zeros((height, width), dtype=np.float32)
    for i, field in enumerate(history):
        spacetime[i, :] = field.data

    # Apply colormap
    pal = palette.create_gradient(colormap, 256)
    img = palette.apply(pal, spacetime)

    return img

def run_wolfram_rule(rule_number, width=400, steps=300, colormap='binary',
                    initial_pattern='single'):
    """Run a specific Wolfram rule and visualize.

    Args:
        rule_number: Wolfram rule (0-255)
        width: CA width
        steps: Number of generations
        colormap: Visualization colormap
        initial_pattern: 'single', 'random', or 'center'
    """
    print(f"Running Rule {rule_number}: {width} cells, {steps} generations")

    # Initialize
    if initial_pattern == 'single':
        # Single cell in center
        data = np.zeros(width, dtype=np.int32)
        data[width // 2] = 1
        field = cellular.CellularField1D(data, states=2)
    elif initial_pattern == 'random':
        # Random initialization
        field = cellular.random_init(width, states=2, density=0.5, seed=rule_number)
    elif initial_pattern == 'center':
        # Few cells in center
        data = np.zeros(width, dtype=np.int32)
        data[width//2-2:width//2+3] = 1
        field = cellular.CellularField1D(data, states=2)
    else:
        raise ValueError(f"Unknown initial pattern: {initial_pattern}")

    # Generate history
    history = cellular.history(field, rule_number, steps=steps)

    # Visualize
    img = visualize_wolfram_history(history, colormap=colormap)

    # Save
    output_path = f"output_wolfram_rule{rule_number:03d}.png"
    image.save(img, output_path)

    print(f"Saved: {output_path}")

def run_famous_rules():
    """Run the most famous and interesting Wolfram rules."""
    print("=" * 60)
    print("FAMOUS WOLFRAM RULES")
    print("=" * 60)
    print()

    famous_rules = [
        (30, "Chaotic - Random number generation", "hot"),
        (90, "Fractal - Sierpinski triangle", "cool"),
        (110, "Complex - Turing complete", "viridis"),
        (184, "Traffic flow simulation", "plasma"),
        (54, "Interesting patterns", "magma"),
        (150, "XOR pattern", "inferno"),
        (225, "Complex boundaries", "cividis"),
    ]

    for rule_num, description, colormap in famous_rules:
        print(f"Rule {rule_num}: {description}")
        run_wolfram_rule(rule_num, width=400, steps=300, colormap=colormap,
                        initial_pattern='single')
        print()

def run_classification_examples():
    """Run examples from each Wolfram classification class.

    Wolfram classified CA into 4 classes:
    - Class 1: Uniform (everything dies)
    - Class 2: Periodic (simple repeating patterns)
    - Class 3: Chaotic (random-looking patterns)
    - Class 4: Complex (interesting structures)
    """
    print("=" * 60)
    print("WOLFRAM CLASSIFICATION EXAMPLES")
    print("=" * 60)
    print()

    examples = [
        # Class 1: Uniform
        (0, "Class 1: Uniform - All cells die", "binary"),
        (8, "Class 1: Uniform - All cells die", "binary"),

        # Class 2: Periodic
        (4, "Class 2: Periodic - Simple patterns", "twilight"),
        (108, "Class 2: Periodic - Regular structure", "twilight_shifted"),

        # Class 3: Chaotic
        (30, "Class 3: Chaotic - Random appearance", "hot"),
        (45, "Class 3: Chaotic - Complex randomness", "afmhot"),

        # Class 4: Complex
        (110, "Class 4: Complex - Localized structures", "viridis"),
        (124, "Class 4: Complex - Propagating patterns", "plasma"),
    ]

    for rule_num, description, colormap in examples:
        print(f"Rule {rule_num}: {description}")
        run_wolfram_rule(rule_num, width=400, steps=300, colormap=colormap,
                        initial_pattern='single')
        print()

def run_random_initial_conditions():
    """Run select rules with random initial conditions."""
    print("=" * 60)
    print("RANDOM INITIAL CONDITIONS")
    print("=" * 60)
    print()

    rules = [30, 90, 110, 150]

    for rule_num in rules:
        print(f"Rule {rule_num} with random initialization")
        run_wolfram_rule(rule_num, width=400, steps=300,
                        colormap='plasma', initial_pattern='random')
        print()

def generate_rule_gallery(rules_per_image=16):
    """Generate a gallery of multiple rules in one image.

    Args:
        rules_per_image: Number of rules to show in gallery
    """
    print(f"Generating rule gallery ({rules_per_image} rules)...")

    width = 100
    steps = 100

    # Select interesting rules
    interesting_rules = [
        0, 4, 18, 22, 30, 45, 54, 60,
        73, 90, 105, 110, 124, 150, 182, 184,
        193, 195, 220, 225, 226, 227, 240, 250
    ]

    selected_rules = interesting_rules[:rules_per_image]

    # Generate gallery
    gallery_rows = []

    for rule_num in selected_rules:
        # Initialize
        data = np.zeros(width, dtype=np.int32)
        data[width // 2] = 1
        field = cellular.CellularField1D(data, states=2)

        # Generate history
        history = cellular.history(field, rule_num, steps=steps)

        # Create spacetime diagram
        spacetime = np.zeros((steps + 1, width), dtype=np.float32)
        for i, f in enumerate(history):
            spacetime[i, :] = f.data

        gallery_rows.append(spacetime)

    # Stack all rules vertically
    gallery = np.vstack(gallery_rows)

    # Visualize
    pal = palette.create_gradient('binary', 256)
    img = palette.apply(pal, gallery)

    # Save
    output_path = f"output_wolfram_gallery_{rules_per_image}_rules.png"
    image.save(img, output_path)

    print(f"Gallery saved: {output_path}")
    print(f"Rules: {selected_rules}")

def analyze_rule_behavior(rule_number, width=400, steps=500):
    """Analyze the behavior of a specific rule.

    Computes statistics about pattern evolution.
    """
    print(f"Analyzing Rule {rule_number}...")

    # Initialize
    data = np.zeros(width, dtype=np.int32)
    data[width // 2] = 1
    field = cellular.CellularField1D(data, states=2)

    # Evolve and track statistics
    alive_counts = []
    densities = []

    for step in range(steps):
        alive = np.sum(field.data == 1)
        density = alive / width

        alive_counts.append(alive)
        densities.append(density)

        field = cellular.step(field, rule_number)

    # Compute statistics
    avg_density = np.mean(densities)
    std_density = np.std(densities)
    max_alive = max(alive_counts)
    min_alive = min(alive_counts)

    print(f"  Average density: {avg_density:.4f}")
    print(f"  Std dev density: {std_density:.4f}")
    print(f"  Min/Max alive: {min_alive}/{max_alive}")

    # Classify behavior
    if std_density < 0.01:
        if avg_density < 0.01:
            classification = "Class 1: Uniform (dies)"
        else:
            classification = "Class 2: Periodic (stable)"
    elif std_density > 0.2:
        classification = "Class 3: Chaotic"
    else:
        classification = "Class 4: Complex"

    print(f"  Classification: {classification}")
    print()

    return {
        'rule': rule_number,
        'avg_density': avg_density,
        'std_density': std_density,
        'classification': classification
    }

def main():
    """Run all Wolfram CA demonstrations."""
    print("=" * 60)
    print("WOLFRAM ELEMENTARY CELLULAR AUTOMATA DEMO")
    print("=" * 60)
    print()

    # Demo 1: Famous rules
    print("Demo 1: Famous and interesting rules")
    print("-" * 60)
    run_famous_rules()
    print()

    # Demo 2: Classification examples
    print("Demo 2: Examples from each Wolfram class")
    print("-" * 60)
    run_classification_examples()
    print()

    # Demo 3: Random initial conditions
    print("Demo 3: Random initial conditions")
    print("-" * 60)
    run_random_initial_conditions()
    print()

    # Demo 4: Rule gallery
    print("Demo 4: Rule gallery")
    print("-" * 60)
    generate_rule_gallery(rules_per_image=16)
    print()

    # Demo 5: Rule analysis
    print("Demo 5: Behavioral analysis")
    print("-" * 60)
    interesting_rules = [0, 30, 90, 110, 184]
    for rule_num in interesting_rules:
        analyze_rule_behavior(rule_num, width=400, steps=500)
    print()

    print("=" * 60)
    print("ALL WOLFRAM CA DEMOS COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
