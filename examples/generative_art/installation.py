"""Generative Art Installation - Cross-Domain Showcase

This example demonstrates evolving generative art using optimization algorithms
to discover aesthetically pleasing parameter combinations.

Domains Integrated:
- Noise: Procedural texture and pattern generation
- Optimization: Evolutionary algorithms for parameter search
- Palette: Color scheme generation and mapping
- Visual: Rendering and composition
- Image: Output and blending
- Color: Advanced color operations

The installation creates art through:
1. Procedural noise-based patterns
2. Evolutionary optimization of parameters
3. Multi-layer composition
4. Dynamic color palette evolution
5. Aesthetic fitness evaluation

Cross-Domain Integration:
- Noise → Optimization: Parameter space for evolution
- Optimization → Noise: Best parameters generate art
- Noise → Palette → Visual: Colored procedural art
- All → Image: Final composition

Run: python examples/generative_art/installation.py
"""

import numpy as np
from pathlib import Path
from typing import Callable, Tuple, List
from morphogen.stdlib import (
    noise, optimization, palette, visual, image, color, field
)
from morphogen.stdlib.optimization import DifferentialEvolution, OptimizationResult


# ============================================================================
# ART GENERATION FUNCTIONS
# ============================================================================

def generate_layered_noise_art(params: np.ndarray, size: int = 512) -> np.ndarray:
    """Generate multi-layer noise-based art from parameters.

    Args:
        params: Parameter vector controlling art generation
            [0]: Perlin scale (2-32)
            [1]: Perlin octaves (1-8)
            [2]: Worley scale (2-32)
            [3]: FBM scale (2-32)
            [4]: FBM octaves (1-8)
            [5]: Layer 1 blend weight (0-1)
            [6]: Layer 2 blend weight (0-1)
            [7]: Layer 3 blend weight (0-1)
        size: Output size

    Returns:
        Grayscale art image (0-1)
    """
    # Decode parameters
    perlin_scale = params[0]
    perlin_octaves = int(params[1])
    worley_scale = params[2]
    fbm_scale = params[3]
    fbm_octaves = int(params[4])
    w1, w2, w3 = params[5], params[6], params[7]

    # Normalize weights
    total_weight = w1 + w2 + w3 + 1e-10
    w1, w2, w3 = w1/total_weight, w2/total_weight, w3/total_weight

    # Generate noise layers
    layer1 = noise.perlin((size, size), scale=perlin_scale, octaves=perlin_octaves, seed=42)
    layer2 = noise.worley((size, size), scale=worley_scale, seed=43)
    layer3 = noise.fbm((size, size), scale=fbm_scale, octaves=fbm_octaves, seed=44)

    # Composite layers
    result = w1 * layer1.data + w2 * layer2.data + w3 * layer3.data

    # Normalize to 0-1
    result = result - result.min()
    if result.max() > 0:
        result = result / result.max()

    return result


def aesthetic_fitness(art: np.ndarray) -> float:
    """Evaluate aesthetic quality of art (higher is better).

    This is a heuristic fitness function that prefers:
    - Good contrast (wide value range)
    - Interesting structure (medium frequency content)
    - Smooth gradients (not too noisy)
    - Visual balance (entropy)

    Args:
        art: Grayscale art image (0-1)

    Returns:
        Fitness score (higher = more aesthetic)
    """
    # Contrast: Prefer wide dynamic range
    contrast = art.max() - art.min()

    # Variance: Prefer moderate variance (not flat, not too chaotic)
    variance = np.var(art)
    variance_score = 1.0 - abs(variance - 0.15)  # Optimal around 0.15

    # Edge content: Prefer moderate edge density
    grad_y, grad_x = np.gradient(art)
    edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    edge_score = np.mean(edge_magnitude)

    # Entropy: Prefer high entropy (interesting distribution)
    hist, _ = np.histogram(art, bins=64, range=(0, 1))
    hist = hist / (hist.sum() + 1e-10)
    entropy = -np.sum(hist * np.log(hist + 1e-10))
    entropy_score = entropy / np.log(64)  # Normalize

    # Frequency analysis: Prefer medium frequencies
    fft = np.fft.fft2(art)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = np.abs(fft_shift)

    # Get radial frequency distribution
    h, w = art.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2)

    # Medium frequency energy
    mask_medium = (r > h * 0.1) & (r < h * 0.4)
    medium_freq_energy = np.sum(magnitude_spectrum[mask_medium])
    total_energy = np.sum(magnitude_spectrum) + 1e-10
    freq_score = medium_freq_energy / total_energy

    # Combine scores
    fitness = (
        contrast * 0.2 +
        variance_score * 0.2 +
        edge_score * 0.2 +
        entropy_score * 0.2 +
        freq_score * 0.2
    )

    return fitness


def objective_function(params: np.ndarray) -> float:
    """Optimization objective: Minimize negative aesthetic fitness.

    Args:
        params: Art generation parameters

    Returns:
        Negative fitness (for minimization)
    """
    art = generate_layered_noise_art(params, size=256)  # Smaller for speed
    fitness = aesthetic_fitness(art)
    return -fitness  # Minimize negative = maximize fitness


# ============================================================================
# PALETTE EVOLUTION
# ============================================================================

def evolve_palette(base_hue: float, n_colors: int = 256, seed: int = 42) -> palette.Palette:
    """Create an evolved color palette based on base hue.

    Args:
        base_hue: Base hue (0-360)
        n_colors: Number of colors in palette
        seed: Random seed

    Returns:
        Evolved color palette
    """
    np.random.seed(seed)

    # Generate color gradient with variation
    colors = []

    for i in range(n_colors):
        t = i / n_colors

        # Hue variation
        hue = (base_hue + t * 120 + np.sin(t * np.pi * 4) * 30) % 360

        # Saturation variation
        sat = 0.5 + 0.5 * np.sin(t * np.pi * 2)

        # Lightness variation
        light = 0.3 + 0.4 * t

        # Convert HSL to RGB
        c = (1 - abs(2 * light - 1)) * sat
        x = c * (1 - abs((hue / 60) % 2 - 1))
        m = light - c / 2

        if hue < 60:
            r, g, b = c, x, 0
        elif hue < 120:
            r, g, b = x, c, 0
        elif hue < 180:
            r, g, b = 0, c, x
        elif hue < 240:
            r, g, b = 0, x, c
        elif hue < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x

        r, g, b = (r + m) * 255, (g + m) * 255, (b + m) * 255

        colors.append([int(r), int(g), int(b)])

    return palette.Palette(colors=np.array(colors, dtype=np.uint8))


# ============================================================================
# ART INSTALLATION MODES
# ============================================================================

def mode1_optimized_single_piece(output_dir: Path):
    """Mode 1: Single optimized art piece.

    Uses differential evolution to find aesthetically pleasing parameters.
    """
    print("  Mode 1: Optimized Single Piece")
    print("    Running optimization (this may take a minute)...")

    # Define parameter bounds
    bounds = [
        (2.0, 32.0),    # Perlin scale
        (1.0, 8.0),     # Perlin octaves
        (2.0, 32.0),    # Worley scale
        (2.0, 32.0),    # FBM scale
        (1.0, 8.0),     # FBM octaves
        (0.0, 1.0),     # Layer 1 weight
        (0.0, 1.0),     # Layer 2 weight
        (0.0, 1.0),     # Layer 3 weight
    ]

    # Run optimization
    result = DifferentialEvolution.optimize(
        objective_fn=objective_function,
        bounds=bounds,
        population_size=20,
        max_iterations=30,
        seed=42
    )

    print(f"    Optimization complete: fitness = {-result.best_fitness:.4f}")

    # Generate high-res art with best parameters
    art = generate_layered_noise_art(result.best_solution, size=1024)

    # Apply palette
    pal = evolve_palette(base_hue=200, seed=42)
    colored_art = palette.map(pal, art)

    # Save
    output_path = output_dir / "mode1_optimized_single.png"
    image.save(colored_art, str(output_path))

    print(f"    ✓ Saved: {output_path}")


def mode2_evolution_sequence(output_dir: Path):
    """Mode 2: Sequence showing evolution progress.

    Shows how art evolves during optimization.
    """
    print("  Mode 2: Evolution Sequence")

    # Custom callback to capture evolution
    evolution_frames = []

    def capture_callback(iteration: int, best_solution: np.ndarray, best_fitness: float):
        if iteration % 5 == 0:  # Every 5 iterations
            art = generate_layered_noise_art(best_solution, size=512)
            evolution_frames.append((iteration, art, -best_fitness))

    # Run optimization with callback
    bounds = [
        (2.0, 32.0),    # Perlin scale
        (1.0, 8.0),     # Perlin octaves
        (2.0, 32.0),    # Worley scale
        (2.0, 32.0),    # FBM scale
        (1.0, 8.0),     # FBM octaves
        (0.0, 1.0),     # Layer 1 weight
        (0.0, 1.0),     # Layer 2 weight
        (0.0, 1.0),     # Layer 3 weight
    ]

    print("    Running optimization...")
    result = DifferentialEvolution.optimize(
        objective_fn=objective_function,
        bounds=bounds,
        population_size=20,
        max_iterations=25,
        seed=123,
        callback=capture_callback
    )

    # Save evolution frames
    pal = evolve_palette(base_hue=30, seed=123)

    for i, (iteration, art, fitness) in enumerate(evolution_frames):
        colored_art = palette.map(pal, art)
        output_path = output_dir / f"mode2_evolution_iter{iteration:03d}_fit{fitness:.3f}.png"
        image.save(colored_art, str(output_path))

    print(f"    ✓ Saved {len(evolution_frames)} evolution frames")


def mode3_palette_variations(output_dir: Path):
    """Mode 3: Same art with different color palettes.

    Shows how palette choice affects aesthetic.
    """
    print("  Mode 3: Palette Variations")

    # Generate base art (using good parameters from experience)
    params = np.array([12.0, 4, 8.0, 16.0, 5, 0.4, 0.3, 0.3])
    art = generate_layered_noise_art(params, size=512)

    # Try different palettes
    palettes = [
        ("magma", palette.magma(256)),
        ("viridis", palette.viridis(256)),
        ("plasma", palette.plasma(256)),
        ("fire", palette.fire(256)),
        ("ice", palette.ice(256)),
        ("evolved_blue", evolve_palette(200, seed=1)),
        ("evolved_red", evolve_palette(0, seed=2)),
        ("evolved_green", evolve_palette(120, seed=3)),
    ]

    for name, pal in palettes:
        colored_art = palette.map(pal, art)
        output_path = output_dir / f"mode3_palette_{name}.png"
        image.save(colored_art, str(output_path))

    print(f"    ✓ Saved {len(palettes)} palette variations")


def mode4_multi_optimization_gallery(output_dir: Path):
    """Mode 4: Gallery of multiple optimized pieces.

    Runs optimization multiple times with different seeds.
    """
    print("  Mode 4: Multi-Optimization Gallery")

    bounds = [
        (2.0, 32.0),
        (1.0, 8.0),
        (2.0, 32.0),
        (2.0, 32.0),
        (1.0, 8.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
    ]

    n_pieces = 6

    for i in range(n_pieces):
        print(f"    Optimizing piece {i+1}/{n_pieces}...")

        seed = 100 + i * 10

        result = DifferentialEvolution.optimize(
            objective_fn=objective_function,
            bounds=bounds,
            population_size=15,
            max_iterations=20,
            seed=seed
        )

        # Generate art
        art = generate_layered_noise_art(result.best_solution, size=512)

        # Use different palette for each piece
        pal = evolve_palette(base_hue=(i * 60) % 360, seed=seed)
        colored_art = palette.map(pal, art)

        # Save
        fitness = -result.best_fitness
        output_path = output_dir / f"mode4_piece{i+1:02d}_fit{fitness:.3f}.png"
        image.save(colored_art, str(output_path))

    print(f"    ✓ Saved {n_pieces} optimized pieces")


def mode5_layered_composition(output_dir: Path):
    """Mode 5: Multi-layer composition with blend modes.

    Creates complex art by layering and blending multiple patterns.
    """
    print("  Mode 5: Layered Composition")

    size = 512

    # Generate multiple layers with different characteristics
    layer1 = noise.perlin((size, size), scale=16.0, octaves=4, seed=1)
    layer2 = noise.worley((size, size), scale=8.0, seed=2)
    layer3 = noise.fbm((size, size), scale=12.0, octaves=6, seed=3)
    layer4 = noise.turbulence((size, size), scale=10.0, octaves=5, seed=4)

    # Normalize layers
    for layer in [layer1, layer2, layer3, layer4]:
        layer.data = (layer.data - layer.data.min()) / (layer.data.max() - layer.data.min() + 1e-10)

    # Create colored versions with different palettes
    pal1 = palette.magma(256)
    pal2 = palette.viridis(256)
    pal3 = palette.plasma(256)
    pal4 = palette.fire(256)

    img1 = palette.map(pal1, layer1.data)
    img2 = palette.map(pal2, layer2.data)
    img3 = palette.map(pal3, layer3.data)
    img4 = palette.map(pal4, layer4.data)

    # Blend layers with different modes
    # Note: Using simple alpha blending since we don't have image.blend
    def alpha_blend(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
        return (base * (1 - alpha) + overlay * alpha).astype(np.uint8)

    # Composite
    result = img1
    result = alpha_blend(result, img2, 0.5)
    result = alpha_blend(result, img3, 0.3)
    result = alpha_blend(result, img4, 0.2)

    output_path = output_dir / "mode5_layered_composition.png"
    image.save(result, str(output_path))

    print(f"    ✓ Saved layered composition")


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Run generative art installation demonstration."""
    print("=" * 70)
    print("GENERATIVE ART INSTALLATION - CROSS-DOMAIN SHOWCASE")
    print("=" * 70)
    print()
    print("Demonstrating cross-domain integration:")
    print("  • Noise: Procedural pattern generation")
    print("  • Optimization: Evolutionary art parameter search")
    print("  • Palette: Dynamic color scheme evolution")
    print("  • Visual: Rendering and composition")
    print("  • Image: Multi-layer blending")
    print()

    # Create output directory
    output_dir = Path("examples/generative_art/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Art Installations")
    print("-" * 70)

    # Mode 1: Single optimized piece
    print("Step 1: Optimized Single Piece")
    print("-" * 70)
    mode1_optimized_single_piece(output_dir)
    print()

    # Mode 2: Evolution sequence
    print("Step 2: Evolution Sequence")
    print("-" * 70)
    mode2_evolution_sequence(output_dir)
    print()

    # Mode 3: Palette variations
    print("Step 3: Palette Variations")
    print("-" * 70)
    mode3_palette_variations(output_dir)
    print()

    # Mode 4: Multi-optimization gallery
    print("Step 4: Multi-Optimization Gallery")
    print("-" * 70)
    mode4_multi_optimization_gallery(output_dir)
    print()

    # Mode 5: Layered composition
    print("Step 5: Layered Composition")
    print("-" * 70)
    mode5_layered_composition(output_dir)
    print()

    print("=" * 70)
    print("INSTALLATION COMPLETE!")
    print("=" * 70)
    print()
    print("Cross-Domain Integration Demonstrated:")
    print("  ✓ Noise → Optimization: Parameter space exploration")
    print("  ✓ Optimization → Art: Evolutionary aesthetics")
    print("  ✓ Noise → Palette: Procedural coloring")
    print("  ✓ Multi-layer → Composition: Complex imagery")
    print()
    print(f"All outputs saved to: {output_dir}/")
    print()
    print("Installation Modes:")
    print("  1. Optimized Single Piece - Evolutionary parameter search")
    print("  2. Evolution Sequence - Watching art evolve")
    print("  3. Palette Variations - Color scheme exploration")
    print("  4. Multi-Optimization Gallery - Diverse optimized pieces")
    print("  5. Layered Composition - Multi-layer blending")
    print()
    print("Key Insights:")
    print("  • Optimization can discover aesthetically pleasing parameters")
    print("  • Noise provides infinite procedural variation")
    print("  • Palette choice dramatically affects perception")
    print("  • Layer composition creates complex emergent patterns")
    print("  • Evolution guides exploration of creative parameter spaces")


if __name__ == "__main__":
    main()
