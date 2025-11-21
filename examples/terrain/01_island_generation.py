"""
Example: Procedural Island Terrain Generation

Demonstrates heightmap generation with erosion and biome classification.
Creates a realistic island terrain with different biomes.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from morphogen.stdlib.terrain import TerrainOperations, BiomeType
import matplotlib.pyplot as plt
import numpy as np


def main():
    """Generate a procedural island terrain"""

    print("=== Procedural Island Generation ===\n")

    # Parameters
    size = 256
    seed = 42

    print(f"Terrain size: {size}x{size}")
    print(f"Random seed: {seed}\n")

    # Step 1: Generate base heightmap using Perlin noise
    print("Step 1: Generating base terrain with Perlin noise...")
    terrain = TerrainOperations.from_noise_perlin(
        width=size,
        height=size,
        scale=80.0,
        octaves=6,
        persistence=0.5,
        lacunarity=2.0,
        seed=seed
    )
    print(f"  ✓ Height range: {terrain.data.min():.3f} - {terrain.data.max():.3f}")

    # Step 2: Apply island mask (radial falloff)
    print("\nStep 2: Applying island mask...")
    terrain = TerrainOperations.island_mask(terrain, falloff=1.5)
    print(f"  ✓ Height range: {terrain.data.min():.3f} - {terrain.data.max():.3f}")

    # Step 3: Apply erosion for realism
    print("\nStep 3: Simulating erosion...")
    terrain_hydraulic = TerrainOperations.hydraulic_erosion(
        terrain,
        iterations=50,
        rain_amount=0.01,
        evaporation=0.5
    )
    print(f"  ✓ Hydraulic erosion complete")

    terrain_thermal = TerrainOperations.thermal_erosion(
        terrain_hydraulic,
        iterations=30,
        talus_angle=0.3
    )
    print(f"  ✓ Thermal erosion complete")

    # Step 4: Calculate terrain properties
    print("\nStep 4: Analyzing terrain properties...")
    slope = TerrainOperations.calculate_slope(terrain_thermal)
    aspect = TerrainOperations.calculate_aspect(terrain_thermal)

    avg_slope = np.rad2deg(np.mean(slope))
    max_slope = np.rad2deg(np.max(slope))
    print(f"  ✓ Average slope: {avg_slope:.1f}°")
    print(f"  ✓ Maximum slope: {max_slope:.1f}°")

    # Step 5: Classify biomes
    print("\nStep 5: Classifying biomes...")
    biome_map = TerrainOperations.classify_biomes(terrain_thermal)

    # Count biomes
    biome_counts = {}
    for biome_type in BiomeType:
        count = np.sum(biome_map.data == biome_map.biome_types.index(biome_type))
        percentage = (count / (size * size)) * 100
        biome_counts[biome_type.value] = percentage

    print(f"  ✓ Biome distribution:")
    for biome, percentage in sorted(biome_counts.items(), key=lambda x: x[1], reverse=True):
        if percentage > 0.1:
            print(f"    • {biome:12} : {percentage:5.1f}%")

    # Visualize
    visualize_terrain(terrain, terrain_hydraulic, terrain_thermal,
                     slope, biome_map, size)


def visualize_terrain(terrain_base, terrain_hydraulic, terrain_thermal,
                     slope, biome_map, size):
    """Visualize terrain generation pipeline"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Base terrain
    ax = axes[0, 0]
    im = ax.imshow(terrain_base.data, cmap='terrain', origin='lower')
    ax.set_title('Base Terrain (Perlin Noise + Island Mask)', fontweight='bold', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Elevation')

    # Plot 2: After hydraulic erosion
    ax = axes[0, 1]
    im = ax.imshow(terrain_hydraulic.data, cmap='terrain', origin='lower')
    ax.set_title('After Hydraulic Erosion', fontweight='bold', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Elevation')

    # Plot 3: After thermal erosion
    ax = axes[0, 2]
    im = ax.imshow(terrain_thermal.data, cmap='terrain', origin='lower')
    ax.set_title('After Thermal Erosion (Final)', fontweight='bold', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Elevation')

    # Plot 4: Slope map
    ax = axes[1, 0]
    im = ax.imshow(np.rad2deg(slope), cmap='YlOrRd', origin='lower', vmin=0, vmax=45)
    ax.set_title('Slope Map', fontweight='bold', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, label='Slope (degrees)')

    # Plot 5: 3D visualization
    ax = fig.add_subplot(2, 3, 5, projection='3d')
    x = np.arange(0, size, 4)
    y = np.arange(0, size, 4)
    X, Y = np.meshgrid(x, y)
    Z = terrain_thermal.data[::4, ::4]

    surf = ax.plot_surface(X, Y, Z, cmap='terrain', linewidth=0,
                          antialiased=True, alpha=0.9, shade=True)
    ax.set_title('3D Terrain View', fontweight='bold', fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Elevation')
    ax.view_init(elev=30, azim=45)
    plt.colorbar(surf, ax=ax, label='Elevation', shrink=0.5)

    # Plot 6: Biome map
    ax = axes[1, 2]

    # Create color map for biomes
    biome_colors = {
        0: [0.1, 0.2, 0.5],   # Ocean - dark blue
        1: [0.9, 0.9, 0.6],   # Beach - sand
        2: [0.4, 0.7, 0.3],   # Grassland - green
        3: [0.1, 0.5, 0.2],   # Forest - dark green
        4: [0.5, 0.5, 0.5],   # Mountain - gray
        5: [0.9, 0.9, 1.0],   # Snow - white
        6: [0.9, 0.8, 0.5],   # Desert - tan
    }

    # Create RGB image
    biome_rgb = np.zeros((size, size, 3))
    for biome_idx, color in biome_colors.items():
        mask = biome_map.data == biome_idx
        biome_rgb[mask] = color

    ax.imshow(biome_rgb, origin='lower')
    ax.set_title('Biome Classification', fontweight='bold', fontsize=12)
    ax.axis('off')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=biome_colors[i], label=BiomeType(biome_map.biome_types[i]).value.capitalize())
        for i in range(len(biome_map.biome_types))
        if np.any(biome_map.data == i)
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)

    plt.suptitle('Procedural Island Terrain Generation Pipeline',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/tmp/terrain_island_generation.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: /tmp/terrain_island_generation.png")
    plt.show()


if __name__ == "__main__":
    main()
