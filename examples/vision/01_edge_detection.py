"""
Example: Edge Detection and Feature Extraction

Demonstrates various edge detection algorithms and corner detection.
Processes an image to extract edges and key features.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from morphogen.stdlib.vision import VisionOperations, EdgeDetector, MorphOp
import matplotlib.pyplot as plt
import numpy as np


def create_test_image():
    """Create a test image with shapes"""
    size = 256
    img = np.zeros((size, size))

    # Add circle
    cy, cx = 80, 80
    y, x = np.ogrid[:size, :size]
    circle = (x - cx)**2 + (y - cy)**2 <= 30**2
    img[circle] = 1.0

    # Add rectangle
    img[100:150, 140:200] = 1.0

    # Add triangle
    for i in range(50):
        img[180 + i, 60 - i:60 + i] = 1.0

    # Add some noise
    np.random.seed(42)
    noise = np.random.randn(size, size) * 0.05
    img = np.clip(img + noise, 0, 1)

    # Add gradient background
    gradient = np.linspace(0, 0.2, size).reshape(1, -1)
    img = np.clip(img + gradient, 0, 1)

    return VisionOperations.create_image(img)


def main():
    """Demonstrate edge detection and feature extraction"""

    print("=== Edge Detection and Feature Extraction ===\n")

    # Create test image
    print("Creating test image with geometric shapes...")
    img = create_test_image()
    print(f"  ✓ Image size: {img.shape}")

    # Apply different edge detectors
    print("\nApplying edge detection algorithms...")

    # Sobel
    print("  • Sobel edge detection...")
    sobel_edges = VisionOperations.sobel(img)
    print(f"    Max edge strength: {sobel_edges.magnitude.max():.3f}")

    # Laplacian
    print("  • Laplacian edge detection...")
    laplacian_edges = VisionOperations.laplacian(img)
    print(f"    Max edge strength: {laplacian_edges.data.max():.3f}")

    # Canny
    print("  • Canny edge detection...")
    canny_edges = VisionOperations.canny(img, low_threshold=0.1, high_threshold=0.3)
    edge_pixels = np.sum(canny_edges.data > 0)
    print(f"    Edge pixels: {edge_pixels} ({edge_pixels/(img.shape[0]*img.shape[1])*100:.1f}%)")

    # Gaussian blur
    print("\n  • Applying Gaussian blur...")
    blurred = VisionOperations.gaussian_blur(img, sigma=2.0)

    # Corner detection
    print("\nDetecting corners...")
    corners = VisionOperations.harris_corners(img, k=0.04, threshold=0.01)
    print(f"  ✓ Found {len(corners)} corners")

    # Find top corners
    corners_sorted = sorted(corners, key=lambda c: c.response, reverse=True)
    print(f"  Top 5 corners:")
    for i, corner in enumerate(corners_sorted[:5], 1):
        print(f"    {i}. Position: ({corner.x:.0f}, {corner.y:.0f}), "
              f"Response: {corner.response:.4f}")

    # Morphological operations
    print("\nApplying morphological operations to Canny edges...")
    dilated = VisionOperations.morphological(canny_edges, MorphOp.DILATE, kernel_size=3)
    eroded = VisionOperations.morphological(canny_edges, MorphOp.ERODE, kernel_size=3)
    opened = VisionOperations.morphological(canny_edges, MorphOp.OPEN, kernel_size=3)
    closed = VisionOperations.morphological(canny_edges, MorphOp.CLOSE, kernel_size=3)
    print(f"  ✓ Morphological operations complete")

    # Find contours
    print("\nFinding contours in edge map...")
    contours = VisionOperations.find_contours(canny_edges, min_area=50.0)
    print(f"  ✓ Found {len(contours)} contours")

    for i, contour in enumerate(contours[:5], 1):
        print(f"    {i}. Area: {contour.area:.0f} px², "
              f"Perimeter: {contour.perimeter:.0f} px, "
              f"Centroid: ({contour.centroid[0]:.0f}, {contour.centroid[1]:.0f})")

    # Visualize
    visualize_results(img, sobel_edges, laplacian_edges, canny_edges,
                     blurred, dilated, eroded, corners_sorted[:20], contours)


def visualize_results(img, sobel, laplacian, canny, blurred,
                     dilated, eroded, corners, contours):
    """Visualize edge detection results"""

    fig, axes = plt.subplots(3, 3, figsize=(16, 16))

    # Original image
    ax = axes[0, 0]
    ax.imshow(img.data, cmap='gray', origin='lower')
    ax.set_title('Original Image', fontweight='bold', fontsize=12)
    ax.axis('off')

    # Sobel magnitude
    ax = axes[0, 1]
    im = ax.imshow(sobel.magnitude, cmap='hot', origin='lower')
    ax.set_title('Sobel Edge Detection (Magnitude)', fontweight='bold', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Sobel direction
    ax = axes[0, 2]
    im = ax.imshow(sobel.direction, cmap='hsv', origin='lower')
    ax.set_title('Sobel Edge Direction', fontweight='bold', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, label='Angle (rad)')

    # Laplacian
    ax = axes[1, 0]
    im = ax.imshow(laplacian.data, cmap='hot', origin='lower')
    ax.set_title('Laplacian Edge Detection', fontweight='bold', fontsize=12)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

    # Canny
    ax = axes[1, 1]
    ax.imshow(canny.data, cmap='gray', origin='lower')
    ax.set_title('Canny Edge Detection', fontweight='bold', fontsize=12)
    ax.axis('off')

    # Gaussian blur
    ax = axes[1, 2]
    ax.imshow(blurred.data, cmap='gray', origin='lower')
    ax.set_title('Gaussian Blur (σ=2.0)', fontweight='bold', fontsize=12)
    ax.axis('off')

    # Morphology: Dilation
    ax = axes[2, 0]
    ax.imshow(dilated.data, cmap='gray', origin='lower')
    ax.set_title('Morphology: Dilation', fontweight='bold', fontsize=12)
    ax.axis('off')

    # Morphology: Erosion
    ax = axes[2, 1]
    ax.imshow(eroded.data, cmap='gray', origin='lower')
    ax.set_title('Morphology: Erosion', fontweight='bold', fontsize=12)
    ax.axis('off')

    # Corner detection + Contours
    ax = axes[2, 2]
    ax.imshow(img.data, cmap='gray', origin='lower', alpha=0.7)

    # Draw corners
    for corner in corners:
        circle = plt.Circle((corner.x, corner.y), 3,
                          color='red', fill=False, linewidth=2)
        ax.add_patch(circle)

    # Draw contours
    for contour in contours:
        if len(contour.points) > 0:
            ax.plot(contour.points[:, 0], contour.points[:, 1],
                   'cyan', linewidth=1.5, alpha=0.7)
            # Mark centroid
            ax.plot(contour.centroid[0], contour.centroid[1],
                   'yellow', marker='x', markersize=8, markeredgewidth=2)

    ax.set_title(f'Corners (red) & Contours (cyan)', fontweight='bold', fontsize=12)
    ax.axis('off')

    plt.suptitle('Edge Detection and Feature Extraction',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/tmp/vision_edge_detection.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: /tmp/vision_edge_detection.png")
    plt.show()


if __name__ == "__main__":
    main()
