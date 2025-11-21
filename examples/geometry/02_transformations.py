"""Geometric Transformations Example.

Demonstrates 2D transformations:
- Translation (moving shapes)
- Rotation (rotating shapes)
- Scaling (resizing shapes)
- Chaining transformations
"""

import numpy as np
from morphogen.stdlib.geometry import (
    point2d, circle, rectangle, polygon,
    translate_circle, translate_rectangle, translate_polygon,
    rotate_circle, rotate_rectangle, rotate_polygon,
    scale_circle, scale_rectangle, scale_polygon,
    area, centroid
)


def main():
    print("=" * 60)
    print("Geometric Transformations Example")
    print("=" * 60)

    # Translation
    print("\n1. Translation (Moving Shapes)")
    print("-" * 40)

    circ1 = circle(center=point2d(0, 0), radius=3.0)
    print(f"Original circle: {circ1}")

    circ2 = translate_circle(circ1, dx=10.0, dy=5.0)
    print(f"Translated circle: {circ2}")
    print(f"  Center moved from ({circ1.center.x}, {circ1.center.y}) "
          f"to ({circ2.center.x}, {circ2.center.y})")

    # Rotation
    print("\n2. Rotation")
    print("-" * 40)

    rect1 = rectangle(point2d(5, 0), width=4, height=2, rotation=0)
    print(f"Original rectangle: {rect1}")
    print(f"  Center: ({rect1.center.x}, {rect1.center.y})")

    # Rotate 45 degrees around origin
    rect2 = rotate_rectangle(rect1, angle=np.pi/4)
    print(f"\nRotated 45° around its center:")
    print(f"  New rotation: {rect2.rotation * 180 / np.pi:.1f}°")

    # Rotation around custom center
    p1 = point2d(4, 0)
    p2 = rotate_circle(
        circle(center=p1, radius=1.0),
        center=point2d(0, 0),
        angle=np.pi/2
    )
    print(f"\nRotating point (4, 0) by 90° around origin:")
    print(f"  Original: ({p1.x}, {p1.y})")
    print(f"  Rotated: ({p2.center.x:.3f}, {p2.center.y:.3f})")
    print(f"  Expected: (0, 4)")

    # Scaling
    print("\n3. Scaling (Resizing)")
    print("-" * 40)

    circ1 = circle(center=point2d(2, 2), radius=1.0)
    print(f"Original circle:")
    print(f"  Center: ({circ1.center.x}, {circ1.center.y})")
    print(f"  Radius: {circ1.radius}")
    print(f"  Area: {area(circ1):.3f}")

    # Scale by 3x from origin
    circ2 = scale_circle(circ1, center=point2d(0, 0), scale=3.0)
    print(f"\nScaled 3x from origin:")
    print(f"  Center: ({circ2.center.x}, {circ2.center.y})")
    print(f"  Radius: {circ2.radius}")
    print(f"  Area: {area(circ2):.3f} (9x larger)")

    # Non-uniform scaling
    print("\n4. Non-Uniform Scaling")
    print("-" * 40)

    rect1 = rectangle(point2d(0, 0), width=2, height=3)
    print(f"Original rectangle: {rect1.width}x{rect1.height}")
    print(f"  Area: {area(rect1):.3f}")

    rect2 = scale_rectangle(rect1, center=rect1.center, scale_x=2.0, scale_y=3.0)
    print(f"\nScaled 2x horizontally, 3x vertically:")
    print(f"  Dimensions: {rect2.width}x{rect2.height}")
    print(f"  Area: {area(rect2):.3f} (6x larger)")

    # Chaining transformations
    print("\n5. Chaining Transformations")
    print("-" * 40)

    vertices = np.array([[0, 0], [1, 0], [0.5, 1]])
    tri = polygon(vertices)
    print(f"Original triangle:")
    print(f"  Area: {area(tri):.3f}")
    print(f"  Centroid: {centroid(tri)}")

    # Transform sequence: translate → rotate → scale
    tri = translate_polygon(tri, dx=5.0, dy=3.0)
    print(f"\nAfter translation:")
    print(f"  Centroid: {centroid(tri)}")

    c = centroid(tri)
    tri = rotate_polygon(tri, center=c, angle=np.pi/4)
    print(f"\nAfter 45° rotation around centroid:")
    print(f"  Centroid: {centroid(tri)} (unchanged)")

    tri = scale_polygon(tri, center=c, scale_x=2.0, scale_y=2.0)
    print(f"\nAfter 2x uniform scaling:")
    print(f"  Area: {area(tri):.3f} (4x original)")
    print(f"  Centroid: {centroid(tri)} (unchanged)")

    # Orbit simulation (rotation around point)
    print("\n6. Orbit Simulation")
    print("-" * 40)

    planet = circle(center=point2d(10, 0), radius=0.5)
    sun = point2d(0, 0)

    print(f"Planet orbiting sun at origin")
    print(f"Initial position: ({planet.center.x}, {planet.center.y})")

    for i in range(4):
        angle = (i + 1) * np.pi / 2  # 90° increments
        planet = rotate_circle(planet, center=sun, angle=np.pi/2)
        print(f"  After {(i+1)*90}°: ({planet.center.x:.3f}, {planet.center.y:.3f})")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
