"""Basic Geometry Shapes Example.

Demonstrates creating and working with basic 2D geometric primitives:
- Points
- Lines
- Circles
- Rectangles
- Polygons
"""

import numpy as np
from morphogen.stdlib.geometry import (
    point2d, line2d, circle, rectangle, polygon, regular_polygon,
    area, perimeter, centroid, bounding_box
)


def main():
    print("=" * 60)
    print("Basic Geometry Shapes Example")
    print("=" * 60)

    # Create points
    print("\n1. Creating Points")
    print("-" * 40)
    p1 = point2d(x=0.0, y=0.0)
    p2 = point2d(x=3.0, y=4.0)
    print(f"Point 1: {p1}")
    print(f"Point 2: {p2}")

    # Create line
    print("\n2. Creating Line Segment")
    print("-" * 40)
    line = line2d(start=p1, end=p2)
    print(f"Line: {line}")
    print(f"Length: {line.length:.3f}")
    print(f"Direction: {line.direction}")

    # Create circle
    print("\n3. Creating Circle")
    print("-" * 40)
    circ = circle(center=point2d(0, 0), radius=5.0)
    print(f"Circle: {circ}")
    print(f"Area: {area(circ):.3f}")
    print(f"Circumference: {perimeter(circ):.3f}")

    # Create rectangle
    print("\n4. Creating Rectangle")
    print("-" * 40)
    rect = rectangle(
        center=point2d(0, 0),
        width=10.0,
        height=6.0,
        rotation=0.0
    )
    print(f"Rectangle: {rect}")
    print(f"Area: {area(rect):.3f}")
    print(f"Perimeter: {perimeter(rect):.3f}")
    print(f"Vertices:\n{rect.get_vertices()}")

    # Create polygon (triangle)
    print("\n5. Creating Triangle")
    print("-" * 40)
    vertices = np.array([
        [0.0, 0.0],
        [4.0, 0.0],
        [2.0, 3.0]
    ])
    tri = polygon(vertices)
    print(f"Triangle: {tri}")
    print(f"Area: {area(tri):.3f}")
    print(f"Perimeter: {perimeter(tri):.3f}")
    print(f"Centroid: {centroid(tri)}")

    # Create regular polygons
    print("\n6. Creating Regular Polygons")
    print("-" * 40)

    # Pentagon
    pent = regular_polygon(center=point2d(0, 0), radius=1.0, num_sides=5)
    print(f"Regular Pentagon:")
    print(f"  Vertices: {pent.num_vertices}")
    print(f"  Area: {area(pent):.3f}")
    print(f"  Perimeter: {perimeter(pent):.3f}")

    # Hexagon
    hex = regular_polygon(center=point2d(0, 0), radius=1.0, num_sides=6)
    print(f"\nRegular Hexagon:")
    print(f"  Vertices: {hex.num_vertices}")
    print(f"  Area: {area(hex):.3f}")
    print(f"  Perimeter: {perimeter(hex):.3f}")

    # Bounding boxes
    print("\n7. Bounding Boxes")
    print("-" * 40)
    bbox_circle = bounding_box(circ)
    print(f"Circle bounding box: {bbox_circle}")
    print(f"  Width: {bbox_circle.width:.3f}")
    print(f"  Height: {bbox_circle.height:.3f}")
    print(f"  Center: {bbox_circle.center}")

    bbox_tri = bounding_box(tri)
    print(f"\nTriangle bounding box: {bbox_tri}")
    print(f"  Width: {bbox_tri.width:.3f}")
    print(f"  Height: {bbox_tri.height:.3f}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
