"""Spatial Queries Example.

Demonstrates spatial relationship queries:
- Distance calculations
- Intersection tests
- Containment tests
- Closest point queries
"""

import numpy as np
from morphogen.stdlib.geometry import (
    point2d, line2d, circle, rectangle, polygon,
    distance_point_point, distance_point_line, distance_point_circle,
    intersect_circle_circle,
    contains_circle_point, contains_rectangle_point, contains_polygon_point,
    closest_point_circle, closest_point_line
)


def main():
    print("=" * 60)
    print("Spatial Queries Example")
    print("=" * 60)

    # Distance calculations
    print("\n1. Distance Calculations")
    print("-" * 40)

    p1 = point2d(0, 0)
    p2 = point2d(3, 4)
    dist = distance_point_point(p1, p2)
    print(f"Distance between {p1} and {p2}:")
    print(f"  {dist:.3f} (expected: 5.0)")

    # Distance to line
    print("\n2. Distance to Line Segment")
    print("-" * 40)

    line = line2d(point2d(0, 0), point2d(10, 0))
    test_point = point2d(5, 3)

    dist = distance_point_line(test_point, line)
    print(f"Line from (0,0) to (10,0)")
    print(f"Point at (5,3)")
    print(f"  Distance: {dist:.3f} (perpendicular distance)")

    # Closest point on line
    closest = closest_point_line(line, test_point)
    print(f"  Closest point on line: ({closest.x:.3f}, {closest.y:.3f})")

    # Distance to circle
    print("\n3. Distance to Circle")
    print("-" * 40)

    circ = circle(center=point2d(0, 0), radius=5.0)

    # Point outside
    p_outside = point2d(10, 0)
    dist = distance_point_circle(p_outside, circ)
    print(f"Circle at origin with radius 5.0")
    print(f"Point outside at (10, 0):")
    print(f"  Distance to perimeter: {dist:.3f}")

    # Point inside
    p_inside = point2d(2, 0)
    dist = distance_point_circle(p_inside, circ)
    print(f"\nPoint inside at (2, 0):")
    print(f"  Distance to perimeter: {dist:.3f} (negative = inside)")

    # Closest point on circle
    closest = closest_point_circle(circ, p_outside)
    print(f"\nClosest point on circle to (10,0): ({closest.x:.3f}, {closest.y:.3f})")

    # Circle intersection
    print("\n4. Circle-Circle Intersection")
    print("-" * 40)

    c1 = circle(center=point2d(0, 0), radius=5.0)
    c2 = circle(center=point2d(6, 0), radius=5.0)

    result = intersect_circle_circle(c1, c2)

    if result:
        p1, p2 = result
        print(f"Circles intersect at two points:")
        print(f"  Point 1: ({p1.x:.3f}, {p1.y:.3f})")
        print(f"  Point 2: ({p2.x:.3f}, {p2.y:.3f})")

        # Verify points are on both circles
        d1_c1 = distance_point_point(c1.center, p1)
        d1_c2 = distance_point_point(c2.center, p1)
        print(f"  Verification: distances to centers = {d1_c1:.3f}, {d1_c2:.3f}")
    else:
        print("Circles do not intersect")

    # Non-intersecting circles
    c3 = circle(center=point2d(20, 0), radius=2.0)
    result = intersect_circle_circle(c1, c3)
    print(f"\nCircles at (0,0) r=5 and (20,0) r=2:")
    print(f"  Intersect: {result is not None}")

    # Containment tests
    print("\n5. Point Containment Tests")
    print("-" * 40)

    # Circle containment
    circ = circle(center=point2d(0, 0), radius=10.0)
    test_points = [
        point2d(5, 0),    # Inside
        point2d(10, 0),   # On boundary
        point2d(15, 0),   # Outside
    ]

    print("Circle at origin, radius 10:")
    for p in test_points:
        inside = contains_circle_point(circ, p)
        print(f"  Point ({p.x}, {p.y}): {'inside' if inside else 'outside'}")

    # Rectangle containment
    print("\n6. Rectangle Containment")
    print("-" * 40)

    rect = rectangle(point2d(0, 0), width=10, height=6, rotation=0)
    test_points = [
        point2d(0, 0),    # Center
        point2d(4, 2),    # Inside
        point2d(5, 3),    # On edge
        point2d(10, 10),  # Outside
    ]

    print("Rectangle centered at origin, 10x6:")
    for p in test_points:
        inside = contains_rectangle_point(rect, p)
        print(f"  Point ({p.x}, {p.y}): {'inside' if inside else 'outside'}")

    # Rotated rectangle
    print("\nRotated rectangle (45°):")
    rect_rot = rectangle(point2d(0, 0), width=4, height=2, rotation=np.pi/4)
    p_test = point2d(0, 0)
    print(f"  Point at center: {'inside' if contains_rectangle_point(rect_rot, p_test) else 'outside'}")

    # Polygon containment
    print("\n7. Polygon Containment (Ray Casting)")
    print("-" * 40)

    # Create triangle
    vertices = np.array([
        [0, 0],
        [6, 0],
        [3, 5]
    ])
    tri = polygon(vertices)

    test_points = [
        point2d(3, 2),     # Inside
        point2d(3, 0),     # On edge
        point2d(10, 10),   # Outside
    ]

    print("Triangle vertices: (0,0), (6,0), (3,5)")
    for p in test_points:
        inside = contains_polygon_point(tri, p)
        print(f"  Point ({p.x}, {p.y}): {'inside' if inside else 'outside'}")

    # Complex polygon (pentagon)
    print("\n8. Complex Polygon Containment")
    print("-" * 40)

    vertices = np.array([
        [0, 2],
        [2, 0],
        [3, 3],
        [1, 4],
        [-1, 3]
    ])
    pent = polygon(vertices)

    grid_size = 5
    print(f"Testing {grid_size}x{grid_size} grid of points:")

    count_inside = 0
    for x in range(-2, 3):
        for y in range(0, 5):
            p = point2d(float(x), float(y))
            if contains_polygon_point(pent, p):
                count_inside += 1

    print(f"  Points inside polygon: {count_inside}/{grid_size*grid_size}")

    # Collision detection example
    print("\n9. Simple Collision Detection")
    print("-" * 40)

    player = circle(center=point2d(5, 5), radius=1.0)
    obstacles = [
        circle(center=point2d(10, 5), radius=2.0),
        circle(center=point2d(3, 8), radius=1.5),
        circle(center=point2d(8, 2), radius=1.0),
    ]

    print(f"Player at ({player.center.x}, {player.center.y})")
    print(f"Checking collision with {len(obstacles)} obstacles:")

    for i, obs in enumerate(obstacles):
        # Check if circles overlap
        dist_centers = distance_point_point(player.center, obs.center)
        colliding = dist_centers < (player.radius + obs.radius)

        print(f"  Obstacle {i+1} at ({obs.center.x}, {obs.center.y}): "
              f"{'COLLISION' if colliding else 'safe'} (dist={dist_centers:.3f})")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
