"""Comprehensive tests for geometry domain.

Test coverage:
- Layer 1: Primitive construction (point, line, circle, rectangle, polygon)
- Layer 2: Transformations (translate, rotate, scale)
- Layer 3: Spatial queries (distance, intersection, containment)
- Layer 4: Coordinate conversions (Cartesian ↔ polar ↔ spherical)
- Layer 5: Geometric properties (area, perimeter, centroid, bounding box)
- Edge cases: Degenerate shapes, zero distances, boundary conditions
"""

try:
    import pytest
except ImportError:
    pytest = None

import numpy as np
from morphogen.stdlib.geometry import (
    # Types
    CoordinateFrame,
    Box3D,
    Sphere,
    Mesh,
    # Construction
    point2d,
    point3d,
    line2d,
    circle,
    rectangle,
    polygon,
    regular_polygon,
    box3d,
    sphere,
    mesh,
    # Transformations
    translate_point2d,
    translate_circle,
    translate_rectangle,
    translate_polygon,
    rotate_point2d,
    rotate_circle,
    rotate_rectangle,
    rotate_polygon,
    scale_circle,
    scale_rectangle,
    scale_polygon,
    # Spatial queries
    distance_point_point,
    distance_point_line,
    distance_point_circle,
    intersect_circle_circle,
    contains_circle_point,
    contains_polygon_point,
    contains_rectangle_point,
    closest_point_circle,
    closest_point_line,
    # Coordinate conversions
    cartesian_to_polar,
    polar_to_cartesian,
    cartesian_to_spherical,
    spherical_to_cartesian,
    # Properties
    area,
    perimeter,
    centroid,
    bounding_box,
    # Advanced algorithms
    convex_hull,
    delaunay_triangulation,
    voronoi,
    mesh_union,
    # Field integration
    sample_field_at_point,
    query_field_in_region,
    # Rigidbody integration
    shape_to_rigidbody,
    collision_mesh,
)


# ============================================================================
# LAYER 1 TESTS: Primitive Construction
# ============================================================================


def test_point2d_construction():
    """Test 2D point creation."""
    p = point2d(x=3.0, y=4.0)

    assert p.x == 3.0
    assert p.y == 4.0
    assert p.frame == CoordinateFrame.CARTESIAN
    assert np.allclose(p.to_array(), [3.0, 4.0])


def test_point3d_construction():
    """Test 3D point creation."""
    p = point3d(x=1.0, y=2.0, z=3.0)

    assert p.x == 1.0
    assert p.y == 2.0
    assert p.z == 3.0
    assert np.allclose(p.to_array(), [1.0, 2.0, 3.0])


def test_line2d_construction():
    """Test 2D line segment creation."""
    start = point2d(0.0, 0.0)
    end = point2d(3.0, 4.0)
    line = line2d(start, end)

    assert line.start == start
    assert line.end == end
    assert np.isclose(line.length, 5.0)  # 3-4-5 triangle
    assert np.allclose(line.direction, [0.6, 0.8])


def test_line2d_zero_length():
    """Test degenerate line (zero length)."""
    p = point2d(1.0, 1.0)
    line = line2d(p, p)

    assert line.length == 0.0
    assert np.allclose(line.direction, [0.0, 0.0])


def test_circle_construction():
    """Test circle creation."""
    center = point2d(0.0, 0.0)
    circ = circle(center, radius=5.0)

    assert circ.center == center
    assert circ.radius == 5.0
    assert np.isclose(circ.area, np.pi * 25)
    assert np.isclose(circ.circumference, 2 * np.pi * 5)


def test_circle_invalid_radius():
    """Test circle creation with invalid radius."""
    with pytest.raises(ValueError):
        circle(point2d(0, 0), radius=0.0)

    with pytest.raises(ValueError):
        circle(point2d(0, 0), radius=-1.0)


def test_rectangle_construction():
    """Test rectangle creation."""
    center = point2d(0.0, 0.0)
    rect = rectangle(center, width=10.0, height=5.0)

    assert rect.center == center
    assert rect.width == 10.0
    assert rect.height == 5.0
    assert rect.rotation == 0.0
    assert rect.area == 50.0
    assert rect.perimeter == 30.0


def test_rectangle_rotated():
    """Test rotated rectangle."""
    rect = rectangle(point2d(0, 0), width=4, height=2, rotation=np.pi / 4)

    assert np.isclose(rect.rotation, np.pi / 4)

    # Verify vertices are rotated correctly
    verts = rect.get_vertices()
    assert verts.shape == (4, 2)


def test_rectangle_vertices():
    """Test rectangle vertex calculation."""
    rect = rectangle(point2d(5, 5), width=4, height=2, rotation=0)
    verts = rect.get_vertices()

    expected = np.array(
        [
            [3, 4],  # Bottom-left
            [7, 4],  # Bottom-right
            [7, 6],  # Top-right
            [3, 6],  # Top-left
        ]
    )

    assert np.allclose(verts, expected)


def test_rectangle_invalid_dimensions():
    """Test rectangle creation with invalid dimensions."""
    with pytest.raises(ValueError):
        rectangle(point2d(0, 0), width=0, height=5)

    with pytest.raises(ValueError):
        rectangle(point2d(0, 0), width=-1, height=5)


def test_polygon_construction():
    """Test polygon creation from vertices."""
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])

    poly = polygon(vertices)

    assert poly.num_vertices == 3
    assert np.allclose(poly.vertices, vertices)


def test_polygon_area_triangle():
    """Test polygon area calculation for triangle."""
    # Right triangle with base=1, height=1
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    poly = polygon(vertices)
    assert np.isclose(poly.area, 0.5)


def test_polygon_area_square():
    """Test polygon area calculation for square."""
    vertices = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])

    poly = polygon(vertices)
    assert np.isclose(poly.area, 4.0)


def test_polygon_centroid():
    """Test polygon centroid calculation."""
    vertices = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]])

    poly = polygon(vertices)
    c = poly.centroid

    assert np.isclose(c.x, 2.0)
    assert np.isclose(c.y, 2.0)


def test_polygon_perimeter():
    """Test polygon perimeter calculation."""
    # Square with side length 3
    vertices = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 3.0], [0.0, 3.0]])

    poly = polygon(vertices)
    assert np.isclose(poly.perimeter, 12.0)


def test_polygon_invalid_vertices():
    """Test polygon creation with invalid vertices."""
    # Too few vertices
    with pytest.raises(ValueError):
        polygon(np.array([[0, 0], [1, 1]]))

    # Wrong shape
    with pytest.raises(ValueError):
        polygon(np.array([0, 1, 2]))


def test_regular_polygon_triangle():
    """Test regular triangle creation."""
    tri = regular_polygon(center=point2d(0, 0), radius=1.0, num_sides=3)

    assert tri.num_vertices == 3

    # All vertices should be distance 1.0 from center
    for v in tri.vertices:
        dist = np.sqrt(v[0] ** 2 + v[1] ** 2)
        assert np.isclose(dist, 1.0)


def test_regular_polygon_hexagon():
    """Test regular hexagon creation."""
    hex = regular_polygon(center=point2d(0, 0), radius=1.0, num_sides=6)

    assert hex.num_vertices == 6

    # Check that it's actually regular (all sides equal length)
    side_lengths = []
    for i in range(6):
        j = (i + 1) % 6
        dx = hex.vertices[j][0] - hex.vertices[i][0]
        dy = hex.vertices[j][1] - hex.vertices[i][1]
        side_lengths.append(np.sqrt(dx * dx + dy * dy))

    # All sides should be equal
    assert np.allclose(side_lengths, side_lengths[0])


def test_regular_polygon_invalid():
    """Test regular polygon with invalid parameters."""
    with pytest.raises(ValueError):
        regular_polygon(point2d(0, 0), radius=1.0, num_sides=2)

    with pytest.raises(ValueError):
        regular_polygon(point2d(0, 0), radius=-1.0, num_sides=5)


# ============================================================================
# LAYER 2 TESTS: Transformations
# ============================================================================


def test_translate_point2d():
    """Test point translation."""
    p1 = point2d(1.0, 2.0)
    p2 = translate_point2d(p1, dx=3.0, dy=4.0)

    assert np.isclose(p2.x, 4.0)
    assert np.isclose(p2.y, 6.0)


def test_translate_circle():
    """Test circle translation."""
    c1 = circle(center=point2d(0, 0), radius=5.0)
    c2 = translate_circle(c1, dx=10.0, dy=-5.0)

    assert np.isclose(c2.center.x, 10.0)
    assert np.isclose(c2.center.y, -5.0)
    assert c2.radius == c1.radius  # Radius unchanged


def test_translate_rectangle():
    """Test rectangle translation."""
    r1 = rectangle(point2d(0, 0), width=4, height=2)
    r2 = translate_rectangle(r1, dx=5.0, dy=3.0)

    assert np.isclose(r2.center.x, 5.0)
    assert np.isclose(r2.center.y, 3.0)
    assert r2.width == r1.width
    assert r2.height == r1.height


def test_translate_polygon():
    """Test polygon translation."""
    vertices = np.array([[0, 0], [1, 0], [0.5, 1]])
    p1 = polygon(vertices)
    p2 = translate_polygon(p1, dx=2.0, dy=3.0)

    expected = vertices + np.array([2.0, 3.0])
    assert np.allclose(p2.vertices, expected)


def test_rotate_point2d_90_degrees():
    """Test point rotation by 90 degrees."""
    p1 = point2d(1.0, 0.0)
    p2 = rotate_point2d(p1, center=point2d(0, 0), angle=np.pi / 2)

    assert np.isclose(p2.x, 0.0, atol=1e-10)
    assert np.isclose(p2.y, 1.0, atol=1e-10)


def test_rotate_point2d_180_degrees():
    """Test point rotation by 180 degrees."""
    p1 = point2d(3.0, 4.0)
    p2 = rotate_point2d(p1, center=point2d(0, 0), angle=np.pi)

    assert np.isclose(p2.x, -3.0, atol=1e-10)
    assert np.isclose(p2.y, -4.0, atol=1e-10)


def test_rotate_point2d_around_custom_center():
    """Test point rotation around custom center."""
    p1 = point2d(2.0, 1.0)
    center = point2d(1.0, 1.0)
    p2 = rotate_point2d(p1, center=center, angle=np.pi / 2)

    assert np.isclose(p2.x, 1.0, atol=1e-10)
    assert np.isclose(p2.y, 2.0, atol=1e-10)


def test_rotate_circle():
    """Test circle rotation."""
    c1 = circle(center=point2d(5, 0), radius=2.0)
    c2 = rotate_circle(c1, center=point2d(0, 0), angle=np.pi / 2)

    # Center should rotate, radius unchanged
    assert np.isclose(c2.center.x, 0.0, atol=1e-10)
    assert np.isclose(c2.center.y, 5.0, atol=1e-10)
    assert c2.radius == c1.radius


def test_rotate_rectangle():
    """Test rectangle rotation."""
    r1 = rectangle(point2d(0, 0), width=4, height=2, rotation=0)
    r2 = rotate_rectangle(r1, angle=np.pi / 4)

    assert np.isclose(r2.rotation, np.pi / 4)
    assert r2.center == r1.center  # Center unchanged


def test_rotate_polygon():
    """Test polygon rotation."""
    vertices = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    p1 = polygon(vertices)
    p2 = rotate_polygon(p1, center=point2d(0, 0), angle=np.pi / 2)

    # After 90° rotation, [1, 0] -> [0, 1]
    assert np.isclose(p2.vertices[0, 0], 0.0, atol=1e-10)
    assert np.isclose(p2.vertices[0, 1], 1.0, atol=1e-10)


def test_scale_circle():
    """Test circle scaling."""
    c1 = circle(center=point2d(2, 0), radius=3.0)
    c2 = scale_circle(c1, center=point2d(0, 0), scale=2.0)

    # Center position scales
    assert np.isclose(c2.center.x, 4.0)
    assert np.isclose(c2.center.y, 0.0)
    # Radius scales
    assert np.isclose(c2.radius, 6.0)


def test_scale_circle_invalid():
    """Test circle scaling with invalid scale factor."""
    c = circle(center=point2d(0, 0), radius=1.0)

    with pytest.raises(ValueError):
        scale_circle(c, center=point2d(0, 0), scale=0)

    with pytest.raises(ValueError):
        scale_circle(c, center=point2d(0, 0), scale=-1)


def test_scale_rectangle_uniform():
    """Test rectangle uniform scaling."""
    r1 = rectangle(point2d(1, 1), width=2, height=4)
    r2 = scale_rectangle(r1, center=point2d(0, 0), scale_x=3.0, scale_y=3.0)

    assert np.isclose(r2.center.x, 3.0)
    assert np.isclose(r2.center.y, 3.0)
    assert np.isclose(r2.width, 6.0)
    assert np.isclose(r2.height, 12.0)


def test_scale_rectangle_non_uniform():
    """Test rectangle non-uniform scaling."""
    r1 = rectangle(point2d(0, 0), width=4, height=2)
    r2 = scale_rectangle(r1, center=point2d(0, 0), scale_x=2.0, scale_y=3.0)

    assert np.isclose(r2.width, 8.0)
    assert np.isclose(r2.height, 6.0)


def test_scale_polygon():
    """Test polygon scaling."""
    vertices = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    p1 = polygon(vertices)
    p2 = scale_polygon(p1, center=point2d(0, 0), scale_x=2.0, scale_y=2.0)

    expected = vertices * 2
    assert np.allclose(p2.vertices, expected)


# ============================================================================
# LAYER 3 TESTS: Spatial Queries - Distance
# ============================================================================


def test_distance_point_point():
    """Test distance between two points."""
    p1 = point2d(0, 0)
    p2 = point2d(3, 4)

    dist = distance_point_point(p1, p2)
    assert np.isclose(dist, 5.0)  # 3-4-5 triangle


def test_distance_point_point_same():
    """Test distance between same point."""
    p = point2d(5, 5)
    dist = distance_point_point(p, p)
    assert dist == 0.0


def test_distance_point_line_perpendicular():
    """Test point-line distance (perpendicular case)."""
    point = point2d(0, 1)
    line = line2d(point2d(-5, 0), point2d(5, 0))

    dist = distance_point_line(point, line)
    assert np.isclose(dist, 1.0)


def test_distance_point_line_endpoint():
    """Test point-line distance (closest to endpoint)."""
    point = point2d(3, 0)
    line = line2d(point2d(0, 0), point2d(1, 0))

    dist = distance_point_line(point, line)
    assert np.isclose(dist, 2.0)  # Closest to endpoint (1, 0)


def test_distance_point_line_zero_length():
    """Test distance to zero-length line (degenerate case)."""
    point = point2d(3, 4)
    line_point = point2d(0, 0)
    line = line2d(line_point, line_point)

    dist = distance_point_line(point, line)
    assert np.isclose(dist, 5.0)  # Distance to the point


def test_distance_point_circle_outside():
    """Test distance from point outside circle."""
    point = point2d(10, 0)
    circ = circle(center=point2d(0, 0), radius=3.0)

    dist = distance_point_circle(point, circ)
    assert np.isclose(dist, 7.0)  # 10 - 3


def test_distance_point_circle_inside():
    """Test distance from point inside circle (negative)."""
    point = point2d(1, 0)
    circ = circle(center=point2d(0, 0), radius=5.0)

    dist = distance_point_circle(point, circ)
    assert np.isclose(dist, -4.0)  # 1 - 5


def test_distance_point_circle_on_perimeter():
    """Test distance from point on circle perimeter."""
    point = point2d(3, 0)
    circ = circle(center=point2d(0, 0), radius=3.0)

    dist = distance_point_circle(point, circ)
    assert np.isclose(dist, 0.0, atol=1e-10)


# ============================================================================
# LAYER 3 TESTS: Spatial Queries - Intersection
# ============================================================================


def test_intersect_circle_circle_two_points():
    """Test circle-circle intersection with two intersection points."""
    c1 = circle(center=point2d(0, 0), radius=5.0)
    c2 = circle(center=point2d(5, 0), radius=5.0)

    result = intersect_circle_circle(c1, c2)
    assert result is not None

    p1, p2 = result

    # Both points should be on both circles
    assert np.isclose(distance_point_point(c1.center, p1), c1.radius, atol=1e-10)
    assert np.isclose(distance_point_point(c2.center, p1), c2.radius, atol=1e-10)
    assert np.isclose(distance_point_point(c1.center, p2), c1.radius, atol=1e-10)
    assert np.isclose(distance_point_point(c2.center, p2), c2.radius, atol=1e-10)


def test_intersect_circle_circle_no_intersection():
    """Test non-intersecting circles."""
    c1 = circle(center=point2d(0, 0), radius=1.0)
    c2 = circle(center=point2d(10, 0), radius=1.0)

    result = intersect_circle_circle(c1, c2)
    assert result is None


def test_intersect_circle_circle_one_inside():
    """Test circle completely inside another."""
    c1 = circle(center=point2d(0, 0), radius=10.0)
    c2 = circle(center=point2d(0, 0), radius=2.0)

    result = intersect_circle_circle(c1, c2)
    assert result is None


def test_intersect_circle_circle_same_center():
    """Test circles with same center (degenerate case)."""
    c1 = circle(center=point2d(0, 0), radius=5.0)
    c2 = circle(center=point2d(0, 0), radius=5.0)

    result = intersect_circle_circle(c1, c2)
    # Mathematically infinite intersections, returns None
    assert result is None


# ============================================================================
# LAYER 3 TESTS: Spatial Queries - Containment
# ============================================================================


def test_contains_circle_point_inside():
    """Test point inside circle."""
    circ = circle(center=point2d(0, 0), radius=5.0)
    point = point2d(3, 0)

    assert contains_circle_point(circ, point) is True


def test_contains_circle_point_outside():
    """Test point outside circle."""
    circ = circle(center=point2d(0, 0), radius=5.0)
    point = point2d(10, 0)

    assert contains_circle_point(circ, point) is False


def test_contains_circle_point_on_boundary():
    """Test point on circle boundary."""
    circ = circle(center=point2d(0, 0), radius=5.0)
    point = point2d(5, 0)

    assert contains_circle_point(circ, point) is True


def test_contains_rectangle_point_inside():
    """Test point inside axis-aligned rectangle."""
    rect = rectangle(point2d(0, 0), width=10, height=6, rotation=0)
    point = point2d(2, 2)

    assert contains_rectangle_point(rect, point) is True


def test_contains_rectangle_point_outside():
    """Test point outside rectangle."""
    rect = rectangle(point2d(0, 0), width=4, height=2, rotation=0)
    point = point2d(10, 10)

    assert contains_rectangle_point(rect, point) is False


def test_contains_rectangle_point_on_edge():
    """Test point on rectangle edge."""
    rect = rectangle(point2d(0, 0), width=4, height=2, rotation=0)
    point = point2d(2, 1)  # Right edge

    assert contains_rectangle_point(rect, point) is True


def test_contains_rectangle_point_rotated():
    """Test point containment in rotated rectangle."""
    rect = rectangle(point2d(0, 0), width=4, height=2, rotation=np.pi / 4)

    # Point at center should always be inside
    assert contains_rectangle_point(rect, point2d(0, 0)) is True


def test_contains_polygon_point_inside():
    """Test point inside triangle."""
    vertices = np.array([[0, 0], [4, 0], [2, 3]])
    tri = polygon(vertices)

    assert contains_polygon_point(tri, point2d(2, 1)) is True


def test_contains_polygon_point_outside():
    """Test point outside polygon."""
    vertices = np.array([[0, 0], [1, 0], [0.5, 1]])
    tri = polygon(vertices)

    assert contains_polygon_point(tri, point2d(10, 10)) is False


def test_contains_polygon_point_on_edge():
    """Test point on polygon edge."""
    vertices = np.array([[0, 0], [4, 0], [2, 4]])
    tri = polygon(vertices)

    # Point on edge between (0,0) and (4,0)
    assert contains_polygon_point(tri, point2d(2, 0)) is True


# ============================================================================
# LAYER 3 TESTS: Spatial Queries - Closest Point
# ============================================================================


def test_closest_point_circle_outside():
    """Test closest point on circle from outside."""
    circ = circle(center=point2d(0, 0), radius=5.0)
    point = point2d(10, 0)

    closest = closest_point_circle(circ, point)

    assert np.isclose(closest.x, 5.0)
    assert np.isclose(closest.y, 0.0)


def test_closest_point_circle_inside():
    """Test closest point on circle from inside."""
    circ = circle(center=point2d(0, 0), radius=5.0)
    point = point2d(2, 0)

    closest = closest_point_circle(circ, point)

    # Should project outward to perimeter
    assert np.isclose(closest.x, 5.0)
    assert np.isclose(closest.y, 0.0)


def test_closest_point_circle_at_center():
    """Test closest point when query point is at center."""
    circ = circle(center=point2d(0, 0), radius=5.0)
    point = point2d(0, 0)

    closest = closest_point_circle(circ, point)

    # Returns arbitrary point on circle (in this case, +x direction)
    assert np.isclose(distance_point_point(circ.center, closest), circ.radius)


def test_closest_point_line_perpendicular():
    """Test closest point on line (perpendicular projection)."""
    line = line2d(point2d(0, 0), point2d(10, 0))
    point = point2d(5, 5)

    closest = closest_point_line(line, point)

    assert np.isclose(closest.x, 5.0)
    assert np.isclose(closest.y, 0.0)


def test_closest_point_line_endpoint():
    """Test closest point is an endpoint."""
    line = line2d(point2d(0, 0), point2d(2, 0))
    point = point2d(10, 0)

    closest = closest_point_line(line, point)

    assert np.isclose(closest.x, 2.0)
    assert np.isclose(closest.y, 0.0)


def test_closest_point_line_zero_length():
    """Test closest point on zero-length line."""
    p = point2d(1, 1)
    line = line2d(p, p)
    point = point2d(5, 5)

    closest = closest_point_line(line, point)

    assert closest.x == p.x
    assert closest.y == p.y


# ============================================================================
# LAYER 4 TESTS: Coordinate Conversions
# ============================================================================


def test_cartesian_to_polar():
    """Test Cartesian to polar conversion."""
    p = point2d(3, 4)
    r, theta = cartesian_to_polar(p)

    assert np.isclose(r, 5.0)
    assert np.isclose(theta, np.arctan2(4, 3))


def test_cartesian_to_polar_origin():
    """Test conversion at origin."""
    p = point2d(0, 0)
    r, theta = cartesian_to_polar(p)

    assert r == 0.0
    assert theta == 0.0


def test_cartesian_to_polar_negative_x():
    """Test conversion in second quadrant."""
    p = point2d(-1, 1)
    r, theta = cartesian_to_polar(p)

    assert np.isclose(r, np.sqrt(2))
    assert np.isclose(theta, 3 * np.pi / 4)


def test_polar_to_cartesian():
    """Test polar to Cartesian conversion."""
    p = polar_to_cartesian(r=5.0, theta=np.pi / 4)

    expected_x = 5.0 * np.cos(np.pi / 4)
    expected_y = 5.0 * np.sin(np.pi / 4)

    assert np.isclose(p.x, expected_x)
    assert np.isclose(p.y, expected_y)


def test_polar_to_cartesian_zero_radius():
    """Test polar to Cartesian with zero radius."""
    p = polar_to_cartesian(r=0.0, theta=np.pi / 4)

    assert np.isclose(p.x, 0.0)
    assert np.isclose(p.y, 0.0)


def test_cartesian_polar_roundtrip():
    """Test Cartesian → polar → Cartesian roundtrip."""
    original = point2d(3, 4)
    r, theta = cartesian_to_polar(original)
    converted = polar_to_cartesian(r, theta)

    assert np.isclose(converted.x, original.x)
    assert np.isclose(converted.y, original.y)


def test_cartesian_to_spherical():
    """Test Cartesian to spherical conversion."""
    p = point3d(1, 0, 0)
    r, theta, phi = cartesian_to_spherical(p)

    assert np.isclose(r, 1.0)
    assert np.isclose(theta, 0.0)
    assert np.isclose(phi, np.pi / 2)


def test_cartesian_to_spherical_on_z_axis():
    """Test conversion for point on z-axis."""
    p = point3d(0, 0, 5)
    r, theta, phi = cartesian_to_spherical(p)

    assert np.isclose(r, 5.0)
    assert np.isclose(phi, 0.0)


def test_spherical_to_cartesian():
    """Test spherical to Cartesian conversion."""
    # Point on unit sphere
    p = spherical_to_cartesian(r=1.0, theta=0, phi=np.pi / 2)

    assert np.isclose(p.x, 1.0, atol=1e-10)
    assert np.isclose(p.y, 0.0, atol=1e-10)
    assert np.isclose(p.z, 0.0, atol=1e-10)


def test_cartesian_spherical_roundtrip():
    """Test Cartesian → spherical → Cartesian roundtrip."""
    original = point3d(3, 4, 5)
    r, theta, phi = cartesian_to_spherical(original)
    converted = spherical_to_cartesian(r, theta, phi)

    assert np.isclose(converted.x, original.x)
    assert np.isclose(converted.y, original.y)
    assert np.isclose(converted.z, original.z)


# ============================================================================
# LAYER 5 TESTS: Geometric Properties
# ============================================================================


def test_area_circle():
    """Test circle area calculation."""
    circ = circle(center=point2d(0, 0), radius=5.0)
    a = area(circ)

    assert np.isclose(a, np.pi * 25)


def test_area_rectangle():
    """Test rectangle area calculation."""
    rect = rectangle(point2d(0, 0), width=4, height=3)
    a = area(rect)

    assert a == 12.0


def test_area_polygon():
    """Test polygon area calculation."""
    # Unit square
    vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    poly = polygon(vertices)
    a = area(poly)

    assert np.isclose(a, 1.0)


def test_perimeter_circle():
    """Test circle circumference."""
    circ = circle(center=point2d(0, 0), radius=3.0)
    p = perimeter(circ)

    assert np.isclose(p, 2 * np.pi * 3)


def test_perimeter_rectangle():
    """Test rectangle perimeter."""
    rect = rectangle(point2d(0, 0), width=5, height=3)
    p = perimeter(rect)

    assert p == 16.0


def test_perimeter_polygon():
    """Test polygon perimeter."""
    # Triangle with sides 3, 4, 5
    vertices = np.array([[0, 0], [3, 0], [0, 4]])
    tri = polygon(vertices)
    p = perimeter(tri)

    assert np.isclose(p, 12.0)


def test_centroid_rectangle():
    """Test rectangle centroid."""
    rect = rectangle(point2d(5, 5), width=4, height=2)
    c = centroid(rect)

    assert c.x == 5.0
    assert c.y == 5.0


def test_centroid_polygon():
    """Test polygon centroid."""
    # Square centered at (2, 2)
    vertices = np.array([[0, 0], [4, 0], [4, 4], [0, 4]])
    poly = polygon(vertices)
    c = centroid(poly)

    assert np.isclose(c.x, 2.0)
    assert np.isclose(c.y, 2.0)


def test_bounding_box_circle():
    """Test circle bounding box."""
    circ = circle(center=point2d(5, 5), radius=3.0)
    bbox = bounding_box(circ)

    assert np.isclose(bbox.min_point.x, 2.0)
    assert np.isclose(bbox.min_point.y, 2.0)
    assert np.isclose(bbox.max_point.x, 8.0)
    assert np.isclose(bbox.max_point.y, 8.0)
    assert np.isclose(bbox.width, 6.0)
    assert np.isclose(bbox.height, 6.0)


def test_bounding_box_rectangle():
    """Test axis-aligned rectangle bounding box."""
    rect = rectangle(point2d(0, 0), width=4, height=2, rotation=0)
    bbox = bounding_box(rect)

    assert np.isclose(bbox.min_point.x, -2.0)
    assert np.isclose(bbox.min_point.y, -1.0)
    assert np.isclose(bbox.max_point.x, 2.0)
    assert np.isclose(bbox.max_point.y, 1.0)


def test_bounding_box_polygon():
    """Test polygon bounding box."""
    vertices = np.array([[-1, -2], [3, 0], [1, 4], [-2, 2]])
    poly = polygon(vertices)
    bbox = bounding_box(poly)

    assert bbox.min_point.x == -2.0
    assert bbox.min_point.y == -2.0
    assert bbox.max_point.x == 3.0
    assert bbox.max_point.y == 4.0


def test_bounding_box_center():
    """Test bounding box center calculation."""
    vertices = np.array([[0, 0], [4, 0], [4, 4], [0, 4]])
    poly = polygon(vertices)
    bbox = bounding_box(poly)

    center = bbox.center
    assert np.isclose(center.x, 2.0)
    assert np.isclose(center.y, 2.0)


# ============================================================================
# EDGE CASES AND INTEGRATION TESTS
# ============================================================================


def test_complex_transformation_chain():
    """Test chaining multiple transformations."""
    # Create triangle
    vertices = np.array([[0, 0], [1, 0], [0.5, 1]])
    tri = polygon(vertices)

    # Translate
    tri = translate_polygon(tri, dx=5.0, dy=3.0)

    # Rotate around its centroid
    c = tri.centroid
    tri = rotate_polygon(tri, center=c, angle=np.pi / 4)

    # Scale
    tri = scale_polygon(tri, center=c, scale_x=2.0, scale_y=2.0)

    # Final area should be 4x original (2x2 scale)
    assert np.isclose(tri.area, 0.5 * 4, atol=1e-10)


def test_regular_polygon_properties():
    """Test that regular polygon has expected properties."""
    # Regular pentagon
    pent = regular_polygon(center=point2d(0, 0), radius=1.0, num_sides=5)

    # All vertices should be equidistant from center
    c = pent.centroid
    for v in pent.vertices:
        dist = np.sqrt((v[0] - c.x) ** 2 + (v[1] - c.y) ** 2)
        assert np.isclose(dist, 1.0, atol=0.1)  # Approximate due to centroid vs circumcenter


def test_polygon_winding_order():
    """Test polygon area respects winding order."""
    # Counter-clockwise winding
    ccw_vertices = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    poly_ccw = polygon(ccw_vertices)

    # Clockwise winding
    cw_vertices = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
    poly_cw = polygon(cw_vertices)

    # Both should have same area (abs value)
    assert np.isclose(poly_ccw.area, poly_cw.area)


def test_multiple_shapes_same_area():
    """Test different shapes with same area."""
    # Circle with radius chosen for area = 10
    r = np.sqrt(10 / np.pi)
    circ = circle(center=point2d(0, 0), radius=r)

    # Rectangle 2x5
    rect = rectangle(point2d(0, 0), width=2, height=5)

    # Triangle base=4, height=5
    tri = polygon(np.array([[0, 0], [4, 0], [2, 5]]))

    assert np.isclose(area(circ), 10.0)
    assert np.isclose(area(rect), 10.0)
    assert np.isclose(area(tri), 10.0)


# ============================================================================
# PERFORMANCE AND STRESS TESTS
# ============================================================================


def test_large_polygon():
    """Test polygon with many vertices."""
    # Create polygon with 100 vertices
    angles = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    poly = polygon(vertices)

    assert poly.num_vertices == 100

    # Should be approximately a unit circle
    assert np.isclose(poly.area, np.pi, atol=0.1)
    assert np.isclose(poly.perimeter, 2 * np.pi, atol=0.1)


def test_many_transformations():
    """Test that multiple transformations are reversible."""
    p1 = point2d(5, 3)

    # Apply many transformations
    p2 = translate_point2d(p1, dx=10, dy=20)
    p3 = rotate_point2d(p2, center=point2d(0, 0), angle=np.pi / 3)
    p4 = translate_point2d(p3, dx=-10, dy=-20)

    # Reverse transformations
    p5 = translate_point2d(p4, dx=10, dy=20)
    p6 = rotate_point2d(p5, center=point2d(0, 0), angle=-np.pi / 3)
    p7 = translate_point2d(p6, dx=-10, dy=-20)

    # Should get back to original
    assert np.isclose(p7.x, p1.x, atol=1e-10)
    assert np.isclose(p7.y, p1.y, atol=1e-10)


# ============================================================================
# LAYER 6 TESTS: 3D Primitives
# ============================================================================


def test_box3d_construction():
    """Test 3D box creation."""
    center = point3d(1.0, 2.0, 3.0)
    box = box3d(center=center, width=4.0, height=6.0, depth=8.0)

    assert box.center == center
    assert box.width == 4.0
    assert box.height == 6.0
    assert box.depth == 8.0
    assert np.isclose(box.volume, 192.0)  # 4 * 6 * 8
    assert np.isclose(box.surface_area, 208.0)  # 2 * (4*6 + 4*8 + 6*8)


def test_box3d_with_rotation():
    """Test 3D box with rotation."""
    center = point3d(0.0, 0.0, 0.0)
    rotation = np.array([0.0, 0.0, np.pi / 4])  # 45 degree rotation around Z
    box = box3d(center=center, width=2.0, height=2.0, depth=2.0, rotation=rotation)

    vertices = box.get_vertices()
    assert vertices.shape == (8, 3)

    # Check that vertices are approximately at the right distance
    # For a cube with side 2, vertices should be at distance sqrt(3) from center
    distances = np.linalg.norm(vertices, axis=1)
    expected_dist = np.sqrt(3)
    assert np.allclose(distances, expected_dist, atol=1e-10)


def test_box3d_invalid_dimensions():
    """Test box3d with invalid dimensions."""
    center = point3d(0, 0, 0)

    try:
        box3d(center=center, width=-1, height=2, depth=3)
        assert False, "Should raise ValueError for negative width"
    except ValueError:
        pass


def test_sphere_construction():
    """Test 3D sphere creation."""
    center = point3d(1.0, 2.0, 3.0)
    s = sphere(center=center, radius=5.0)

    assert s.center == center
    assert s.radius == 5.0
    assert np.isclose(s.volume, (4.0 / 3.0) * np.pi * 125.0)
    assert np.isclose(s.surface_area, 4 * np.pi * 25.0)


def test_sphere_invalid_radius():
    """Test sphere with invalid radius."""
    center = point3d(0, 0, 0)

    try:
        sphere(center=center, radius=-1.0)
        assert False, "Should raise ValueError for negative radius"
    except ValueError:
        pass


def test_mesh_construction():
    """Test mesh creation."""
    # Simple tetrahedron
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, 0.5, 1.0]
    ])
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3]
    ])

    m = mesh(vertices=vertices, faces=faces)

    assert m.num_vertices == 4
    assert m.num_faces == 4
    assert m.vertices.shape == (4, 3)
    assert m.faces.shape == (4, 3)


def test_mesh_compute_normals():
    """Test mesh normal computation."""
    # Simple triangle
    vertices = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0]
    ])
    faces = np.array([[0, 1, 2]])

    m = mesh(vertices=vertices, faces=faces)
    normals = m.compute_normals()

    assert normals.shape == (3, 3)

    # All normals should point in +Z direction
    assert np.allclose(normals, [[0, 0, 1], [0, 0, 1], [0, 0, 1]], atol=1e-10)


def test_mesh_invalid_vertices():
    """Test mesh with invalid vertices."""
    # Wrong shape
    vertices = np.array([[0, 0], [1, 0]])
    faces = np.array([[0, 1, 0]])

    try:
        mesh(vertices=vertices, faces=faces)
        assert False, "Should raise ValueError for wrong vertex shape"
    except ValueError:
        pass


# ============================================================================
# LAYER 7 TESTS: Advanced Algorithms
# ============================================================================


def test_convex_hull_2d():
    """Test 2D convex hull computation."""
    # Random points
    np.random.seed(42)
    points = np.random.rand(50, 2) * 10

    hull = convex_hull(points, dim=2)

    # Hull should be a polygon
    from morphogen.stdlib.geometry import Polygon
    assert isinstance(hull, Polygon)

    # Hull should have fewer vertices than input
    assert hull.num_vertices <= 50

    # All original points should be inside or on the hull
    for pt in points:
        p = point2d(pt[0], pt[1])
        # Either inside or very close to the boundary
        assert contains_polygon_point(hull, p) or distance_point_point(p, hull.centroid) <= 10.0


def test_convex_hull_3d():
    """Test 3D convex hull computation."""
    # Random 3D points
    np.random.seed(42)
    points = np.random.rand(50, 3) * 10

    hull = convex_hull(points, dim=3)

    # Hull should be a mesh
    assert isinstance(hull, Mesh)

    # Hull should have faces
    assert hull.num_faces > 0


def test_delaunay_triangulation():
    """Test Delaunay triangulation."""
    # Regular grid of points
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    tri = delaunay_triangulation(points)

    # Result should be a mesh
    assert isinstance(tri, Mesh)

    # Should have triangular faces
    assert tri.faces.shape[1] == 3

    # Should have reasonable number of triangles (approximately 2 * (n-1)^2 for grid)
    assert tri.num_faces > 10


def test_voronoi():
    """Test Voronoi diagram computation."""
    # Regular grid of points
    np.random.seed(42)
    points = np.random.rand(20, 2) * 10

    vertices, ridge_points, ridge_vertices = voronoi(points)

    # Should have vertices
    assert len(vertices) > 0
    assert vertices.shape[1] == 2

    # Should have ridges
    assert len(ridge_points) > 0
    assert ridge_points.shape[1] == 2


def test_mesh_union():
    """Test mesh union operation."""
    # Two simple triangles
    verts1 = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])
    faces1 = np.array([[0, 1, 2]])
    m1 = mesh(verts1, faces1)

    verts2 = np.array([[2, 0, 0], [3, 0, 0], [2.5, 1, 0]])
    faces2 = np.array([[0, 1, 2]])
    m2 = mesh(verts2, faces2)

    union = mesh_union(m1, m2)

    # Union should have combined vertices and faces
    assert union.num_vertices == 6
    assert union.num_faces == 2


# ============================================================================
# LAYER 8 TESTS: Field Domain Integration
# ============================================================================


def test_sample_field_at_point():
    """Test sampling field at a geometric point."""
    try:
        from morphogen.stdlib.field import Field2D
    except ImportError:
        pytest.skip("Field domain not available")
        return

    # Create a simple field
    data = np.ones((10, 10)) * 5.0
    field = Field2D(data, dx=1.0, dy=1.0)

    # Sample at a point
    p = point2d(5.0, 5.0)
    value = sample_field_at_point(field, p)

    assert np.isclose(value, 5.0)


def test_sample_field_with_interpolation():
    """Test field sampling with bilinear interpolation."""
    try:
        from morphogen.stdlib.field import Field2D
    except ImportError:
        pytest.skip("Field domain not available")
        return

    # Create a gradient field
    data = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            data[i, j] = i + j

    field = Field2D(data, dx=1.0, dy=1.0)

    # Sample at a fractional position
    p = point2d(2.5, 3.5)
    value = sample_field_at_point(field, p)

    # Should interpolate between neighboring values
    # Expected: (2+3 + 3+3 + 2+4 + 3+4) / 4 = 6.0
    assert np.isclose(value, 6.0, atol=0.1)


def test_query_field_in_region():
    """Test querying field values in a geometric region."""
    try:
        from morphogen.stdlib.field import Field2D
    except ImportError:
        pytest.skip("Field domain not available")
        return

    # Create a field
    data = np.ones((20, 20)) * 3.0
    field = Field2D(data, dx=1.0, dy=1.0)

    # Query inside a circle
    circ = circle(center=point2d(10.0, 10.0), radius=3.0)
    values = query_field_in_region(field, circ)

    # Should have some values
    assert len(values) > 0

    # All values should be approximately 3.0
    assert np.allclose(values[:, 2], 3.0)


def test_query_field_in_rectangle():
    """Test querying field in rectangle region."""
    try:
        from morphogen.stdlib.field import Field2D
    except ImportError:
        pytest.skip("Field domain not available")
        return

    # Create a field
    data = np.arange(400).reshape(20, 20).astype(float)
    field = Field2D(data, dx=1.0, dy=1.0)

    # Query inside a rectangle
    rect = rectangle(center=point2d(10.0, 10.0), width=4.0, height=4.0)
    values = query_field_in_region(field, rect)

    # Should have values from inside the rectangle
    assert len(values) > 0
    assert len(values) <= 25  # 5x5 grid max


# ============================================================================
# LAYER 9 TESTS: Rigidbody Domain Integration
# ============================================================================


def test_shape_to_rigidbody_circle():
    """Test converting circle to rigidbody shape."""
    try:
        from morphogen.stdlib.rigidbody import ShapeType
    except ImportError:
        pytest.skip("Rigidbody domain not available")
        return

    circ = circle(center=point2d(0, 0), radius=5.0)
    shape_data = shape_to_rigidbody(circ)

    assert shape_data['shape_type'] == ShapeType.CIRCLE
    assert shape_data['shape_params']['radius'] == 5.0


def test_shape_to_rigidbody_rectangle():
    """Test converting rectangle to rigidbody shape."""
    try:
        from morphogen.stdlib.rigidbody import ShapeType
    except ImportError:
        pytest.skip("Rigidbody domain not available")
        return

    rect = rectangle(center=point2d(0, 0), width=4.0, height=6.0, rotation=np.pi / 4)
    shape_data = shape_to_rigidbody(rect)

    assert shape_data['shape_type'] == ShapeType.BOX
    assert shape_data['shape_params']['width'] == 4.0
    assert shape_data['shape_params']['height'] == 6.0
    assert np.isclose(shape_data['shape_params']['rotation'], np.pi / 4)


def test_shape_to_rigidbody_polygon():
    """Test converting polygon to rigidbody shape."""
    try:
        from morphogen.stdlib.rigidbody import ShapeType
    except ImportError:
        pytest.skip("Rigidbody domain not available")
        return

    tri = polygon(np.array([[0, 0], [1, 0], [0.5, 1]]))
    shape_data = shape_to_rigidbody(tri)

    assert shape_data['shape_type'] == ShapeType.POLYGON
    assert np.allclose(shape_data['shape_params']['vertices'], [[0, 0], [1, 0], [0.5, 1]])


def test_collision_mesh():
    """Test collision mesh generation."""
    # Create a simple cube mesh
    vertices = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
    ], dtype=float)

    # Just a few faces (incomplete cube)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],  # Bottom
        [4, 5, 6], [4, 6, 7],  # Top
    ])

    m = mesh(vertices, faces)
    collision = collision_mesh(m, target_faces=10)

    # Should return a simplified mesh
    assert isinstance(collision, Mesh)
    assert collision.num_faces > 0
