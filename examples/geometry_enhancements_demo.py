"""Geometry Domain Enhancements Demo.

This example demonstrates the new geometry domain enhancements:
1. 3D primitives (boxes, spheres, meshes)
2. Advanced algorithms (convex hull, Delaunay, Voronoi, mesh booleans)
3. Field domain integration (spatial field sampling)
4. Rigidbody domain integration (collision shapes)
"""

import numpy as np

from morphogen.stdlib.geometry import (
    # 3D primitives
    point3d,
    box3d,
    sphere,
    mesh,
    # 2D primitives (for comparison)
    point2d,
    circle,
    rectangle,
    polygon,
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


def demo_3d_primitives():
    """Demonstrate 3D primitive creation and properties."""
    print("=" * 60)
    print("3D PRIMITIVES DEMO")
    print("=" * 60)

    # Create a 3D box
    print("\n1. Box3D:")
    center = point3d(0, 0, 0)
    box = box3d(center=center, width=2.0, height=3.0, depth=4.0)
    print(f"   {box}")
    print(f"   Volume: {box.volume:.2f}")
    print(f"   Surface Area: {box.surface_area:.2f}")
    print(f"   Vertices shape: {box.get_vertices().shape}")

    # Create a rotated box
    print("\n2. Rotated Box3D:")
    rotation = np.array([0, 0, np.pi / 4])  # 45° around Z-axis
    rotated_box = box3d(center=center, width=1.0, height=1.0, depth=1.0, rotation=rotation)
    print(f"   {rotated_box}")
    print(f"   Rotation: {rotation}")

    # Create a sphere
    print("\n3. Sphere:")
    s = sphere(center=point3d(5, 0, 0), radius=3.0)
    print(f"   {s}")
    print(f"   Volume: {s.volume:.2f}")
    print(f"   Surface Area: {s.surface_area:.2f}")

    # Create a mesh (tetrahedron)
    print("\n4. Mesh (Tetrahedron):")
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
    tetrahedron = mesh(vertices=vertices, faces=faces)
    print(f"   {tetrahedron}")
    print(f"   Normals computed: {tetrahedron.compute_normals().shape}")


def demo_advanced_algorithms():
    """Demonstrate advanced computational geometry algorithms."""
    print("\n" + "=" * 60)
    print("ADVANCED ALGORITHMS DEMO")
    print("=" * 60)

    # 2D Convex Hull
    print("\n1. 2D Convex Hull:")
    np.random.seed(42)
    points_2d = np.random.rand(50, 2) * 10
    hull_2d = convex_hull(points_2d, dim=2)
    print(f"   Input points: {len(points_2d)}")
    print(f"   Hull vertices: {hull_2d.num_vertices}")
    print(f"   Hull area: {hull_2d.area:.2f}")

    # 3D Convex Hull
    print("\n2. 3D Convex Hull:")
    points_3d = np.random.rand(100, 3) * 10
    hull_3d = convex_hull(points_3d, dim=3)
    print(f"   Input points: {len(points_3d)}")
    print(f"   Hull faces: {hull_3d.num_faces}")
    print(f"   Hull vertices: {hull_3d.num_vertices}")

    # Delaunay Triangulation
    print("\n3. Delaunay Triangulation:")
    grid_points = np.random.rand(25, 2) * 10
    triangulation = delaunay_triangulation(grid_points)
    print(f"   Input points: {len(grid_points)}")
    print(f"   Triangles: {triangulation.num_faces}")

    # Voronoi Diagram
    print("\n4. Voronoi Diagram:")
    vor_points = np.random.rand(20, 2) * 10
    vertices, ridge_points, ridge_vertices = voronoi(vor_points)
    print(f"   Input points: {len(vor_points)}")
    print(f"   Voronoi vertices: {len(vertices)}")
    print(f"   Ridges: {len(ridge_points)}")

    # Mesh Union
    print("\n5. Mesh Boolean Operations (Union):")
    verts1 = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])
    faces1 = np.array([[0, 1, 2]])
    mesh1 = mesh(verts1, faces1)

    verts2 = np.array([[2, 0, 0], [3, 0, 0], [2.5, 1, 0]])
    faces2 = np.array([[0, 1, 2]])
    mesh2 = mesh(verts2, faces2)

    union = mesh_union(mesh1, mesh2)
    print(f"   Mesh 1: {mesh1.num_vertices} vertices, {mesh1.num_faces} faces")
    print(f"   Mesh 2: {mesh2.num_vertices} vertices, {mesh2.num_faces} faces")
    print(f"   Union: {union.num_vertices} vertices, {union.num_faces} faces")


def demo_field_integration():
    """Demonstrate field domain integration."""
    print("\n" + "=" * 60)
    print("FIELD DOMAIN INTEGRATION DEMO")
    print("=" * 60)

    try:
        from morphogen.stdlib.field import Field2D

        # Create a field with a gradient
        print("\n1. Creating a 2D field with gradient:")
        data = np.zeros((50, 50))
        for i in range(50):
            for j in range(50):
                data[i, j] = i + j
        field = Field2D(data, dx=0.2, dy=0.2)
        print(f"   Field shape: {field.shape}")
        print(f"   Grid spacing: dx={field.dx}, dy={field.dy}")

        # Sample field at a point
        print("\n2. Sampling field at a point:")
        p = point2d(5.0, 5.0)
        value = sample_field_at_point(field, p)
        print(f"   Point: {p}")
        print(f"   Sampled value: {value:.2f}")

        # Query field in a circular region
        print("\n3. Querying field in a circular region:")
        circ = circle(center=point2d(5.0, 5.0), radius=2.0)
        values = query_field_in_region(field, circ)
        print(f"   Circle center: {circ.center}")
        print(f"   Circle radius: {circ.radius}")
        print(f"   Points sampled: {len(values)}")
        if len(values) > 0:
            print(f"   Value range: [{values[:, 2].min():.2f}, {values[:, 2].max():.2f}]")

        # Query field in a rectangular region
        print("\n4. Querying field in a rectangular region:")
        rect = rectangle(center=point2d(6.0, 6.0), width=3.0, height=3.0)
        rect_values = query_field_in_region(field, rect)
        print(f"   Rectangle center: {rect.center}")
        print(f"   Rectangle size: {rect.width}x{rect.height}")
        print(f"   Points sampled: {len(rect_values)}")

    except ImportError:
        print("   [Field domain not available - skipping demo]")


def demo_rigidbody_integration():
    """Demonstrate rigidbody domain integration."""
    print("\n" + "=" * 60)
    print("RIGIDBODY DOMAIN INTEGRATION DEMO")
    print("=" * 60)

    try:
        from morphogen.stdlib.rigidbody import ShapeType, RigidBody2D

        # Convert circle to rigidbody shape
        print("\n1. Converting Circle to RigidBody:")
        circ = circle(center=point2d(0, 0), radius=5.0)
        shape_data = shape_to_rigidbody(circ)
        print(f"   Geometry: {circ}")
        print(f"   RigidBody shape type: {shape_data['shape_type']}")
        print(f"   RigidBody params: {shape_data['shape_params']}")

        # Create a rigidbody from geometric circle
        body = RigidBody2D(
            position=np.array([0.0, 0.0]),
            mass=10.0,
            shape_type=shape_data['shape_type'],
            shape_params=shape_data['shape_params']
        )
        print(f"   Created RigidBody: mass={body.mass}, radius={body.shape_params['radius']}")

        # Convert rectangle to rigidbody shape
        print("\n2. Converting Rectangle to RigidBody:")
        rect = rectangle(center=point2d(5, 5), width=4.0, height=6.0, rotation=np.pi / 6)
        rect_shape_data = shape_to_rigidbody(rect)
        print(f"   Geometry: {rect}")
        print(f"   RigidBody shape type: {rect_shape_data['shape_type']}")
        print(f"   RigidBody params: {rect_shape_data['shape_params']}")

        # Convert polygon to rigidbody shape
        print("\n3. Converting Polygon to RigidBody:")
        tri = polygon(np.array([[0, 0], [2, 0], [1, 2]]))
        tri_shape_data = shape_to_rigidbody(tri)
        print(f"   Geometry: {tri}")
        print(f"   RigidBody shape type: {tri_shape_data['shape_type']}")
        print(f"   RigidBody vertices: {len(tri_shape_data['shape_params']['vertices'])}")

        # Collision mesh generation
        print("\n4. Generating Collision Mesh:")
        # Create a high-poly mesh (cube subdivided)
        cube_verts = np.random.rand(100, 3) * 2 - 1  # Random points in cube
        hull = convex_hull(cube_verts, dim=3)
        collision = collision_mesh(hull, target_faces=20)
        print(f"   Original mesh: {hull.num_faces} faces")
        print(f"   Collision mesh: {collision.num_faces} faces")

    except ImportError:
        print("   [RigidBody domain not available - skipping demo]")


def main():
    """Run all demos."""
    print("\n" + "#" * 60)
    print("# GEOMETRY DOMAIN ENHANCEMENTS DEMO")
    print("# Showcasing 3D primitives, advanced algorithms, and")
    print("# cross-domain integration features")
    print("#" * 60)

    demo_3d_primitives()
    demo_advanced_algorithms()
    demo_field_integration()
    demo_rigidbody_integration()

    print("\n" + "#" * 60)
    print("# DEMO COMPLETE")
    print("#" * 60)
    print("\nAll geometry enhancements are now production-ready!")
    print("\nKey features added:")
    print("  ✓ 3D primitives: Box3D, Sphere, Mesh")
    print("  ✓ Advanced algorithms: Convex Hull, Delaunay, Voronoi, Mesh Booleans")
    print("  ✓ Field integration: Spatial sampling & region queries")
    print("  ✓ RigidBody integration: Collision shape conversion")
    print()


if __name__ == "__main__":
    main()
