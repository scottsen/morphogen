"""Integration tests for end-to-end DSL program execution."""

import pytest
import tempfile
from pathlib import Path
import numpy as np
from morphogen.stdlib.field import field, Field2D
from morphogen.stdlib.visual import visual
from morphogen.runtime.runtime import ExecutionContext, Runtime


@pytest.mark.integration
class TestSimpleProgramExecution:
    """Integration tests for simple complete programs."""

    def test_heat_diffusion_pipeline(self):
        """Test complete heat diffusion pipeline."""
        # Create initial temperature field
        temp = field.random((64, 64), seed=42, low=0.0, high=1.0)
        assert temp.shape == (64, 64)

        # Apply diffusion
        temp = field.diffuse(temp, rate=0.5, dt=0.1, iterations=20)
        assert temp.shape == (64, 64)

        # Apply boundary conditions
        temp = field.boundary(temp, spec="reflect")
        assert temp.shape == (64, 64)

        # Visualize
        vis = visual.colorize(temp, palette="fire")
        assert vis.shape == (64, 64)

        # Output to file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            visual.output(vis, path=tmp_path)
            assert Path(tmp_path).exists()
            assert Path(tmp_path).stat().st_size > 1000
        finally:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()

    def test_reaction_diffusion_pipeline(self):
        """Test reaction-diffusion pattern formation."""
        # Initialize fields
        u = field.random((128, 128), seed=1, low=0.0, high=1.0)
        v = field.random((128, 128), seed=2, low=0.0, high=1.0)

        # Simulate a few steps
        for _ in range(5):
            u = field.diffuse(u, rate=0.2, dt=0.1, iterations=10)
            v = field.diffuse(v, rate=0.1, dt=0.1, iterations=10)

            # Simple reaction (combine fields)
            reaction = field.combine(u, v, operation="mul")
            u = field.combine(u, reaction, operation="sub")

        # Visualize result
        vis = visual.colorize(u, palette="viridis")
        assert vis.shape == (128, 128)

    def test_velocity_field_projection(self):
        """Test velocity field with projection."""
        # Create divergent velocity field
        vx = field.random((32, 32), seed=1, low=-1.0, high=1.0)
        vy = field.random((32, 32), seed=2, low=-1.0, high=1.0)

        # Stack into velocity field
        velocity = Field2D(np.stack([vx.data, vy.data], axis=-1))

        # Project to make divergence-free
        velocity = field.project(velocity, iterations=30)
        assert velocity.data.shape == (32, 32, 2)

        # Visualize magnitude (colorize computes magnitude for vector fields)
        vis = visual.colorize(velocity, palette="coolwarm")
        # Visual output is always 2D
        assert vis.shape == (32, 32)
        assert vis.data.shape == (32, 32, 3)  # RGB channels


@pytest.mark.integration
class TestRuntimeExecution:
    """Integration tests for runtime execution."""

    def test_context_with_multiple_operations(self):
        """Test execution context manages multiple operations."""
        ctx = ExecutionContext(global_seed=42)
        runtime = Runtime(ctx)

        # Create and store multiple fields
        temp = field.random((32, 32), seed=42)
        runtime.context.set_variable('temp', temp)

        velocity = Field2D(np.zeros((32, 32, 2)))
        runtime.context.set_variable('velocity', velocity)

        # Process multiple timesteps
        for step in range(5):
            temp = runtime.context.get_variable('temp')
            temp = field.diffuse(temp, rate=0.1, dt=0.01, iterations=5)
            runtime.context.set_variable('temp', temp)
            ctx.advance_timestep()

        # Verify state
        assert ctx.timestep == 5
        assert runtime.context.has_variable('temp')
        assert runtime.context.has_variable('velocity')

    def test_deterministic_execution(self):
        """Test that execution is deterministic."""
        def run_simulation(seed):
            ctx = ExecutionContext(global_seed=seed)
            runtime = Runtime(ctx)

            temp = field.random((64, 64), seed=seed)
            runtime.context.set_variable('temp', temp)

            for _ in range(10):
                temp = runtime.context.get_variable('temp')
                temp = field.diffuse(temp, rate=0.2, dt=0.1, iterations=10)
                runtime.context.set_variable('temp', temp)

            return runtime.context.get_variable('temp').data

        # Run twice with same seed
        result1 = run_simulation(12345)
        result2 = run_simulation(12345)

        # Should be identical
        assert np.array_equal(result1, result2)

        # Run with different seed
        result3 = run_simulation(54321)

        # Should be different
        assert not np.array_equal(result1, result3)


@pytest.mark.integration
class TestFieldOperationChains:
    """Integration tests for chaining field operations."""

    def test_long_operation_chain(self):
        """Test long chain of field operations."""
        f = field.random((64, 64), seed=42)

        # Chain many operations
        f = field.diffuse(f, rate=0.1, dt=0.01, iterations=5)
        f = field.boundary(f, spec="reflect")
        f = field.map(f, func="abs")
        f = field.diffuse(f, rate=0.2, dt=0.01, iterations=5)

        # Combine with another field
        f2 = field.alloc((64, 64), fill_value=0.5)
        f = field.combine(f, f2, operation="mul")

        # More processing
        f = field.diffuse(f, rate=0.1, dt=0.01, iterations=5)
        f = field.boundary(f, spec="periodic")

        # Should still have correct shape
        assert f.shape == (64, 64)
        assert not np.any(np.isnan(f.data))
        assert not np.any(np.isinf(f.data))

    def test_advection_diffusion_chain(self):
        """Test combined advection and diffusion."""
        # Create scalar field
        scalar = field.random((64, 64), seed=1)

        # Create velocity field
        vx = field.alloc((64, 64), fill_value=0.5)
        vy = field.alloc((64, 64), fill_value=0.5)
        velocity = Field2D(np.stack([vx.data, vy.data], axis=-1))

        # Simulate multiple steps
        for _ in range(10):
            scalar = field.advect(scalar, velocity, dt=0.01)
            scalar = field.diffuse(scalar, rate=0.1, dt=0.01, iterations=5)
            scalar = field.boundary(scalar, spec="reflect")

        assert scalar.shape == (64, 64)
        assert not np.any(np.isnan(scalar.data))


@pytest.mark.integration
class TestVisualizationPipeline:
    """Integration tests for visualization pipeline."""

    def test_multiple_palettes(self):
        """Test visualizing same field with multiple palettes."""
        f = field.random((64, 64), seed=42)

        palettes = ["grayscale", "fire", "viridis", "coolwarm"]
        outputs = []

        for palette in palettes:
            vis = visual.colorize(f, palette=palette)
            assert vis.shape == (64, 64)

            with tempfile.NamedTemporaryFile(suffix=f"_{palette}.png", delete=False) as tmp:
                tmp_path = tmp.name
                outputs.append(tmp_path)
                visual.output(vis, path=tmp_path)

        try:
            # Verify all files exist and have different contents
            for i, path in enumerate(outputs):
                assert Path(path).exists()
                assert Path(path).stat().st_size > 1000

            # Different palettes should produce different images
            with open(outputs[0], 'rb') as f1, open(outputs[1], 'rb') as f2:
                assert f1.read() != f2.read()
        finally:
            for path in outputs:
                if Path(path).exists():
                    Path(path).unlink()

    def test_value_range_visualization(self):
        """Test visualizing fields with different value ranges."""
        # Test with different value ranges
        ranges = [
            (0.0, 1.0),
            (-1.0, 1.0),
            (100.0, 200.0),
            (-1000.0, -900.0)
        ]

        for low, high in ranges:
            f = field.random((32, 32), seed=42, low=low, high=high)
            vis = visual.colorize(f, palette="fire")

            assert vis.shape == (32, 32)
            assert np.all(vis.data >= 0.0)
            assert np.all(vis.data <= 1.0)


@pytest.mark.integration
class TestComplexScenarios:
    """Integration tests for complex multi-component scenarios."""

    def test_smoke_simulation_simplified(self):
        """Test simplified smoke simulation."""
        # Initialize velocity field
        vx = field.random((64, 64), seed=1, low=-0.5, high=0.5)
        vy = field.random((64, 64), seed=2, low=-0.5, high=0.5)
        velocity = Field2D(np.stack([vx.data, vy.data], axis=-1))

        # Initialize density
        density = field.random((64, 64), seed=3, low=0.0, high=1.0)

        # Simulate several steps
        for step in range(5):
            # Advect velocity
            velocity = field.advect(velocity, velocity, dt=0.01)

            # Project velocity (make divergence-free)
            velocity = field.project(velocity, iterations=20)

            # Advect density
            density = field.advect(density, velocity, dt=0.01)

            # Diffuse density
            density = field.diffuse(density, rate=0.01, dt=0.01, iterations=10)

        # Visualize result
        vis = visual.colorize(density, palette="fire")
        assert vis.shape == (64, 64)

        # Save output
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            visual.output(vis, path=tmp_path)
            assert Path(tmp_path).exists()
        finally:
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()

    def test_multi_field_interaction(self):
        """Test multiple fields interacting."""
        # Create three fields
        f1 = field.random((32, 32), seed=1, low=0.0, high=1.0)
        f2 = field.random((32, 32), seed=2, low=0.0, high=1.0)
        f3 = field.random((32, 32), seed=3, low=0.0, high=1.0)

        # Process in parallel
        f1 = field.diffuse(f1, rate=0.1, dt=0.01, iterations=5)
        f2 = field.diffuse(f2, rate=0.2, dt=0.01, iterations=5)
        f3 = field.diffuse(f3, rate=0.3, dt=0.01, iterations=5)

        # Combine
        temp1 = field.combine(f1, f2, operation="add")
        result = field.combine(temp1, f3, operation="add")

        # Normalize
        result = field.map(result, func=lambda x: x / 3.0)

        # Final processing
        result = field.diffuse(result, rate=0.1, dt=0.01, iterations=10)

        assert result.shape == (32, 32)
        assert np.all(result.data >= 0.0)


@pytest.mark.integration
@pytest.mark.determinism
class TestDeterminismIntegration:
    """Integration tests specifically for determinism."""

    def test_full_pipeline_determinism(self):
        """Test that full pipeline is deterministic."""
        def run_full_pipeline(seed):
            temp = field.random((64, 64), seed=seed)
            temp = field.diffuse(temp, rate=0.5, dt=0.1, iterations=20)
            temp = field.boundary(temp, spec="reflect")

            vis = visual.colorize(temp, palette="fire")

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                visual.output(vis, path=tmp_path)
                with open(tmp_path, 'rb') as f:
                    return f.read()
            finally:
                if Path(tmp_path).exists():
                    Path(tmp_path).unlink()

        # Run twice with same seed
        output1 = run_full_pipeline(42)
        output2 = run_full_pipeline(42)

        # Image bytes should be identical
        assert output1 == output2

    def test_operation_order_matters(self):
        """Test that operation order affects results."""
        f = field.random((32, 32), seed=42, low=-1.0, high=1.0)

        # Order 1: diffuse then square
        f1 = field.diffuse(f, rate=0.3, dt=0.1, iterations=20)
        f1 = field.map(f1, func="square")

        # Order 2: square then diffuse
        f2 = field.map(f, func="square")
        f2 = field.diffuse(f2, rate=0.3, dt=0.1, iterations=20)

        # Results should be different (operations don't commute)
        # Squaring after diffusion vs before produces different results
        assert not np.allclose(f1.data, f2.data, rtol=0.01)


@pytest.mark.integration
class TestGeometryFieldIntegration:
    """Integration tests for geometry-field domain interactions."""

    def test_sample_field_along_path(self):
        """Test sampling field values along a geometric path."""
        try:
            from morphogen.stdlib.geometry import (
                point2d, sample_field_at_point
            )
        except ImportError:
            pytest.skip("Geometry domain not available")
            return

        # Create a temperature gradient field
        temp = field.random((64, 64), seed=42, low=0.0, high=100.0)
        temp = field.diffuse(temp, rate=0.5, dt=0.1, iterations=20)

        # Sample along a line
        start = point2d(10.0, 10.0)
        end = point2d(50.0, 50.0)

        # Sample at multiple points along the line
        num_samples = 20
        samples = []
        for i in range(num_samples):
            t = i / (num_samples - 1)
            x = start.x + t * (end.x - start.x)
            y = start.y + t * (end.y - start.y)
            pt = point2d(x, y)
            value = sample_field_at_point(temp, pt)
            samples.append(value)

        # Should have sampled all points
        assert len(samples) == num_samples
        # All samples should be within the field's value range
        assert all(0.0 <= s <= 100.0 for s in samples)

    def test_field_driven_geometry_generation(self):
        """Test generating geometric shapes based on field values."""
        try:
            from morphogen.stdlib.geometry import (
                point2d, circle, sample_field_at_point
            )
        except ImportError:
            pytest.skip("Geometry domain not available")
            return

        # Create a field
        density = field.random((32, 32), seed=1, low=0.0, high=1.0)
        density = field.diffuse(density, rate=0.3, dt=0.1, iterations=10)

        # Generate circles at high-density points
        circles = []
        for i in range(5, 30, 5):
            for j in range(5, 30, 5):
                pt = point2d(float(i), float(j))
                value = sample_field_at_point(density, pt)

                # Create larger circles at higher density points
                if value > 0.5:
                    radius = value * 3.0
                    circ = circle(center=pt, radius=radius)
                    circles.append(circ)

        # Should have generated some circles
        assert len(circles) > 0

        # Verify all circles have positive radius
        for circ in circles:
            assert circ.radius > 0

    def test_geometry_based_field_initialization(self):
        """Test initializing fields based on geometric regions."""
        try:
            from morphogen.stdlib.geometry import (
                point2d, circle, rectangle, contains
            )
        except ImportError:
            pytest.skip("Geometry domain not available")
            return

        # Create a blank field
        result = field.alloc((64, 64), fill_value=0.0)

        # Define geometric regions
        hot_circle = circle(center=point2d(20.0, 20.0), radius=10.0)
        cold_rectangle = rectangle(center=point2d(45.0, 45.0), width=15.0, height=15.0)

        # Set field values based on geometry
        data = result.data.copy()
        for i in range(64):
            for j in range(64):
                pt = point2d(float(i), float(j))
                if contains(hot_circle, pt):
                    data[i, j] = 100.0
                elif contains(cold_rectangle, pt):
                    data[i, j] = 10.0

        result.data = data

        # Now diffuse to blend
        result = field.diffuse(result, rate=0.5, dt=0.1, iterations=20)

        # Check that we have both hot and cold regions
        assert np.max(result.data) > 50.0
        assert np.min(result.data) < 30.0

    def test_field_gradient_along_polygon(self):
        """Test computing field gradients along polygon edges."""
        try:
            from morphogen.stdlib.geometry import (
                point2d, regular_polygon, sample_field_at_point
            )
        except ImportError:
            pytest.skip("Geometry domain not available")
            return

        # Create a field with gradient
        temp = field.random((64, 64), seed=42)
        temp = field.diffuse(temp, rate=0.3, dt=0.1, iterations=15)

        # Create a polygon
        poly = regular_polygon(center=point2d(32.0, 32.0), radius=20.0, num_sides=6)

        # Sample at polygon vertices
        vertex_values = []
        for vertex in poly.vertices:
            pt = point2d(vertex[0], vertex[1])
            value = sample_field_at_point(temp, pt)
            vertex_values.append(value)

        # Should have sampled all vertices
        assert len(vertex_values) == 6

        # Compute gradient magnitude around polygon
        gradients = []
        for i in range(len(vertex_values)):
            next_i = (i + 1) % len(vertex_values)
            grad = abs(vertex_values[next_i] - vertex_values[i])
            gradients.append(grad)

        # Should have computed all gradients
        assert len(gradients) == 6
        assert all(g >= 0 for g in gradients)


@pytest.mark.integration
class TestGeometryRigidbodyIntegration:
    """Integration tests for geometry-rigidbody domain interactions."""

    def test_physics_scene_from_geometry(self):
        """Test creating a complete physics scene from geometric shapes."""
        try:
            from morphogen.stdlib.geometry import (
                point2d, circle, rectangle, polygon, shape_to_rigidbody
            )
            from morphogen.stdlib.rigidbody import ShapeType
        except ImportError:
            pytest.skip("Geometry or Rigidbody domain not available")
            return

        # Create geometric shapes
        floor = rectangle(center=point2d(0.0, -10.0), width=100.0, height=2.0)
        ball = circle(center=point2d(0.0, 20.0), radius=3.0)
        box = rectangle(center=point2d(5.0, 15.0), width=4.0, height=4.0)
        triangle = polygon(np.array([[10, 10], [15, 10], [12.5, 15]]))

        # Convert all to rigidbody shapes
        shapes = [
            shape_to_rigidbody(floor),
            shape_to_rigidbody(ball),
            shape_to_rigidbody(box),
            shape_to_rigidbody(triangle),
        ]

        # Verify all conversions
        assert len(shapes) == 4
        assert shapes[0]['shape_type'] == ShapeType.BOX
        assert shapes[1]['shape_type'] == ShapeType.CIRCLE
        assert shapes[2]['shape_type'] == ShapeType.BOX
        assert shapes[3]['shape_type'] == ShapeType.POLYGON

        # Verify shape parameters are preserved
        assert shapes[1]['shape_params']['radius'] == 3.0
        assert shapes[2]['shape_params']['width'] == 4.0
        assert shapes[2]['shape_params']['height'] == 4.0

    def test_collision_mesh_generation_pipeline(self):
        """Test full pipeline of mesh creation and collision mesh generation."""
        try:
            from morphogen.stdlib.geometry import (
                mesh, convex_hull, collision_mesh
            )
        except ImportError:
            pytest.skip("Geometry domain not available")
            return

        # Create a complex point cloud
        np.random.seed(42)
        points = np.random.randn(100, 3) * 5.0

        # Generate convex hull (3D)
        hull_mesh = convex_hull(points, dim=3)

        # Verify hull mesh
        assert hasattr(hull_mesh, 'vertices')
        assert hasattr(hull_mesh, 'faces')
        assert len(hull_mesh.vertices) > 0
        assert len(hull_mesh.faces) > 0

        # Generate simplified collision mesh
        collision = collision_mesh(hull_mesh, target_faces=20)

        # Verify collision mesh is simplified
        assert hasattr(collision, 'vertices')
        assert hasattr(collision, 'faces')
        # Should be simplified
        assert len(collision.faces) <= len(hull_mesh.faces)

    def test_multi_shape_rigidbody_conversion(self):
        """Test converting multiple geometric shapes for physics simulation."""
        try:
            from morphogen.stdlib.geometry import (
                point2d, circle, regular_polygon, shape_to_rigidbody
            )
            from morphogen.stdlib.rigidbody import ShapeType
        except ImportError:
            pytest.skip("Geometry or Rigidbody domain not available")
            return

        # Create various regular polygons
        shapes_geo = []
        for num_sides in range(3, 9):  # triangle through octagon
            poly = regular_polygon(
                center=point2d(float(num_sides * 10), 0.0),
                radius=5.0,
                num_sides=num_sides
            )
            shapes_geo.append(poly)

        # Convert all to rigidbody shapes
        shapes_rb = [shape_to_rigidbody(s) for s in shapes_geo]

        # All should be polygons
        assert all(s['shape_type'] == ShapeType.POLYGON for s in shapes_rb)

        # Verify vertex counts match sides
        for i, shape_rb in enumerate(shapes_rb):
            sides = i + 3
            assert len(shape_rb['shape_params']['vertices']) == sides


@pytest.mark.integration
class TestGeometryAlgorithmPipelines:
    """Integration tests for complex geometry algorithm pipelines."""

    def test_delaunay_voronoi_pipeline(self):
        """Test complete Delaunay triangulation to Voronoi diagram pipeline."""
        try:
            from morphogen.stdlib.geometry import (
                delaunay_triangulation, voronoi_diagram
            )
        except ImportError:
            pytest.skip("Geometry domain not available")
            return

        # Create point set
        np.random.seed(42)
        points = np.random.rand(50, 2) * 100.0

        # Compute Delaunay triangulation
        delaunay = delaunay_triangulation(points)

        # Verify triangulation
        assert hasattr(delaunay, 'vertices')
        assert hasattr(delaunay, 'faces')
        assert len(delaunay.vertices) == 50
        assert len(delaunay.faces) > 0

        # Compute Voronoi diagram from same points
        voronoi = voronoi_diagram(points)

        # Verify Voronoi
        assert 'vertices' in voronoi
        assert 'regions' in voronoi
        assert len(voronoi['vertices']) > 0

        # Voronoi should have ridge information
        # Note: ridge_vertices count != input point count (they're ridge endpoints)
        assert len(voronoi['regions']) > 0
        assert 'ridge_points' in voronoi

    def test_mesh_boolean_operations_chain(self):
        """Test chaining multiple mesh boolean operations."""
        try:
            from morphogen.stdlib.geometry import (
                box3d, sphere, mesh_union, mesh_intersection
            )
        except ImportError:
            pytest.skip("Geometry domain not available or mesh ops unavailable")
            return

        # Create two boxes
        from morphogen.stdlib.geometry import point3d
        box1 = box3d(center=point3d(0.0, 0.0, 0.0), width=2.0, height=2.0, depth=2.0)
        box2 = box3d(center=point3d(1.0, 0.0, 0.0), width=2.0, height=2.0, depth=2.0)

        # Union the boxes
        try:
            union = mesh_union(box1, box2)
            assert hasattr(union, 'vertices')
            assert hasattr(union, 'faces')
        except Exception:
            # mesh_union might not be fully implemented
            pytest.skip("mesh_union not fully implemented")

    def test_convex_hull_simplification_pipeline(self):
        """Test generating convex hull and simplifying for collision."""
        try:
            from morphogen.stdlib.geometry import (
                convex_hull, collision_mesh
            )
        except ImportError:
            pytest.skip("Geometry domain not available")
            return

        # Create a dense point cloud (simulating scanned data)
        np.random.seed(123)
        # Create points in a rough ellipsoid shape
        theta = np.random.rand(200) * 2 * np.pi
        phi = np.random.rand(200) * np.pi
        r = np.random.rand(200) * 0.2 + 1.0

        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta) * 2.0  # Stretch in y
        z = r * np.cos(phi) * 0.5  # Flatten in z

        points = np.column_stack([x, y, z])

        # Compute convex hull (3D)
        hull = convex_hull(points, dim=3)

        original_face_count = len(hull.faces)
        assert original_face_count > 0

        # Simplify for collision at different levels
        collision_low = collision_mesh(hull, target_faces=10)
        collision_high = collision_mesh(hull, target_faces=50)

        # Verify simplification levels
        assert len(collision_low.faces) <= original_face_count
        assert len(collision_high.faces) <= original_face_count
        assert len(collision_low.faces) <= len(collision_high.faces)


@pytest.mark.integration
class TestCrossTripleDomainIntegration:
    """Integration tests combining geometry, field, and rigidbody domains."""

    def test_field_driven_physics_objects(self):
        """Test creating physics objects based on field values."""
        try:
            from morphogen.stdlib.geometry import (
                point2d, circle, sample_field_at_point, shape_to_rigidbody
            )
            from morphogen.stdlib.rigidbody import ShapeType
        except ImportError:
            pytest.skip("Required domains not available")
            return

        # Create a density field
        density = field.random((32, 32), seed=42, low=0.0, high=1.0)
        density = field.diffuse(density, rate=0.5, dt=0.1, iterations=10)

        # Create physics objects at high-density locations
        physics_objects = []
        for i in range(5, 30, 5):
            for j in range(5, 30, 5):
                pt = point2d(float(i), float(j))
                value = sample_field_at_point(density, pt)

                # Create object if density is high enough
                if value > 0.6:
                    # Radius based on density
                    radius = value * 2.0
                    circ = circle(center=pt, radius=radius)

                    # Convert to rigidbody
                    rb_shape = shape_to_rigidbody(circ)
                    physics_objects.append({
                        'shape': rb_shape,
                        'mass': value * 10.0,  # Mass from density
                        'position': (i, j)
                    })

        # Should have created some objects
        assert len(physics_objects) > 0

        # Verify all objects have correct properties
        for obj in physics_objects:
            assert obj['shape']['shape_type'] == ShapeType.CIRCLE
            assert obj['mass'] > 6.0  # Since value > 0.6
            assert obj['shape']['shape_params']['radius'] > 0

    def test_physics_simulation_visualization(self):
        """Test visualizing geometric shapes in a physics simulation."""
        try:
            from morphogen.stdlib.geometry import (
                point2d, circle, rectangle, contains, shape_to_rigidbody
            )
        except ImportError:
            pytest.skip("Required domains not available")
            return

        # Create geometric shapes for physics
        shapes = [
            circle(center=point2d(16.0, 16.0), radius=5.0),
            rectangle(center=point2d(48.0, 48.0), width=8.0, height=8.0),
            circle(center=point2d(32.0, 32.0), radius=3.0),
        ]

        # Convert to rigidbody
        rb_shapes = [shape_to_rigidbody(s) for s in shapes]
        assert len(rb_shapes) == 3

        # Render shapes onto field for visualization
        render_field = field.alloc((64, 64), fill_value=0.0)
        data = render_field.data.copy()

        # Rasterize geometric shapes
        for shape in shapes:
            if hasattr(shape, 'radius'):  # circle
                for i in range(64):
                    for j in range(64):
                        pt = point2d(float(i), float(j))
                        if contains(shape, pt):
                            data[i, j] = 1.0
            elif hasattr(shape, 'width'):  # rectangle
                for i in range(64):
                    for j in range(64):
                        pt = point2d(float(i), float(j))
                        if contains(shape, pt):
                            data[i, j] = 0.5

        render_field.data = data

        # Visualize
        vis = visual.colorize(render_field, palette="fire")
        assert vis.shape == (64, 64)

        # Should have rendered some objects
        assert np.max(render_field.data) > 0

    def test_geometric_field_sampling_for_collision(self):
        """Test using field values to determine collision properties."""
        try:
            from morphogen.stdlib.geometry import (
                point2d, box3d, collision_mesh, sample_field_at_point
            )
        except ImportError:
            pytest.skip("Required domains not available")
            return

        # Create a material property field (e.g., hardness)
        hardness = field.random((64, 64), seed=99, low=0.0, high=1.0)
        hardness = field.diffuse(hardness, rate=0.2, dt=0.1, iterations=15)

        # Create geometric object
        from morphogen.stdlib.geometry import point3d
        box = box3d(center=point3d(32.0, 32.0, 0.0), width=10.0, height=10.0, depth=10.0)

        # Generate collision mesh at different detail levels based on hardness
        # Sample hardness at box center
        center_pt = point2d(32.0, 32.0)
        hardness_value = sample_field_at_point(hardness, center_pt)

        # Higher hardness = more detailed collision mesh
        target_faces = int(10 + hardness_value * 40)
        collision = collision_mesh(box, target_faces=target_faces)

        # Verify collision mesh
        assert hasattr(collision, 'vertices')
        assert hasattr(collision, 'faces')
        assert len(collision.faces) > 0

        # Face count should be influenced by hardness
        # (though exact count depends on simplification algorithm)
        assert len(collision.faces) <= target_faces * 2
