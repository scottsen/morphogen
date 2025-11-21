"""Interactive Physics Sandbox - Cross-Domain Showcase

This example demonstrates a 2D physics sandbox with rigid body simulation,
collision detection, and real-time visualization.

Domains Integrated:
- RigidBody: 2D physics simulation with collisions
- Field: Velocity field visualization
- Palette: Color mapping for physics quantities
- Visual: Rendering and composition
- Image: Output generation

The sandbox shows various physics scenarios:
1. Falling objects (gravity)
2. Bouncing balls (elastic collisions)
3. Box stacking (friction and contact)
4. Domino chains (impulse propagation)
5. Particle rain (many-body dynamics)

Cross-Domain Integration:
- RigidBody → Visual: Physics state drives rendering
- RigidBody → Field: Velocity vectors create field visualization
- Field → Palette: Velocity magnitude mapped to color
- All → Visual: Composite visualization

Run: python examples/interactive_physics_sandbox/demo.py
"""

import numpy as np
from pathlib import Path
from morphogen.stdlib import rigidbody, field, palette, visual, image, color
from morphogen.stdlib.rigidbody import (
    PhysicsWorld2D, RigidBody2D,
    create_circle_body, create_box_body,
    step_world, get_body_vertices
)
from morphogen.stdlib.field import Field2D


# ============================================================================
# RENDERING UTILITIES
# ============================================================================

def world_to_pixel(world_pos: np.ndarray,
                   world_bounds: tuple,
                   image_size: tuple) -> np.ndarray:
    """Convert world coordinates to pixel coordinates."""
    x_min, x_max, y_min, y_max = world_bounds
    width, height = image_size

    # Normalize to 0-1
    x_norm = (world_pos[0] - x_min) / (x_max - x_min)
    y_norm = (world_pos[1] - y_min) / (y_max - y_min)

    # Convert to pixel coordinates (flip Y for screen space)
    px = int(x_norm * width)
    py = int((1.0 - y_norm) * height)

    return np.array([px, py])


def render_world(world: PhysicsWorld2D,
                world_bounds: tuple = (-10, 10, -10, 10),
                image_size: tuple = (800, 800),
                show_velocity: bool = False) -> np.ndarray:
    """Render physics world to RGB image.

    Args:
        world: Physics world to render
        world_bounds: (x_min, x_max, y_min, y_max) in world space
        image_size: (width, height) in pixels
        show_velocity: Overlay velocity vectors

    Returns:
        RGB image array
    """
    width, height = image_size

    # Create blank canvas
    img = np.ones((height, width, 3), dtype=np.uint8) * 240  # Light background

    # Draw each body
    for body in world.bodies:
        if body.shape_type == rigidbody.ShapeType.CIRCLE:
            # Draw circle
            radius = body.shape_params["radius"]
            center_px = world_to_pixel(body.position, world_bounds, image_size)

            # Calculate radius in pixels
            x_scale = width / (world_bounds[1] - world_bounds[0])
            y_scale = height / (world_bounds[3] - world_bounds[2])
            radius_px = int(radius * (x_scale + y_scale) / 2)

            # Color based on velocity magnitude
            speed = np.linalg.norm(body.velocity)
            color_val = min(1.0, speed / 10.0)

            # Color gradient: blue (slow) -> red (fast)
            r = int(color_val * 200 + 55)
            g = int((1.0 - color_val) * 100 + 55)
            b = int((1.0 - color_val) * 200 + 55)

            # Draw filled circle
            y_grid, x_grid = np.ogrid[:height, :width]
            circle_mask = (x_grid - center_px[0])**2 + (y_grid - center_px[1])**2 <= radius_px**2

            if np.any(circle_mask):
                img[circle_mask] = [r, g, b]

            # Draw outline
            outline_mask = ((x_grid - center_px[0])**2 + (y_grid - center_px[1])**2 <= radius_px**2) & \
                          ((x_grid - center_px[0])**2 + (y_grid - center_px[1])**2 > (radius_px-2)**2)
            if np.any(outline_mask):
                img[outline_mask] = [30, 30, 30]

        elif body.shape_type == rigidbody.ShapeType.BOX:
            # Draw box
            vertices = get_body_vertices(body)

            if vertices is not None:
                # Convert vertices to pixel coords
                verts_px = np.array([world_to_pixel(v, world_bounds, image_size) for v in vertices])

                # Color based on velocity
                speed = np.linalg.norm(body.velocity)
                color_val = min(1.0, speed / 10.0)

                r = int(color_val * 200 + 55)
                g = int((1.0 - color_val) * 100 + 55)
                b = int((1.0 - color_val) * 200 + 55)

                # Fill polygon (simple scanline fill)
                from scipy.ndimage import binary_fill_holes

                # Create mask
                mask = np.zeros((height, width), dtype=bool)

                # Draw edges
                for i in range(len(verts_px)):
                    v1 = verts_px[i]
                    v2 = verts_px[(i + 1) % len(verts_px)]

                    # Bresenham's line algorithm (simplified)
                    n_points = int(np.linalg.norm(v2 - v1)) + 1
                    x_vals = np.linspace(v1[0], v2[0], n_points).astype(int)
                    y_vals = np.linspace(v1[1], v2[1], n_points).astype(int)

                    # Clip to image bounds
                    valid = (x_vals >= 0) & (x_vals < width) & (y_vals >= 0) & (y_vals < height)
                    x_vals = x_vals[valid]
                    y_vals = y_vals[valid]

                    if len(x_vals) > 0:
                        mask[y_vals, x_vals] = True

                # Fill interior
                mask = binary_fill_holes(mask)
                img[mask] = [r, g, b]

                # Draw outline
                for i in range(len(verts_px)):
                    v1 = verts_px[i]
                    v2 = verts_px[(i + 1) % len(verts_px)]

                    n_points = int(np.linalg.norm(v2 - v1)) + 1
                    x_vals = np.linspace(v1[0], v2[0], n_points).astype(int)
                    y_vals = np.linspace(v1[1], v2[1], n_points).astype(int)

                    valid = (x_vals >= 0) & (x_vals < width) & (y_vals >= 0) & (y_vals < height)
                    x_vals = x_vals[valid]
                    y_vals = y_vals[valid]

                    if len(x_vals) > 0:
                        img[y_vals, x_vals] = [30, 30, 30]

        # Draw velocity vector if requested
        if show_velocity and not body.is_static:
            center_px = world_to_pixel(body.position, world_bounds, image_size)
            end_pos = body.position + body.velocity * 0.5  # Scale for visibility
            end_px = world_to_pixel(end_pos, world_bounds, image_size)

            # Draw arrow
            n_points = int(np.linalg.norm(end_px - center_px)) + 1
            if n_points > 1:
                x_vals = np.linspace(center_px[0], end_px[0], n_points).astype(int)
                y_vals = np.linspace(center_px[1], end_px[1], n_points).astype(int)

                valid = (x_vals >= 0) & (x_vals < width) & (y_vals >= 0) & (y_vals < height)
                x_vals = x_vals[valid]
                y_vals = y_vals[valid]

                if len(x_vals) > 0:
                    img[y_vals, x_vals] = [0, 200, 0]

    return img


def create_velocity_field_visualization(world: PhysicsWorld2D,
                                        world_bounds: tuple = (-10, 10, -10, 10),
                                        grid_size: tuple = (100, 100)) -> np.ndarray:
    """Create a field visualization of body velocities.

    Args:
        world: Physics world
        world_bounds: World coordinate bounds
        grid_size: Field grid resolution

    Returns:
        RGB image of velocity field
    """
    x_min, x_max, y_min, y_max = world_bounds
    grid_h, grid_w = grid_size

    # Create velocity magnitude field
    vel_field = np.zeros((grid_h, grid_w), dtype=np.float32)

    # Sample velocity at grid points (influenced by nearby bodies)
    for i in range(grid_h):
        for j in range(grid_w):
            # Grid point in world space
            x = x_min + (j / grid_w) * (x_max - x_min)
            y = y_min + ((grid_h - 1 - i) / grid_h) * (y_max - y_min)
            point = np.array([x, y])

            # Accumulate velocity influence from nearby bodies
            total_vel = 0.0
            for body in world.bodies:
                if not body.is_static:
                    dist = np.linalg.norm(body.position - point)
                    if dist < 5.0:  # Influence radius
                        # Inverse distance weighting
                        weight = 1.0 / (dist + 0.5)
                        speed = np.linalg.norm(body.velocity)
                        total_vel += speed * weight

            vel_field[i, j] = total_vel

    # Normalize
    if vel_field.max() > 0:
        vel_field = vel_field / vel_field.max()

    # Apply colormap
    pal = palette.plasma(256)
    img = palette.map(pal, vel_field)

    return img


# ============================================================================
# DEMO SCENARIOS
# ============================================================================

def scenario1_falling_objects(output_dir: Path):
    """Scenario 1: Falling objects with gravity."""
    print("  Scenario 1: Falling Objects")

    # Create world
    world = PhysicsWorld2D(gravity=np.array([0.0, -9.81]))

    # Add ground (static box)
    ground = create_box_body(
        position=np.array([0.0, -8.0]),
        width=18.0,
        height=1.0,
        mass=0.0,  # Static
        restitution=0.3
    )
    world.add_body(ground)

    # Add falling balls
    for i in range(5):
        ball = create_circle_body(
            position=np.array([-6.0 + i * 3.0, 8.0 - i * 0.5]),
            radius=0.5,
            mass=1.0,
            restitution=0.7  # Bouncy
        )
        world.add_body(ball)

    # Simulate and capture frames
    frames = []
    n_steps = 180
    capture_interval = 10

    for step in range(n_steps):
        if step % capture_interval == 0:
            img = render_world(world, world_bounds=(-10, 10, -10, 10))
            frames.append(img)

        step_world(world)

    # Save key frames
    for i, frame_idx in enumerate([0, len(frames)//3, 2*len(frames)//3, -1]):
        output_path = output_dir / f"scenario1_frame{i:02d}.png"
        image.save(frames[frame_idx], str(output_path))

    print(f"    ✓ Saved {len([0, len(frames)//3, 2*len(frames)//3, -1])} frames")


def scenario2_box_stack(output_dir: Path):
    """Scenario 2: Stacking boxes (friction and stability)."""
    print("  Scenario 2: Box Stack")

    world = PhysicsWorld2D(gravity=np.array([0.0, -9.81]))

    # Ground
    ground = create_box_body(
        position=np.array([0.0, -8.0]),
        width=18.0,
        height=1.0,
        mass=0.0,
        restitution=0.1,
        friction=0.5
    )
    world.add_body(ground)

    # Stack boxes
    n_boxes = 6
    box_height = 1.0

    for i in range(n_boxes):
        box = create_box_body(
            position=np.array([0.0, -7.0 + i * box_height + 0.1]),
            width=1.5,
            height=box_height,
            mass=1.0,
            restitution=0.1,
            friction=0.6
        )
        world.add_body(box)

    # Simulate
    frames = []
    n_steps = 200
    capture_interval = 10

    for step in range(n_steps):
        if step % capture_interval == 0:
            img = render_world(world)
            frames.append(img)

        step_world(world)

    # Save frames
    for i, frame_idx in enumerate([0, len(frames)//2, -1]):
        output_path = output_dir / f"scenario2_frame{i:02d}.png"
        image.save(frames[frame_idx], str(output_path))

    print(f"    ✓ Saved {len([0, len(frames)//2, -1])} frames")


def scenario3_particle_rain(output_dir: Path):
    """Scenario 3: Particle rain (many-body dynamics)."""
    print("  Scenario 3: Particle Rain")

    world = PhysicsWorld2D(gravity=np.array([0.0, -9.81]))

    # Ground
    ground = create_box_body(
        position=np.array([0.0, -8.0]),
        width=18.0,
        height=1.0,
        mass=0.0,
        restitution=0.4
    )
    world.add_body(ground)

    # Create particles in batches
    np.random.seed(42)
    n_particles = 30

    for i in range(n_particles):
        x = np.random.uniform(-8.0, 8.0)
        y = 9.0

        particle = create_circle_body(
            position=np.array([x, y]),
            radius=0.3,
            mass=0.5,
            restitution=0.6
        )
        world.add_body(particle)

    # Simulate
    frames = []
    n_steps = 300
    capture_interval = 15

    for step in range(n_steps):
        if step % capture_interval == 0:
            img = render_world(world)
            frames.append(img)

        step_world(world)

    # Save frames
    for i, frame_idx in enumerate([0, len(frames)//4, len(frames)//2, 3*len(frames)//4, -1]):
        output_path = output_dir / f"scenario3_frame{i:02d}.png"
        image.save(frames[frame_idx], str(output_path))

    print(f"    ✓ Saved {len([0, len(frames)//4, len(frames)//2, 3*len(frames)//4, -1])} frames")


def scenario4_velocity_field(output_dir: Path):
    """Scenario 4: Velocity field visualization."""
    print("  Scenario 4: Velocity Field Visualization")

    world = PhysicsWorld2D(gravity=np.array([0.0, 0.0]))  # No gravity

    # Create moving bodies
    for i in range(8):
        angle = i * np.pi / 4
        radius_orbit = 5.0

        x = radius_orbit * np.cos(angle)
        y = radius_orbit * np.sin(angle)

        vx = -3.0 * np.sin(angle)
        vy = 3.0 * np.cos(angle)

        body = create_circle_body(
            position=np.array([x, y]),
            radius=0.5,
            mass=1.0
        )
        body.velocity = np.array([vx, vy])

        world.add_body(body)

    # Simulate and create field visualizations
    n_steps = 120
    capture_interval = 30

    for step in range(n_steps):
        if step % capture_interval == 0:
            # Create velocity field visualization
            vel_field_img = create_velocity_field_visualization(world, grid_size=(200, 200))
            output_path = output_dir / f"scenario4_field_step{step:03d}.png"
            image.save(vel_field_img, str(output_path))

        step_world(world)

    print(f"    ✓ Saved velocity field frames")


# ============================================================================
# MAIN DEMO
# ============================================================================

def main():
    """Run physics sandbox demonstration."""
    print("=" * 70)
    print("INTERACTIVE PHYSICS SANDBOX - CROSS-DOMAIN SHOWCASE")
    print("=" * 70)
    print()
    print("Demonstrating cross-domain integration:")
    print("  • RigidBody: 2D physics simulation")
    print("  • Field: Velocity field visualization")
    print("  • Palette: Color mapping for physics quantities")
    print("  • Visual: Rendering and composition")
    print()

    # Create output directory
    output_dir = Path("examples/interactive_physics_sandbox/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run scenarios
    print("Running Physics Scenarios")
    print("-" * 70)

    scenario1_falling_objects(output_dir)
    print()

    scenario2_box_stack(output_dir)
    print()

    scenario3_particle_rain(output_dir)
    print()

    scenario4_velocity_field(output_dir)
    print()

    print("=" * 70)
    print("DEMO COMPLETE!")
    print("=" * 70)
    print()
    print("Cross-Domain Integration Demonstrated:")
    print("  ✓ RigidBody → Visual: Physics state drives rendering")
    print("  ✓ RigidBody → Field: Velocity creates field visualization")
    print("  ✓ Field → Palette: Velocity magnitude mapped to color")
    print("  ✓ All → Image: Composite visualization output")
    print()
    print(f"All outputs saved to: {output_dir}/")
    print()
    print("Scenarios:")
    print("  1. Falling Objects - Gravity and elastic collisions")
    print("  2. Box Stack - Friction and stability")
    print("  3. Particle Rain - Many-body dynamics")
    print("  4. Velocity Field - Field-based visualization")
    print()
    print("Key Insights:")
    print("  • Rigid body physics enables realistic object interactions")
    print("  • Velocity visualization reveals system dynamics")
    print("  • Color mapping communicates physical quantities")
    print("  • Cross-domain integration creates rich simulations")


if __name__ == "__main__":
    main()
