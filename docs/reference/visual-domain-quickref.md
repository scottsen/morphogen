# Visual Domain Quick Reference

Quick reference for Morphogen's Visual/Scene domain operators and types.

See [visual-scene-domain.md](./visual-scene-domain.md) for comprehensive architecture documentation.

---

## Scene Management

```julia
# Create and configure scene
scene = scene.create(name, objects=[])
scene.add(object)
scene.remove(object)
scene.set_background(color | gradient | field)
scene.set_camera(position, target, fov)
scene.render(t)
```

---

## Geometry Objects

### 2D Shapes
```julia
curve = geo.curve(f(t), t_range)
dot = geo.dot(position, radius)
arrow = geo.arrow(start, end)
line = geo.line(start, end)
circle = geo.circle(center, radius)
rectangle = geo.rectangle(corner, width, height)
```

### 3D Objects
```julia
curve3d = geo.curve3d(f(t), t_range)
surface = geo.parametric_surface(f(u,v), u_range, v_range)
mesh = geo.mesh(vertices, faces)
pointcloud = geo.pointcloud(points)
```

### Mathematical Objects
```julia
axes = geo.axes(x_range, y_range, [z_range])
grid = geo.grid(spacing, bounds)
plane = geo.number_plane(x_range, y_range)
vfield = geo.vector_field(F(x,y), bounds)
```

### Text & Equations
```julia
text = text.create("Hello", position)
equation = equation.create("E = mc²", position)
label = label.create(object, "name", offset)
```

---

## Object Properties

```julia
# Transforms
obj.set_position(x, y, [z])
obj.set_rotation(angle | quaternion)
obj.set_scale(factor | [x, y, z])

# Appearance
obj.set_color(color)
obj.set_opacity(alpha)
obj.set_stroke(width, color)
obj.set_fill(color)

# Hierarchy
obj.add_child(child)
obj.remove_child(child)
obj.parent = parent_obj
```

---

## Basic Animations

```julia
# Transform animations
anim.move(obj, to=pos, duration, easing="ease_in_out")
anim.rotate(obj, angle, duration, easing)
anim.scale(obj, factor, duration, easing)

# Appearance animations
anim.fade_in(obj, duration)
anim.fade_out(obj, duration)
anim.highlight(obj, color, duration)
anim.color_shift(obj, to=color, duration)

# Transform animations
anim.morph(obj1, to=obj2, duration)
anim.grow(obj, duration)          # Scale 0→1
anim.shrink(obj, duration)        # Scale 1→0
```

---

## Math-Specific Animations

```julia
# Equation animations
anim.write_equation(eq, duration)
anim.transform_equation(eq1, eq2, duration)
anim.show_substitution(eq, var, value)

# Curve animations
anim.draw_curve(curve, duration)
anim.trace_curve(curve, duration)

# Field animations
anim.show_field(field, duration)
anim.evolve_field(field, duration)
```

---

## Timeline Composition

```julia
# Sequential animations
timeline.sequence([anim1, anim2, anim3])

# Parallel animations
timeline.parallel([anim1, anim2, anim3])

# Staggered animations
timeline.stagger([anims], delay=0.1)

# Loops
timeline.loop(anim, count | Inf)

# Wait
timeline.wait(duration)
```

---

## Camera Controls

```julia
# Camera movement
camera.pan(to=point, duration, easing)
camera.zoom(factor, duration, easing)
camera.orbit(angle, duration, easing)

# Camera tracking
camera.follow(object, [offset])
camera.look_at(target)

# Camera properties
camera.set_fov(angle)
camera.set_position(x, y, z)
camera.set_target(x, y, z)
```

---

## Easing Functions

```julia
"linear"           # Constant speed
"ease_in"          # Slow start
"ease_out"         # Slow end
"ease_in_out"      # Slow start and end
"ease_in_cubic"    # Strong slow start
"ease_out_cubic"   # Strong slow end
"bounce"           # Bouncing effect
"elastic"          # Elastic effect
"back"             # Slight overshoot
```

Custom easing:
```julia
custom_ease = t -> t^2  # Quadratic
anim.move(obj, to=pos, duration, easing=custom_ease)
```

---

## Palette Integration

```julia
# Color from palette
obj.set_color(palette.get("viridis", 0.5))

# Animate color mapping
anim.color_map(surface, field=scalar_field, palette="inferno", duration)

# Cycle palette
anim.cycle_palette(obj, palette="rainbow", speed=1.0)
```

---

## Noise Integration

```julia
# Noise-based background
scene.background(noise.fbm2d(x, y, t, octaves=4))

# Noise-driven motion
anim.perturb(obj, noise=noise.simplex3d, amplitude=0.1, frequency=2.0)

# Procedural texture
texture = noise.cellular2d(x, y)
surface.set_texture(texture)
```

---

## Field Integration

```julia
# Vector field visualization
F(x,y) = [-y, x]  # Rotation field
vfield = geo.vector_field(F, bounds=(-5,5,-5,5))
scene.add(vfield)

# Integral curves
curve = field.integral_curve(F, start_point)
anim.draw_curve(curve, duration)

# Field-based coloring
surface.color_by_field(field=scalar_field, palette="coolwarm")
```

---

## Video Export

```julia
# Render scene to video
video = video.from_scene(
    scene,
    fps=60,
    duration=10.0,
    resolution=(1920, 1080)
)

# Add audio
video.add_audio(audio_track, sync=true)

# Encode
video.encode(
    codec="h264",
    crf=18,          # Quality (lower = better, 18-28 typical)
    preset="slow"    # Speed vs compression
)
```

---

## Common Patterns

### Vector Field with Integral Curves
```julia
# Define field
F(x,y) = [-y, x]

# Create scene
scene = scene.create("vector_field")
vfield = geo.vector_field(F, bounds=(-5,5,-5,5))
scene.add(vfield)

# Add integral curves
for start in [(1,0), (2,0), (3,0)]:
    curve = field.integral_curve(F, start)
    scene.add(curve)
    anim.draw_curve(curve, duration=2.0)
```

### Equation Transformation
```julia
# Start equation
eq1 = equation.create("∫₀¹ x² dx")
scene.add(eq1)
anim.write_equation(eq1, duration=1.0)

# Transform to result
eq2 = equation.create("= ⅓")
anim.transform_equation(eq1, eq2, duration=1.0)
```

### Parametric Surface with Color
```julia
# Create surface
surface = geo.parametric_surface(
    (u,v) -> [u*cos(v), u*sin(v), u],
    u=(0,2), v=(0,2π)
)

# Color by height
surface.color_by_field(
    field=(x,y,z) -> z,
    palette="viridis"
)

# Animate
scene.add(surface)
anim.grow(surface, duration=2.0)
camera.orbit(2π, duration=10.0)
```

### Procedural Background
```julia
# Animated noise background
bg_field = noise.fbm2d(x, y, t, octaves=6)
scene.background(
    palette.map(bg_field, "plasma")
)
```

---

## Type Quick Reference

### Core Types
- `Scene` - Container for objects and animations
- `MObject` - Base visual object
- `Animation` - Base animation type
- `Timeline` - Animation sequence container
- `Camera` - Viewpoint and projection

### Geometry Types
- `Curve2D`, `Curve3D` - Parametric curves
- `Surface`, `Mesh` - 3D surfaces
- `PointCloud` - Point-based data
- `Arrow`, `Line`, `Ray` - Linear objects
- `Dot`, `Circle`, `Rectangle` - Basic shapes
- `Axes`, `Grid`, `NumberPlane` - Coordinate systems
- `VectorField` - Vector field visualization
- `Text`, `Equation`, `Label` - Text objects

### Animation Types
- `MoveAnimation` - Position change
- `RotateAnimation` - Rotation change
- `ScaleAnimation` - Scale change
- `FadeAnimation` - Opacity change
- `MorphAnimation` - Object transformation
- `ColorAnimation` - Color change
- `SequenceAnimation` - Sequential composition
- `ParallelAnimation` - Parallel composition

---

## Best Practices

1. **Use composition** - Build complex scenes from simple objects
2. **Animate one property at a time** - Parallel animations for multiple properties
3. **Use easing** - Makes animations feel more natural
4. **Group related objects** - Use parent-child relationships
5. **Cache expensive computations** - Especially for parametric surfaces
6. **Use GPU acceleration** - For large point clouds and fields
7. **Preview at low resolution** - Speed up iteration
8. **Render at high resolution** - Final export at 4K

---

## See Also

- [Visual/Scene Domain Architecture](./visual-scene-domain.md) - Comprehensive documentation
- [Emergence Domain](./emergence-domain.md) - For simulation integration
- [Palette Domain](../../architecture/domain-architecture.md) - Color management
- [Noise Domain](../../architecture/domain-architecture.md) - Procedural content
- [Field Domain](../../architecture/domain-architecture.md) - Field operations
