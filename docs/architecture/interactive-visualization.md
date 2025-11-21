# Interactive Visualization Guide

Creative Computation DSL now supports real-time interactive visualization! Watch your simulations come to life with smooth, controllable playback.

## Quick Start

### Python API

```python
from creative_computation.stdlib.field import field
from creative_computation.stdlib.visual import visual

def my_simulation():
    """Generator that yields frames."""
    temp = field.random((128, 128), seed=42)

    while True:
        temp = field.diffuse(temp, rate=0.2, dt=0.1)
        yield visual.colorize(temp, palette="fire")

# Run interactively
gen = my_simulation()
visual.display(lambda: next(gen), title="My Simulation")
```

### From DSL Files

```bash
# Run with interactive display (when available in DSL)
ccdsl run examples/heat_diffusion_animated.ccdsl --steps 100
```

## Keyboard Controls

### During Simulation

| Key | Action |
|-----|--------|
| **SPACE** | Pause/Resume simulation |
| **→** (Right Arrow) | Step forward one frame (when paused) |
| **↑** (Up Arrow) | Increase simulation speed (+5 FPS) |
| **↓** (Down Arrow) | Decrease simulation speed (-5 FPS) |
| **Q** or **ESC** | Quit simulation |

### On-Screen Display

The visualization window shows:
- **Current FPS** (actual / target)
- **Frame count** (when paused)
- **Status** (RUNNING or PAUSED)
- **Controls** reminder

## API Reference

### visual.display()

Display simulation in real-time interactive window.

```python
def display(
    frame_generator: Callable[[], Optional[Visual]],
    title: str = "Creative Computation DSL",
    target_fps: int = 30,
    scale: int = 2
) -> None
```

**Parameters:**

- `frame_generator`: Callable that returns Visual frames
  - Should return `Visual` object for each frame
  - Return `None` to end simulation
  - Can be a lambda wrapping a generator: `lambda: next(gen)`

- `title`: Window title (default: "Creative Computation DSL")

- `target_fps`: Target frame rate in frames per second (default: 30)
  - Can be adjusted during simulation with ↑↓ keys
  - Actual FPS may be lower if computation is expensive

- `scale`: Display scale factor (default: 2)
  - Multiplier for visual resolution
  - Use larger values for small grids (e.g., 128×128)
  - Use smaller values for large grids (e.g., 512×512)

**Returns:** None (blocks until window is closed)

**Raises:**
- `ImportError`: If pygame is not installed
- `TypeError`: If frame_generator doesn't return Visual

## Examples

### Example 1: Heat Diffusion

```python
from creative_computation.stdlib.field import field
from creative_computation.stdlib.visual import visual

def heat_diffusion():
    temp = field.random((128, 128), seed=42, low=0.0, high=1.0)

    while True:
        temp = field.diffuse(temp, rate=0.2, dt=0.1, iterations=20)
        temp = field.boundary(temp, spec="reflect")
        yield visual.colorize(temp, palette="fire")

gen = heat_diffusion()
visual.display(
    frame_generator=lambda: next(gen),
    title="Heat Diffusion",
    target_fps=30,
    scale=4
)
```

### Example 2: With Finite Steps

```python
def limited_simulation():
    temp = field.random((64, 64), seed=42)

    for i in range(100):  # Only 100 frames
        temp = field.diffuse(temp, rate=0.1, dt=0.1)
        yield visual.colorize(temp, palette="viridis")

    # Return None to signal end
    return None

gen = limited_simulation()
visual.display(lambda: next(gen, None))  # Use next() with default
```

### Example 3: Multiple Fields

```python
def multi_field_visualization():
    """Show different fields each second."""
    temps = [
        field.random((128, 128), seed=i, low=0.0, high=1.0)
        for i in range(5)
    ]

    frame = 0
    while True:
        # Switch field every 30 frames (1 second at 30 FPS)
        idx = (frame // 30) % len(temps)
        temp = temps[idx]

        temp = field.diffuse(temp, rate=0.1, dt=0.1)
        temps[idx] = temp  # Update in list

        yield visual.colorize(temp, palette="fire")
        frame += 1

gen = multi_field_visualization()
visual.display(lambda: next(gen))
```

## Best Practices

### Performance Tips

1. **Choose appropriate grid size**
   - 128×128: Fast, good for testing (~60 FPS)
   - 256×256: Good balance (~30 FPS)
   - 512×512: High quality (~10 FPS)

2. **Adjust scale for visibility**
   - Small grids (64-128): use `scale=4`
   - Medium grids (256): use `scale=2`
   - Large grids (512+): use `scale=1`

3. **Tune iteration counts**
   - Fewer iterations = faster but less accurate
   - Start with 10-20 iterations for diffusion
   - Use 20-40 for projection

### Visual Design

1. **Choose palettes wisely**
   - `fire`: Hot/cold phenomena (temperature, energy)
   - `viridis`: General purpose, colorblind-safe
   - `coolwarm`: Diverging data (positive/negative)
   - `grayscale`: Simple, high contrast

2. **Set appropriate value ranges**
   ```python
   vis = visual.colorize(field, palette="fire", vmin=0.0, vmax=1.0)
   ```
   - Fixed ranges maintain consistent colors
   - Auto ranges (default) adapt to data

3. **Add visual feedback**
   - Use title to describe simulation
   - Show parameter values in title
   ```python
   visual.display(
       lambda: next(gen),
       title=f"Diffusion (rate={rate}, dt={dt})"
   )
   ```

### Error Handling

```python
def safe_simulation():
    try:
        temp = field.random((128, 128), seed=42)

        while True:
            temp = field.diffuse(temp, rate=0.1, dt=0.1)
            yield visual.colorize(temp, palette="fire")

    except KeyboardInterrupt:
        print("Simulation interrupted by user")
        return None
    except Exception as e:
        print(f"Error in simulation: {e}")
        return None

gen = safe_simulation()
visual.display(lambda: next(gen, None))
```

## Troubleshooting

### Window doesn't open

**Problem:** `ImportError: pygame is required`

**Solution:**
```bash
pip install pygame
```

### Slow performance

**Problem:** FPS much lower than target

**Solutions:**
1. Reduce grid size
2. Decrease iteration counts
3. Reduce scale factor
4. Profile your code to find bottlenecks

### Display looks pixelated

**Problem:** Blocky visualization

**Solution:** Increase scale factor or grid resolution
```python
# Use larger scale for small grids
visual.display(..., scale=4)

# Or use larger grid
field.random((256, 256), ...)
```

### Colors look wrong

**Problem:** Washed out or oversaturated

**Solution:** Set explicit value ranges
```python
# Clamp to known physical range
vis = visual.colorize(temp, palette="fire", vmin=0.0, vmax=100.0)
```

## Advanced Usage

### Custom Frame Generator

```python
class SimulationController:
    def __init__(self):
        self.temp = field.random((128, 128), seed=42)
        self.running = True

    def get_frame(self):
        if not self.running:
            return None

        self.temp = field.diffuse(self.temp, rate=0.1, dt=0.1)
        return visual.colorize(self.temp, palette="fire")

controller = SimulationController()
visual.display(controller.get_frame)
```

### Saving Frames While Displaying

```python
def simulation_with_save():
    temp = field.random((128, 128), seed=42)
    frame_count = 0

    while frame_count < 1000:
        temp = field.diffuse(temp, rate=0.1, dt=0.1)
        vis = visual.colorize(temp, palette="fire")

        # Save every 10th frame
        if frame_count % 10 == 0:
            visual.output(vis, path=f"frames/frame_{frame_count:04d}.png")

        frame_count += 1
        yield vis

gen = simulation_with_save()
visual.display(lambda: next(gen, None))
```

## See Also

- [Getting Started Guide](GETTING_STARTED.md)
- [Field Operations Reference](../LANGUAGE_REFERENCE.md#field-operations)
- [Example Programs](../examples/README.md)
