# I/O & Storage Domain Examples

This directory contains examples demonstrating the I/O & Storage domain operations in Kairo.

## Examples Overview

### 1. Image I/O (`01_image_io.py`)

Demonstrates image loading, saving, and manipulation:

- **Example 1**: Generate and save gradient images (PNG, JPEG)
- **Example 2**: Load images and convert to grayscale
- **Example 3**: Procedural texture generation (checkerboard, circular patterns)
- **Example 4**: Data analysis workflow (heatmap visualization)

**Run:**
```bash
python 01_image_io.py
```

**Outputs:**
- `gradient.png` - RGB gradient image
- `gradient.jpg` - JPEG version
- `gradient_gray.png` - Grayscale conversion
- `checkerboard.png` - 8x8 checkerboard pattern
- `circular_pattern.png` - Rainbow circular waves
- `heatmap.png` - Heatmap visualization

---

### 2. Audio I/O (`02_audio_io.py`)

Demonstrates audio generation, saving, and loading:

- **Example 1**: Basic tone generation (440 Hz, 880 Hz)
- **Example 2**: Stereo audio (different frequencies in L/R channels)
- **Example 3**: Musical chord synthesis (A major chord with ADSR envelope)
- **Example 4**: Simple audio effects (tremolo, vibrato, echo)

**Run:**
```bash
python 02_audio_io.py
```

**Outputs:**
- `tone_a4.wav` - 440 Hz sine wave
- `tone_a5.wav` - 880 Hz sine wave
- `stereo_tone.wav` - Binaural tones (L=440 Hz, R=554 Hz)
- `chord_a_major.wav` - A major chord (WAV)
- `chord_a_major.flac` - A major chord (FLAC lossless)
- `effect_tremolo.wav` - Amplitude modulation effect
- `effect_vibrato.wav` - Frequency modulation effect
- `effect_echo.wav` - Delay effect (300ms)

---

### 3. Simulation Checkpointing (`03_simulation_checkpointing.py`)

Demonstrates checkpoint/resume workflow for simulations:

- **Example 1**: Basic checkpoint save/load
- **Example 2**: Resume simulation from checkpoint
- **Example 3**: Periodic checkpointing during long simulations
- **Example 4**: Complex multi-field checkpoints with visualization

**Run:**
```bash
python 03_simulation_checkpointing.py
```

**Outputs:**
- `checkpoint_*.h5` - Basic checkpoints (iterations 0-10)
- `periodic_checkpoint_*.h5` - Periodic checkpoints (every 20 steps)
- `complex_checkpoint.h5` - Multi-field checkpoint (velocity, pressure, temperature)
- `checkpoint_temperature_vis.png` - Visualization of temperature field

---

## API Overview

### Image I/O

```python
from morphogen.stdlib import io_storage as io

# Load image
img = io.load_image("texture.png")  # Returns (H, W, 3) array in [0, 1]
img_gray = io.load_image("photo.jpg", grayscale=True)  # Returns (H, W)

# Save image
io.save_image("output.png", img_array)  # Automatically denormalizes from [0, 1]
io.save_image("output.jpg", img_array, quality=95)  # JPEG with quality control
```

### Audio I/O

```python
from morphogen.stdlib import io_storage as io

# Load audio
audio, sample_rate = io.load_audio("music.wav")
audio_mono, sr = io.load_audio("stereo.flac", mono=True)  # Downmix to mono

# Save audio
io.save_audio("tone.wav", audio_data, sample_rate)
io.save_audio("music.flac", audio_data, sample_rate)  # Lossless compression
```

### JSON I/O

```python
from morphogen.stdlib import io_storage as io

# Load JSON
config = io.load_json("config.json")

# Save JSON (handles NumPy types automatically)
params = {"learning_rate": 0.01, "epochs": 100}
io.save_json("params.json", params, indent=2)
```

### HDF5 I/O

```python
from morphogen.stdlib import io_storage as io

# Save single array
io.save_hdf5("field.h5", velocity_field)

# Save multiple arrays
io.save_hdf5("results.h5", {
    "velocity": vel_field,
    "pressure": press_field,
    "temperature": temp_field
})

# Load all datasets
data = io.load_hdf5("results.h5")
velocity = data["velocity"]

# Load specific dataset
pressure = io.load_hdf5("results.h5", dataset="pressure")
```

### Checkpointing

```python
from morphogen.stdlib import io_storage as io

# Save checkpoint
state = {
    "velocity_field": vel,
    "pressure_field": press,
    "parameters": {"dt": 0.01, "viscosity": 0.1}
}
metadata = {"iteration": 1000, "time": 10.0}
io.save_checkpoint("sim_checkpoint.h5", state, metadata)

# Load checkpoint
loaded_state, loaded_metadata = io.load_checkpoint("sim_checkpoint.h5")
vel = loaded_state["velocity_field"]
iteration = loaded_metadata["iteration"]
```

---

## Supported Formats

### Images
- **PNG** - Lossless compression, supports RGB/RGBA/grayscale
- **JPEG** - Lossy compression with quality control (1-100)
- **BMP** - Uncompressed bitmap format

### Audio
- **WAV** - Uncompressed or lossless (PCM, FLOAT subtypes)
- **FLAC** - Lossless compression (~50% size reduction)
- **OGG** - Lossy compression (via soundfile)

### Data
- **JSON** - Human-readable text format, automatic NumPy type conversion
- **HDF5** - Binary format with compression, efficient for large arrays

---

## Dependencies

The I/O domain requires the following packages:

```bash
pip install numpy pillow soundfile h5py scipy
```

- **Pillow** - Image I/O (PNG, JPEG, BMP, etc.)
- **soundfile** - Audio I/O (WAV, FLAC, OGG)
- **h5py** - HDF5 I/O and checkpointing
- **scipy** - Audio resampling (optional)

---

## Use Cases

### Scientific Computing
- Save simulation results as HDF5 for analysis
- Export field visualizations as PNG/JPEG
- Checkpoint long-running simulations

### Audio Production
- Generate procedural audio and export as WAV/FLAC
- Load audio samples for processing
- Save processed audio with lossless quality

### Machine Learning
- Save training checkpoints with model weights and metadata
- Load datasets from HDF5 files
- Export visualizations of training progress

### Data Analysis
- Convert numerical data to heatmap visualizations
- Save analysis results as JSON for sharing
- Create reproducible data pipelines

---

## Tips

1. **Image formats**: Use PNG for lossless quality, JPEG for smaller file sizes
2. **Audio formats**: Use FLAC for archival, WAV for compatibility
3. **Checkpointing**: Include metadata (iteration, time, version) for reproducibility
4. **HDF5 compression**: Enable compression for large arrays to save disk space
5. **File paths**: All functions accept both strings and `pathlib.Path` objects

---

## Next Steps

After running these examples, explore:

- **Field Operations** - Combine I/O with field simulations
- **Agent Systems** - Save/load agent states with checkpointing
- **Audio Domain** - Use I/O for audio synthesis pipelines
- **Visual Domain** - Integrate with visualization and rendering
