# Audio Visualizer - Cross-Domain Demo

**Real-time audio visualization using multiple Kairo domains**

## Overview

This example demonstrates how temporal audio data can drive spatial visual effects through cross-domain integration. It showcases four different visualization modes that combine audio processing with field operations, cellular automata, and color mapping.

## Domains Integrated

- **Audio**: Synthesis, signal processing, FFT analysis
- **Field**: Diffusion, spatial operations
- **Cellular**: Audio-reactive cellular automata
- **Palette**: Color mapping and gradients
- **Visual**: Rendering and composition
- **Signal**: Spectral analysis
- **Noise**: Texture generation

## Visualization Modes

### Mode 1: Spectral Field Diffusion
Audio spectral energy creates heat sources that diffuse through a 2D field.

**Cross-domain flow**: Audio → FFT → Field → Palette → Visual

**Key techniques**:
- FFT spectral analysis
- Field-based heat diffusion
- Frequency-based source positioning
- Fire color palette

### Mode 2: Audio-Reactive Cellular Automata
Audio amplitude controls the evolution of cellular automata (Game of Life).

**Cross-domain flow**: Audio → Cellular → Palette → Visual

**Key techniques**:
- Amplitude envelope extraction
- Cell birth controlled by audio energy
- Game of Life rules
- Magma color palette

### Mode 3: Frequency Field Visualization
Different frequency bands (low/mid/high) create distinct field patterns.

**Cross-domain flow**: Audio → Multi-Field → RGB Composite → Visual

**Key techniques**:
- Frequency band separation
- Multiple field layers
- RGB composite visualization
- Spatial diffusion per band

### Mode 4: Beat-Synchronized Patterns
Beat detection triggers synchronized pattern generation.

**Cross-domain flow**: Audio → Beat Detection → Field → Palette → Visual

**Key techniques**:
- Energy-based beat detection
- Radial pattern generation
- Noise texture overlay
- Viridis color palette

## Usage

```bash
# Run the demo
python examples/audio_visualizer/real_time_demo.py
```

## Output

The demo generates:
- `output/test_audio.wav` - Test audio file with chord progression and rhythm
- `output/mode1_spectral_diffusion.png` - Spectral field visualization
- `output/mode2_cellular_automata.png` - Audio-reactive CA
- `output/mode3_frequency_fields.png` - Frequency band visualization
- `output/mode4_beat_patterns.png` - Beat-synchronized patterns

## Key Insights

1. **Temporal → Spatial**: Audio (temporal) naturally drives visual effects (spatial)
2. **Frequency Separation**: Different frequency ranges can control different visual layers
3. **Beat Sync**: Beat detection enables synchronized pattern generation
4. **Field Diffusion**: Creates smooth, organic visual flow from discrete audio events
5. **Emergent Complexity**: Cellular automata add emergent behavior to visualizations

## Extension Ideas

- Add video output with frame-by-frame visualization
- Implement live audio input from microphone
- Create interactive controls for parameters
- Add more visualization modes (particle systems, vector fields)
- Implement 3D visualizations using z-axis for time
- Add MIDI input for musical control

## Technical Details

**Audio Processing**:
- Sample rate: 44.1kHz
- FFT window: 2048-4096 samples
- Hop size: 512-2048 samples

**Visual Resolution**:
- Default: 512×512 pixels
- Adjustable for performance/quality tradeoff

**Performance**:
- Runtime: ~10-30 seconds (depends on mode)
- Memory: ~100-200MB peak

## Related Examples

- `examples/showcase/05_audio_visualizer.py` - Comprehensive audio visualizer with video export
- `examples/signal/` - Signal processing examples
- `examples/cellular/` - Cellular automata examples
- `examples/field/` - Field operation examples

## References

- FFT spectral analysis for audio feature extraction
- Cellular automata: Conway's Game of Life
- Field diffusion: Heat equation solver
- Beat detection: Energy-based onset detection
