"""Spectrogram Visualization Demo

Demonstrates the new visual.spectrogram() function for audio analysis.
Shows frequency-time representation of various audio signals.
"""

import numpy as np
from morphogen.stdlib import audio, visual, field


def generate_chirp(duration=2.0, sample_rate=44100, f0=200, f1=2000):
    """Generate a chirp signal (frequency sweep)."""
    t = np.linspace(0, duration, int(duration * sample_rate))
    # Linear chirp from f0 to f1
    phase = 2 * np.pi * (f0 * t + (f1 - f0) * t**2 / (2 * duration))
    return np.sin(phase).astype(np.float32)


def generate_harmonic_series(duration=2.0, sample_rate=44100, fundamental=220):
    """Generate a harmonic series (like a musical note)."""
    t = np.linspace(0, duration, int(duration * sample_rate))
    signal = np.zeros_like(t)

    # Add first 8 harmonics with decreasing amplitude
    for n in range(1, 9):
        amplitude = 1.0 / n
        signal += amplitude * np.sin(2 * np.pi * fundamental * n * t)

    return (signal / np.max(np.abs(signal))).astype(np.float32)


def generate_percussive_hit(duration=1.0, sample_rate=44100):
    """Generate a percussive hit with exponential decay."""
    t = np.linspace(0, duration, int(duration * sample_rate))

    # Multiple inharmonic frequencies (like a drum)
    freqs = [100, 187, 314, 503, 689]
    signal = np.zeros_like(t)

    for freq in freqs:
        signal += np.sin(2 * np.pi * freq * t) * np.exp(-10 * t)

    return (signal / np.max(np.abs(signal))).astype(np.float32)


def main():
    print("Spectrogram Visualization Demo")
    print("=" * 50)

    sample_rate = 44100

    # Example 1: Chirp (frequency sweep)
    print("\n1. Generating chirp signal...")
    chirp = generate_chirp(duration=2.0, sample_rate=sample_rate, f0=200, f1=4000)
    audio_buffer = audio.AudioBuffer(chirp, sample_rate=sample_rate)

    print("   Creating spectrogram with 'fire' palette...")
    spec_vis = visual.spectrogram(
        audio_buffer,
        window_size=2048,
        hop_size=512,
        palette="fire",
        log_scale=True
    )

    visual.output(spec_vis, "output_spectrogram_chirp.png")
    print("   Saved: output_spectrogram_chirp.png")

    # Example 2: Harmonic series (musical note)
    print("\n2. Generating harmonic series (A3 = 220 Hz)...")
    harmonic = generate_harmonic_series(duration=2.0, sample_rate=sample_rate, fundamental=220)
    audio_buffer = audio.AudioBuffer(harmonic, sample_rate=sample_rate)

    print("   Creating spectrogram with 'viridis' palette...")
    spec_vis = visual.spectrogram(
        audio_buffer,
        window_size=4096,  # Higher resolution for harmonic analysis
        hop_size=1024,
        palette="viridis",
        log_scale=True,
        freq_range=(0, 3000)  # Focus on first few harmonics
    )

    visual.output(spec_vis, "output_spectrogram_harmonic.png")
    print("   Saved: output_spectrogram_harmonic.png")

    # Example 3: Percussive hit
    print("\n3. Generating percussive hit...")
    percussion = generate_percussive_hit(duration=1.0, sample_rate=sample_rate)
    audio_buffer = audio.AudioBuffer(percussion, sample_rate=sample_rate)

    print("   Creating spectrogram with 'coolwarm' palette...")
    spec_vis = visual.spectrogram(
        audio_buffer,
        window_size=1024,  # Smaller window for better time resolution
        hop_size=256,
        palette="coolwarm",
        log_scale=True
    )

    visual.output(spec_vis, "output_spectrogram_percussion.png")
    print("   Saved: output_spectrogram_percussion.png")

    # Example 4: Multi-tone signal
    print("\n4. Generating multi-tone signal...")
    t = np.linspace(0, 2.0, int(2.0 * sample_rate))
    multitone = (
        np.sin(2 * np.pi * 440 * t) +      # A4
        np.sin(2 * np.pi * 554.37 * t) +   # C#5
        np.sin(2 * np.pi * 659.25 * t)     # E5
    ) / 3.0

    audio_buffer = audio.AudioBuffer(multitone.astype(np.float32), sample_rate=sample_rate)

    print("   Creating spectrogram with 'grayscale' palette...")
    spec_vis = visual.spectrogram(
        audio_buffer,
        window_size=2048,
        hop_size=512,
        palette="grayscale",
        log_scale=False  # Linear scale
    )

    # Add metrics dashboard
    metrics = {
        "Duration": "2.0 s",
        "Sample Rate": "44.1 kHz",
        "Frequencies": "A4, C#5, E5",
        "Window": 2048
    }
    spec_vis = visual.add_metrics(spec_vis, metrics, position="top-right")

    visual.output(spec_vis, "output_spectrogram_multitone.png")
    print("   Saved: output_spectrogram_multitone.png")

    print("\n" + "=" * 50)
    print("Demo complete! Spectrograms showcase:")
    print("  - Frequency-time analysis")
    print("  - Logarithmic (dB) and linear scales")
    print("  - Different palettes for different visualizations")
    print("  - Frequency range filtering")
    print("  - Metrics overlay integration")


if __name__ == "__main__":
    main()
