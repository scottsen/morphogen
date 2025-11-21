"""
Example: FFT Spectrum Analysis

Demonstrates Fast Fourier Transform for frequency analysis of signals.
Analyzes a complex signal composed of multiple sine waves plus noise.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from morphogen.stdlib.signal import SignalOperations, WindowType
import matplotlib.pyplot as plt
import numpy as np


def main():
    """Analyze signal frequency content using FFT"""

    # Signal parameters
    sample_rate = 1000.0  # Hz
    duration = 2.0  # seconds

    print("=== FFT Spectrum Analysis ===\n")

    # Create a complex signal: sum of 3 sine waves + noise
    freq1, freq2, freq3 = 50.0, 120.0, 200.0
    sig1 = SignalOperations.sine_wave(freq1, duration, sample_rate, amplitude=1.0)
    sig2 = SignalOperations.sine_wave(freq2, duration, sample_rate, amplitude=0.5)
    sig3 = SignalOperations.sine_wave(freq3, duration, sample_rate, amplitude=0.3)
    noise = SignalOperations.white_noise(duration, sample_rate, amplitude=0.1, seed=42)

    # Combine signals
    combined_signal = SignalOperations.create_signal(
        sig1.data + sig2.data + sig3.data + noise.data,
        sample_rate
    )

    print(f"Signal parameters:")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {duration} s")
    print(f"  Samples: {len(combined_signal.data)}")
    print(f"\nSignal composition:")
    print(f"  • {freq1} Hz sine wave (amplitude 1.0)")
    print(f"  • {freq2} Hz sine wave (amplitude 0.5)")
    print(f"  • {freq3} Hz sine wave (amplitude 0.3)")
    print(f"  • White noise (amplitude 0.1)")

    # Compute FFT without windowing
    spectrum_rect = SignalOperations.fft(combined_signal)

    # Apply Hann window and compute FFT
    windowed_signal = SignalOperations.window(combined_signal, WindowType.HANN)
    spectrum_hann = SignalOperations.fft(windowed_signal)

    # Find peaks in the spectrum
    print("\n=== Detected Frequencies (Hann window) ===")
    positive_freqs = spectrum_hann.frequencies >= 0
    freqs = spectrum_hann.frequencies[positive_freqs]
    mags = spectrum_hann.magnitude[positive_freqs]

    # Find peaks
    peaks = SignalOperations.peak_detection(
        SignalOperations.create_signal(mags, 1.0),
        height=10.0,
        distance=20
    )

    for peak_idx in peaks:
        freq = freqs[peak_idx]
        magnitude = mags[peak_idx]
        if freq > 0 and freq < sample_rate / 2:  # Nyquist limit
            print(f"  • {freq:.1f} Hz (magnitude: {magnitude:.1f})")

    # Visualize
    visualize_fft_analysis(combined_signal, spectrum_rect, spectrum_hann,
                          [freq1, freq2, freq3])


def visualize_fft_analysis(signal, spectrum_rect, spectrum_hann, true_freqs):
    """Visualize signal and its FFT spectrum"""

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Plot 1: Time domain signal
    ax = axes[0]
    time = signal.time_axis
    ax.plot(time, signal.data, linewidth=0.5, color='steelblue')
    ax.set_xlabel('Time (s)', fontsize=11)
    ax.set_ylabel('Amplitude', fontsize=11)
    ax.set_title('Time Domain Signal (Multiple Frequencies + Noise)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.2)  # Show only first 200ms for clarity

    # Plot 2: FFT Magnitude (Rectangular window)
    ax = axes[1]
    positive_freqs = spectrum_rect.frequencies >= 0
    freqs = spectrum_rect.frequencies[positive_freqs]
    mags = spectrum_rect.magnitude[positive_freqs]

    ax.plot(freqs, mags, linewidth=1, color='coral')

    # Mark true frequencies
    for freq in true_freqs:
        ax.axvline(freq, color='red', linestyle='--', alpha=0.5, linewidth=1.5)

    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Magnitude', fontsize=11)
    ax.set_title('FFT Magnitude Spectrum (Rectangular Window)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 300)

    # Plot 3: FFT Magnitude (Hann window)
    ax = axes[2]
    freqs_hann = spectrum_hann.frequencies[positive_freqs]
    mags_hann = spectrum_hann.magnitude[positive_freqs]

    ax.plot(freqs_hann, mags_hann, linewidth=1, color='seagreen')

    # Mark true frequencies
    for freq in true_freqs:
        ax.axvline(freq, color='red', linestyle='--', alpha=0.5, linewidth=1.5,
                  label=f'{freq} Hz' if freq == true_freqs[0] else '')

    ax.set_xlabel('Frequency (Hz)', fontsize=11)
    ax.set_ylabel('Magnitude', fontsize=11)
    ax.set_title('FFT Magnitude Spectrum (Hann Window - Better Frequency Resolution)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 300)
    ax.legend(['Spectrum', 'True Frequencies'], loc='upper right')

    plt.suptitle('FFT Spectrum Analysis', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('/tmp/signal_fft_analysis.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: /tmp/signal_fft_analysis.png")
    plt.show()


if __name__ == "__main__":
    main()
