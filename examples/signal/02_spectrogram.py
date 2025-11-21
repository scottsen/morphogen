"""
Example: Spectrogram Analysis

Demonstrates Short-Time Fourier Transform (STFT) for time-frequency analysis.
Creates a chirp signal and visualizes its changing frequency content over time.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from morphogen.stdlib.signal import SignalOperations, WindowType
import matplotlib.pyplot as plt
import numpy as np


def main():
    """Analyze time-varying frequency content using STFT"""

    # Signal parameters
    sample_rate = 4000.0  # Hz
    duration = 3.0  # seconds

    print("=== Spectrogram Analysis ===\n")

    # Create chirp signal (frequency sweep from 100 Hz to 1000 Hz)
    f0, f1 = 100.0, 1000.0
    chirp_signal = SignalOperations.chirp(f0, f1, duration, sample_rate, method='linear')

    print(f"Chirp signal:")
    print(f"  Frequency sweep: {f0} Hz → {f1} Hz")
    print(f"  Duration: {duration} s")
    print(f"  Sample rate: {sample_rate} Hz")

    # Add a stationary component at 500 Hz
    stationary = SignalOperations.sine_wave(500.0, duration, sample_rate, amplitude=0.3)
    combined_signal = SignalOperations.create_signal(
        chirp_signal.data + stationary.data,
        sample_rate
    )

    print(f"\nCombined signal:")
    print(f"  Chirp {f0}-{f1} Hz + steady 500 Hz tone")

    # Compute STFT with different window sizes
    window_size_small = 128  # Better time resolution
    window_size_large = 512  # Better frequency resolution

    spec_small = SignalOperations.stft(combined_signal, window_size=window_size_small,
                                      window_type=WindowType.HANN)
    spec_large = SignalOperations.stft(combined_signal, window_size=window_size_large,
                                      window_type=WindowType.HANN)

    print(f"\n=== STFT Parameters ===")
    print(f"Small window (better time resolution):")
    print(f"  Window size: {window_size_small} samples ({window_size_small/sample_rate*1000:.1f} ms)")
    print(f"  Frequency resolution: {sample_rate/window_size_small:.1f} Hz")
    print(f"  Time frames: {len(spec_small.times)}")

    print(f"\nLarge window (better frequency resolution):")
    print(f"  Window size: {window_size_large} samples ({window_size_large/sample_rate*1000:.1f} ms)")
    print(f"  Frequency resolution: {sample_rate/window_size_large:.1f} Hz")
    print(f"  Time frames: {len(spec_large.times)}")

    # Compute power spectrogram
    power_spec = SignalOperations.spectrogram_power(combined_signal, window_size=256)

    # Visualize
    visualize_spectrogram(combined_signal, spec_small, spec_large, power_spec, f0, f1)


def visualize_spectrogram(signal, spec_small, spec_large, power_spec, f0, f1):
    """Visualize signal and spectrograms"""

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Time domain signal (full width)
    ax1 = fig.add_subplot(gs[0, :])
    time = signal.time_axis
    ax1.plot(time, signal.data, linewidth=0.5, color='steelblue')
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_title('Time Domain Signal (Chirp + Steady Tone)', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Spectrogram with small window
    ax2 = fig.add_subplot(gs[1, 0])
    extent = [spec_small.times[0], spec_small.times[-1],
             spec_small.frequencies[0], spec_small.frequencies[-1]]

    im = ax2.imshow(spec_small.magnitude.T, aspect='auto', origin='lower',
                    cmap='viridis', extent=extent, interpolation='bilinear')

    # Plot theoretical chirp line
    t_chirp = spec_small.times
    f_chirp = f0 + (f1 - f0) * (t_chirp / signal.duration)
    ax2.plot(t_chirp, f_chirp, 'r--', linewidth=2, label='Theoretical chirp', alpha=0.7)
    ax2.axhline(500, color='red', linestyle=':', linewidth=2, label='500 Hz tone', alpha=0.7)

    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Frequency (Hz)', fontsize=11)
    ax2.set_title('STFT (Small Window - Better Time Resolution)', fontweight='bold')
    ax2.set_ylim(0, 1500)
    ax2.legend(loc='upper left', fontsize=9)
    plt.colorbar(im, ax=ax2, label='Magnitude')

    # Plot 3: Spectrogram with large window
    ax3 = fig.add_subplot(gs[1, 1])
    extent = [spec_large.times[0], spec_large.times[-1],
             spec_large.frequencies[0], spec_large.frequencies[-1]]

    im = ax3.imshow(spec_large.magnitude.T, aspect='auto', origin='lower',
                    cmap='viridis', extent=extent, interpolation='bilinear')

    # Plot theoretical chirp line
    t_chirp = spec_large.times
    f_chirp = f0 + (f1 - f0) * (t_chirp / signal.duration)
    ax3.plot(t_chirp, f_chirp, 'r--', linewidth=2, label='Theoretical chirp', alpha=0.7)
    ax3.axhline(500, color='red', linestyle=':', linewidth=2, label='500 Hz tone', alpha=0.7)

    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_ylabel('Frequency (Hz)', fontsize=11)
    ax3.set_title('STFT (Large Window - Better Frequency Resolution)', fontweight='bold')
    ax3.set_ylim(0, 1500)
    ax3.legend(loc='upper left', fontsize=9)
    plt.colorbar(im, ax=ax3, label='Magnitude')

    # Plot 4: Power spectrogram (dB scale) (full width)
    ax4 = fig.add_subplot(gs[2, :])
    extent = [power_spec.times[0], power_spec.times[-1],
             power_spec.frequencies[0], power_spec.frequencies[-1]]

    im = ax4.imshow(power_spec.data.T, aspect='auto', origin='lower',
                    cmap='hot', extent=extent, interpolation='bilinear',
                    vmin=-60, vmax=0)

    # Plot theoretical chirp line
    t_chirp = power_spec.times
    f_chirp = f0 + (f1 - f0) * (t_chirp / signal.duration)
    ax4.plot(t_chirp, f_chirp, 'cyan', linewidth=2, label='Theoretical chirp', alpha=0.9)
    ax4.axhline(500, color='cyan', linestyle=':', linewidth=2, label='500 Hz tone', alpha=0.9)

    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_ylabel('Frequency (Hz)', fontsize=11)
    ax4.set_title('Power Spectrogram (dB scale)', fontweight='bold')
    ax4.set_ylim(0, 1500)
    ax4.legend(loc='upper left', fontsize=9)
    plt.colorbar(im, ax=ax4, label='Power (dB)')

    plt.suptitle('STFT Spectrogram Analysis - Time-Frequency Tradeoff',
                fontsize=14, fontweight='bold')
    plt.savefig('/tmp/signal_spectrogram.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to: /tmp/signal_spectrogram.png")
    plt.show()


if __name__ == "__main__":
    main()
