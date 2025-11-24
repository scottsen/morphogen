#!/usr/bin/env python3
"""
VCF Filter Demonstration: Highpass and Bandpass Filters

Demonstrates voltage-controlled filters with dynamic modulation:
- VCF Highpass with envelope sweep
- VCF Bandpass with LFO modulation (wah effect)

Created: 2025-11-23 (misty-rain-1123)
Related: Priority 2 - Additional VCF variants
"""

import numpy as np
from morphogen.stdlib.audio import AudioBuffer, AudioOperations as audio

def demo_vcf_highpass():
    """Demo 1: VCF Highpass with Envelope Sweep"""
    print("\n" + "="*60)
    print("Demo 1: VCF Highpass with Envelope Sweep")
    print("="*60)

    # Parameters
    sample_rate = 48000
    duration = 3.0

    # Generate sawtooth oscillator (rich harmonics)
    print("Generating sawtooth oscillator at 110 Hz...")
    signal = audio.saw(freq=110.0, sample_rate=sample_rate, duration=duration)

    # Create ADSR envelope for filter cutoff modulation
    print("Creating ADSR envelope (1kHz control rate)...")
    envelope = audio.adsr(
        attack=0.5,   # Slow attack
        decay=0.8,    # Gradual decay
        sustain=0.4,  # Lower sustain
        release=0.7,  # Smooth release
        duration=duration,
        sample_rate=1000  # 1kHz envelope (efficient)
    )

    # Scale envelope to cutoff range: 100Hz → 3000Hz
    print("Scaling envelope to cutoff range (100Hz - 3000Hz)...")
    cutoff_mod = AudioBuffer(
        data=100.0 + envelope.data * 2900.0,
        sample_rate=envelope.sample_rate
    )

    # Apply highpass filter with Q=2 for some resonance
    print("Applying VCF highpass filter (Q=2.0)...")
    filtered = audio.vcf_highpass(signal, cutoff_mod, q=2.0)

    # Analyze results
    signal_rms = np.sqrt(np.mean(signal.data ** 2))
    filtered_rms = np.sqrt(np.mean(filtered.data ** 2))

    print(f"\nResults:")
    print(f"  Duration:      {duration}s")
    print(f"  Sample Rate:   {sample_rate} Hz")
    print(f"  Signal RMS:    {signal_rms:.4f}")
    print(f"  Filtered RMS:  {filtered_rms:.4f}")
    print(f"  Attenuation:   {(1 - filtered_rms/signal_rms)*100:.1f}%")
    print(f"\nFilter sweep: Low frequencies attenuated during attack,")
    print(f"              More passing through during decay/sustain")

    return filtered


def demo_vcf_bandpass():
    """Demo 2: VCF Bandpass with LFO (Wah Effect)"""
    print("\n" + "="*60)
    print("Demo 2: VCF Bandpass with LFO Modulation (Wah)")
    print("="*60)

    # Parameters
    sample_rate = 48000
    duration = 4.0

    # Generate sawtooth oscillator
    print("Generating sawtooth oscillator at 220 Hz...")
    signal = audio.saw(freq=220.0, sample_rate=sample_rate, duration=duration)

    # Create slow LFO for wah effect (0.5 Hz = one cycle per 2 seconds)
    print("Creating LFO at 0.5 Hz (1kHz control rate)...")
    lfo = audio.sine(freq=0.5, sample_rate=1000, duration=duration)

    # Scale LFO to center frequency range: 400Hz → 2000Hz
    print("Scaling LFO to center frequency range (400Hz - 2000Hz)...")
    center_mod = AudioBuffer(
        data=1200.0 + lfo.data * 800.0,
        sample_rate=lfo.sample_rate
    )

    # Apply bandpass filter with high Q for pronounced wah
    print("Applying VCF bandpass filter (Q=8.0 for narrow band)...")
    filtered = audio.vcf_bandpass(signal, center_mod, q=8.0)

    # Analyze results
    signal_rms = np.sqrt(np.mean(signal.data ** 2))
    filtered_rms = np.sqrt(np.mean(filtered.data ** 2))

    # Measure variation across quarters
    quarter = len(filtered.data) // 4
    q1_rms = np.sqrt(np.mean(filtered.data[0:quarter] ** 2))
    q2_rms = np.sqrt(np.mean(filtered.data[quarter:2*quarter] ** 2))
    q3_rms = np.sqrt(np.mean(filtered.data[2*quarter:3*quarter] ** 2))
    q4_rms = np.sqrt(np.mean(filtered.data[3*quarter:] ** 2))

    variation = (max([q1_rms, q2_rms, q3_rms, q4_rms]) -
                 min([q1_rms, q2_rms, q3_rms, q4_rms]))

    print(f"\nResults:")
    print(f"  Duration:      {duration}s")
    print(f"  Sample Rate:   {sample_rate} Hz")
    print(f"  Signal RMS:    {signal_rms:.4f}")
    print(f"  Filtered RMS:  {filtered_rms:.4f}")
    print(f"  Q1 RMS:        {q1_rms:.4f}")
    print(f"  Q2 RMS:        {q2_rms:.4f}")
    print(f"  Q3 RMS:        {q3_rms:.4f}")
    print(f"  Q4 RMS:        {q4_rms:.4f}")
    print(f"  Variation:     {variation:.4f} ({variation/filtered_rms*100:.1f}%)")
    print(f"\nWah effect: Center frequency sweeps from 400Hz to 2000Hz")
    print(f"            Amplitude varies periodically with LFO")

    return filtered


def demo_q_factor_comparison():
    """Demo 3: Q Factor Comparison for Bandpass"""
    print("\n" + "="*60)
    print("Demo 3: Q Factor Comparison (Bandpass)")
    print("="*60)

    # Parameters
    sample_rate = 48000
    duration = 1.0

    # Generate sawtooth
    print("Generating sawtooth oscillator at 220 Hz...")
    signal = audio.saw(freq=220.0, sample_rate=sample_rate, duration=duration)

    # Static center frequency
    center = AudioBuffer(
        data=np.full(int(duration * sample_rate), 1000.0),
        sample_rate=sample_rate
    )

    # Test different Q values
    print("Filtering with different Q values...")
    filtered_q1 = audio.vcf_bandpass(signal, center, q=1.0)   # Wide
    filtered_q5 = audio.vcf_bandpass(signal, center, q=5.0)   # Narrow
    filtered_q10 = audio.vcf_bandpass(signal, center, q=10.0) # Very narrow

    # Analyze
    rms_q1 = np.sqrt(np.mean(filtered_q1.data ** 2))
    rms_q5 = np.sqrt(np.mean(filtered_q5.data ** 2))
    rms_q10 = np.sqrt(np.mean(filtered_q10.data ** 2))

    print(f"\nResults (Center: 1000Hz):")
    print(f"  Q=1.0  RMS: {rms_q1:.4f}  (wide bandwidth)")
    print(f"  Q=5.0  RMS: {rms_q5:.4f}  (narrow bandwidth)")
    print(f"  Q=10.0 RMS: {rms_q10:.4f} (very narrow)")
    print(f"\nHigher Q = Narrower bandwidth = More attenuation")
    print(f"Q=1.0 passes {rms_q1/rms_q10:.1f}x more energy than Q=10.0")


def main():
    """Run all VCF filter demonstrations"""
    print("\n" + "="*70)
    print("VCF FILTER DEMONSTRATIONS")
    print("Voltage-Controlled Highpass and Bandpass Filters")
    print("="*70)

    # Demo 1: Highpass with envelope
    filtered_hp = demo_vcf_highpass()

    # Demo 2: Bandpass with LFO (wah)
    filtered_bp = demo_vcf_bandpass()

    # Demo 3: Q factor comparison
    demo_q_factor_comparison()

    print("\n" + "="*70)
    print("KEY TAKEAWAYS")
    print("="*70)
    print("1. VCF Highpass: Dynamically remove low frequencies")
    print("   - Envelope sweep creates evolving timbre")
    print("   - Higher cutoff = more frequencies pass through")
    print()
    print("2. VCF Bandpass: Dynamically isolate frequency range")
    print("   - LFO modulation creates wah/vowel effects")
    print("   - High Q = narrow band, pronounced effect")
    print()
    print("3. Q Factor: Controls filter resonance/bandwidth")
    print("   - Low Q (0.5-2): Wide, gentle filtering")
    print("   - Medium Q (2-5): Balanced, musical")
    print("   - High Q (5-20): Narrow, resonant, pronounced")
    print()
    print("4. Cross-Rate Modulation: Efficient control signals")
    print("   - 1kHz envelopes/LFOs control 48kHz audio")
    print("   - Scheduler handles automatic rate conversion")
    print("="*70)

    print("\n✅ All demonstrations complete!")
    print("\nNext steps:")
    print("  - Try different Q values to hear the difference")
    print("  - Experiment with envelope shapes (attack/decay/sustain)")
    print("  - Combine with VCA for classic subtractive synthesis")
    print("  - Use multiple filters in series for complex timbres")


if __name__ == "__main__":
    main()
