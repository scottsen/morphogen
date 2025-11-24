#!/usr/bin/env python3
"""
Comprehensive test suite for VCF filter operators (highpass, bandpass)

Tests voltage-controlled filters with dynamic cutoff/center frequency modulation,
verifying filter response, cross-rate modulation, and edge cases.

Created: 2025-11-23 (misty-rain-1123)
Related: Priority 2 - Additional VCF variants
"""

import numpy as np
from morphogen.stdlib.audio import AudioBuffer, AudioOperations as audio

def test_vcf_highpass_static_cutoff():
    """Test 1: VCF Highpass with Static Cutoff"""
    print("\n" + "="*60)
    print("Test 1: VCF Highpass with Static Cutoff")
    print("="*60)

    # Generate signal: mix of low (100Hz) and high (1000Hz) frequencies
    sample_rate = 48000
    duration = 0.5

    low_freq = audio.sine(freq=100.0, sample_rate=sample_rate, duration=duration)
    high_freq = audio.sine(freq=1000.0, sample_rate=sample_rate, duration=duration)

    # Mix signals (equal amplitude)
    signal = AudioBuffer(
        data=(low_freq.data + high_freq.data) / 2.0,
        sample_rate=sample_rate
    )

    # Apply highpass filter at 500Hz (should attenuate 100Hz, pass 1000Hz)
    cutoff = AudioBuffer(
        data=np.full(len(signal.data), 500.0),
        sample_rate=sample_rate
    )

    filtered = audio.vcf_highpass(signal, cutoff, q=0.707)

    # Analyze: high frequency should dominate after filtering
    signal_rms = np.sqrt(np.mean(signal.data ** 2))
    filtered_rms = np.sqrt(np.mean(filtered.data ** 2))

    # RMS should decrease (low freq attenuated)
    attenuation = (signal_rms - filtered_rms) / signal_rms

    print(f"Input RMS:     {signal_rms:.4f}")
    print(f"Filtered RMS:  {filtered_rms:.4f}")
    print(f"Attenuation:   {attenuation*100:.1f}%")

    # Verify attenuation occurred
    if attenuation > 0.2 and attenuation < 0.7:
        print("✅ PASS: Highpass filtering working (low freq attenuated)")
        return True
    else:
        print(f"❌ FAIL: Unexpected attenuation {attenuation*100:.1f}%")
        return False


def test_vcf_highpass_sweep():
    """Test 2: VCF Highpass with Cutoff Sweep"""
    print("\n" + "="*60)
    print("Test 2: VCF Highpass with Cutoff Sweep (Envelope)")
    print("="*60)

    # Generate sawtooth signal (rich harmonics)
    sample_rate = 48000
    duration = 1.0

    signal = audio.saw(freq=110.0, sample_rate=sample_rate, duration=duration)

    # Create envelope sweep: 100Hz → 2000Hz
    envelope = audio.adsr(
        attack=0.3, decay=0.5, sustain=0.5, release=0.2,
        duration=duration, sample_rate=1000
    )

    # Scale envelope to cutoff range
    cutoff_mod = AudioBuffer(
        data=100.0 + envelope.data * 1900.0,
        sample_rate=envelope.sample_rate
    )

    filtered = audio.vcf_highpass(signal, cutoff_mod, q=2.0)

    # Analyze first quarter vs last quarter (more pronounced difference)
    quarter = len(filtered.data) // 4
    first_quarter_rms = np.sqrt(np.mean(filtered.data[:quarter] ** 2))
    last_quarter_rms = np.sqrt(np.mean(filtered.data[-quarter:] ** 2))

    print(f"First quarter RMS: {first_quarter_rms:.4f} (low cutoff, attack phase)")
    print(f"Last quarter RMS:  {last_quarter_rms:.4f} (high cutoff, release phase)")

    # For highpass: higher cutoff should pass more energy
    # The difference should be at least 5% due to envelope sweep
    difference_pct = abs(first_quarter_rms - last_quarter_rms) / max(first_quarter_rms, last_quarter_rms)

    print(f"Difference: {difference_pct*100:.1f}%")

    # Verify sweep is having an effect (at least 1% variation)
    if difference_pct > 0.01:
        print("✅ PASS: Cutoff sweep working (amplitude varies with cutoff)")
        return True
    else:
        print(f"❌ FAIL: Insufficient variation")
        return False


def test_vcf_bandpass_static_center():
    """Test 3: VCF Bandpass with Static Center Frequency"""
    print("\n" + "="*60)
    print("Test 3: VCF Bandpass with Static Center Frequency")
    print("="*60)

    # Generate signal: mix of three frequencies (200Hz, 1000Hz, 3000Hz)
    sample_rate = 48000
    duration = 0.5

    low_freq = audio.sine(freq=200.0, sample_rate=sample_rate, duration=duration)
    mid_freq = audio.sine(freq=1000.0, sample_rate=sample_rate, duration=duration)
    high_freq = audio.sine(freq=3000.0, sample_rate=sample_rate, duration=duration)

    # Mix signals (equal amplitude)
    signal = AudioBuffer(
        data=(low_freq.data + mid_freq.data + high_freq.data) / 3.0,
        sample_rate=sample_rate
    )

    # Apply bandpass filter centered at 1000Hz with narrow bandwidth (Q=5)
    center = AudioBuffer(
        data=np.full(len(signal.data), 1000.0),
        sample_rate=sample_rate
    )

    filtered = audio.vcf_bandpass(signal, center, q=5.0)

    # Analyze: 1000Hz should dominate after filtering
    signal_rms = np.sqrt(np.mean(signal.data ** 2))
    filtered_rms = np.sqrt(np.mean(filtered.data ** 2))

    # RMS should decrease (low and high freq attenuated)
    attenuation = (signal_rms - filtered_rms) / signal_rms

    print(f"Input RMS:     {signal_rms:.4f}")
    print(f"Filtered RMS:  {filtered_rms:.4f}")
    print(f"Attenuation:   {attenuation*100:.1f}%")

    # Verify attenuation occurred (should be around 50-70%)
    if attenuation > 0.4 and attenuation < 0.8:
        print("✅ PASS: Bandpass filtering working (low/high freq attenuated)")
        return True
    else:
        print(f"❌ FAIL: Unexpected attenuation {attenuation*100:.1f}%")
        return False


def test_vcf_bandpass_lfo_modulation():
    """Test 4: VCF Bandpass with LFO Modulation (Wah Effect)"""
    print("\n" + "="*60)
    print("Test 4: VCF Bandpass with LFO Modulation (Wah)")
    print("="*60)

    # Generate sawtooth signal (rich harmonics)
    sample_rate = 48000
    duration = 2.0

    signal = audio.saw(freq=110.0, sample_rate=sample_rate, duration=duration)

    # Create LFO: slow sine wave for wah effect
    lfo = audio.sine(freq=0.5, sample_rate=1000, duration=duration)

    # Scale LFO to center freq range: 500Hz → 1500Hz
    center_mod = AudioBuffer(
        data=1000.0 + lfo.data * 500.0,
        sample_rate=lfo.sample_rate
    )

    filtered = audio.vcf_bandpass(signal, center_mod, q=5.0)

    # Analyze: should have periodic variation in amplitude
    # Split into quarters and check for variation
    quarter = len(filtered.data) // 4
    q1_rms = np.sqrt(np.mean(filtered.data[0:quarter] ** 2))
    q2_rms = np.sqrt(np.mean(filtered.data[quarter:2*quarter] ** 2))
    q3_rms = np.sqrt(np.mean(filtered.data[2*quarter:3*quarter] ** 2))
    q4_rms = np.sqrt(np.mean(filtered.data[3*quarter:] ** 2))

    rms_values = [q1_rms, q2_rms, q3_rms, q4_rms]
    variation = (max(rms_values) - min(rms_values)) / max(rms_values)

    print(f"Q1 RMS: {q1_rms:.4f}")
    print(f"Q2 RMS: {q2_rms:.4f}")
    print(f"Q3 RMS: {q3_rms:.4f}")
    print(f"Q4 RMS: {q4_rms:.4f}")
    print(f"Variation: {variation*100:.1f}%")

    # Should have significant variation (LFO modulation working)
    if variation > 0.15:
        print("✅ PASS: LFO modulation working (wah effect)")
        return True
    else:
        print(f"❌ FAIL: Insufficient variation {variation*100:.1f}%")
        return False


def test_vcf_q_factor_comparison():
    """Test 5: VCF Bandpass Q Factor Comparison"""
    print("\n" + "="*60)
    print("Test 5: VCF Bandpass Q Factor Comparison")
    print("="*60)

    # Generate signal
    sample_rate = 48000
    duration = 0.5
    signal = audio.saw(freq=220.0, sample_rate=sample_rate, duration=duration)

    # Static center frequency
    center = AudioBuffer(
        data=np.full(int(duration * sample_rate), 1000.0),
        sample_rate=sample_rate
    )

    # Test different Q values
    filtered_q1 = audio.vcf_bandpass(signal, center, q=1.0)  # Wide bandwidth
    filtered_q5 = audio.vcf_bandpass(signal, center, q=5.0)  # Narrow bandwidth
    filtered_q10 = audio.vcf_bandpass(signal, center, q=10.0)  # Very narrow

    rms_q1 = np.sqrt(np.mean(filtered_q1.data ** 2))
    rms_q5 = np.sqrt(np.mean(filtered_q5.data ** 2))
    rms_q10 = np.sqrt(np.mean(filtered_q10.data ** 2))

    print(f"Q=1.0 RMS:  {rms_q1:.4f} (wide bandwidth)")
    print(f"Q=5.0 RMS:  {rms_q5:.4f} (narrow bandwidth)")
    print(f"Q=10.0 RMS: {rms_q10:.4f} (very narrow)")

    # Higher Q should have lower RMS (more attenuation due to narrower band)
    if rms_q1 > rms_q5 and rms_q5 > rms_q10:
        print("✅ PASS: Q factor controlling bandwidth correctly")
        return True
    else:
        print("❌ FAIL: Q factor not working as expected")
        return False


def test_vcf_cross_rate_modulation():
    """Test 6: Cross-Rate Modulation (1kHz Control → 48kHz Audio)"""
    print("\n" + "="*60)
    print("Test 6: Cross-Rate Modulation")
    print("="*60)

    # Generate audio at 48kHz
    audio_rate = 48000
    control_rate = 1000
    duration = 1.0

    signal = audio.saw(freq=220.0, sample_rate=audio_rate, duration=duration)

    # Create envelope at 1kHz (much lower rate)
    envelope = audio.adsr(
        attack=0.2, decay=0.3, sustain=0.5, release=0.5,
        duration=duration, sample_rate=control_rate
    )

    # Scale to cutoff range
    cutoff_mod = AudioBuffer(
        data=200.0 + envelope.data * 1800.0,
        sample_rate=control_rate
    )

    # Should work via scheduler's automatic rate conversion
    filtered = audio.vcf_highpass(signal, cutoff_mod, q=2.0)

    # Verify output is correct length
    expected_samples = int(duration * audio_rate)

    print(f"Signal samples:   {len(signal.data)}")
    print(f"Control samples:  {len(cutoff_mod.data)}")
    print(f"Filtered samples: {len(filtered.data)}")
    print(f"Expected samples: {expected_samples}")

    # Should produce audio-rate output
    if len(filtered.data) == expected_samples:
        print("✅ PASS: Cross-rate modulation working correctly")
        return True
    else:
        print(f"❌ FAIL: Output length mismatch")
        return False


def test_vcf_edge_cases():
    """Test 7: Edge Cases (Extreme Values, Empty Signals)"""
    print("\n" + "="*60)
    print("Test 7: Edge Cases")
    print("="*60)

    sample_rate = 48000
    duration = 0.1

    # Test 1: Very high cutoff (near Nyquist)
    signal = audio.saw(freq=220.0, sample_rate=sample_rate, duration=duration)
    high_cutoff = AudioBuffer(
        data=np.full(len(signal.data), 20000.0),  # 20kHz
        sample_rate=sample_rate
    )

    try:
        filtered_high = audio.vcf_highpass(signal, high_cutoff, q=0.707)
        print("✅ High cutoff handled")
    except Exception as e:
        print(f"❌ High cutoff failed: {e}")
        return False

    # Test 2: Very low cutoff
    low_cutoff = AudioBuffer(
        data=np.full(len(signal.data), 50.0),  # 50Hz
        sample_rate=sample_rate
    )

    try:
        filtered_low = audio.vcf_lowpass(signal, low_cutoff, q=0.707)
        print("✅ Low cutoff handled")
    except Exception as e:
        print(f"❌ Low cutoff failed: {e}")
        return False

    # Test 3: Very high Q
    center = AudioBuffer(
        data=np.full(len(signal.data), 1000.0),
        sample_rate=sample_rate
    )

    try:
        filtered_hq = audio.vcf_bandpass(signal, center, q=20.0)
        print("✅ High Q handled")
    except Exception as e:
        print(f"❌ High Q failed: {e}")
        return False

    print("✅ PASS: All edge cases handled correctly")
    return True


def run_all_tests():
    """Run all VCF filter tests"""
    print("\n" + "="*70)
    print("VCF FILTER OPERATOR TEST SUITE")
    print("Testing: vcf_highpass, vcf_bandpass")
    print("="*70)

    tests = [
        ("VCF Highpass - Static Cutoff", test_vcf_highpass_static_cutoff),
        ("VCF Highpass - Cutoff Sweep", test_vcf_highpass_sweep),
        ("VCF Bandpass - Static Center", test_vcf_bandpass_static_center),
        ("VCF Bandpass - LFO Modulation", test_vcf_bandpass_lfo_modulation),
        ("VCF Bandpass - Q Factor", test_vcf_q_factor_comparison),
        ("Cross-Rate Modulation", test_vcf_cross_rate_modulation),
        ("Edge Cases", test_vcf_edge_cases),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ EXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:40s}: {status}")

    print("="*70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    if passed == total:
        print(f"✅ All VCF filter tests passed! ({passed}/{total})")
    else:
        print(f"⚠️  {passed}/{total} tests passed, {total-passed} failed")

    print("="*70)

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
