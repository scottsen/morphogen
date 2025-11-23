#!/usr/bin/env python3
"""
Quick test of operator integration.

Tests that the scheduler can execute real audio operators instead of mocks.
"""

import sys
import numpy as np
from morphogen.graph_ir import GraphIR, GraphIRNode, GraphIREdge, GraphIROutputPort
from morphogen.scheduler.simplified import SimplifiedScheduler


def test_sine_operator():
    """Test that sine operator works through scheduler."""
    print("\n=== Test 1: Sine Operator ===")

    # Create simple graph with sine oscillator
    graph = GraphIR(version="1.0")

    # Add sine node
    sine_node = GraphIRNode(
        id="sine1",
        op="sine",
        rate="audio",
        params={"freq": "440Hz"},
        outputs=[GraphIROutputPort(name="out", type="Sig")]
    )
    graph.nodes.append(sine_node)

    # Add to graph outputs (list of port refs)
    if not hasattr(graph, 'outputs') or not isinstance(graph.outputs, dict):
        graph.outputs = {}
    graph.outputs = {"audio_out": ["sine1:out"]}

    # Validate
    errors = graph.validate()
    if errors:
        print(f"❌ Validation errors: {errors}")
        return False

    print(f"✅ Graph validated")

    # Create scheduler
    scheduler = SimplifiedScheduler(graph, sample_rate=48000)
    print(f"✅ Scheduler created")
    print(f"   Operators discovered: {len(scheduler.operator_registry.list_operators())}")
    print(f"   Sample operators: {scheduler.operator_registry.list_operators()[:10]}")

    # Execute for 1 second
    output = scheduler.execute(duration_seconds=1.0)

    # Check output
    if "audio_out" not in output:
        print(f"❌ No audio_out in outputs: {output.keys()}")
        return False

    audio = output["audio_out"]
    print(f"✅ Generated {len(audio)} samples")
    print(f"   RMS: {np.sqrt(np.mean(audio**2)):.4f} (expected ~0.707)")
    print(f"   Peak: {np.max(np.abs(audio)):.4f} (expected ~1.0)")
    print(f"   Mean: {np.mean(audio):.4f} (expected ~0.0)")

    # Check basic properties
    rms = np.sqrt(np.mean(audio**2))
    if not (0.6 < rms < 0.8):
        print(f"❌ RMS out of range: {rms}")
        return False

    print(f"✅ Sine operator test passed!")
    return True


def test_adsr_operator():
    """Test ADSR envelope operator."""
    print("\n=== Test 2: ADSR Envelope ===")

    # Create graph with ADSR envelope
    graph = GraphIR(version="1.0")

    # ADSR envelope (control rate)
    adsr_node = GraphIRNode(
        id="env1",
        op="adsr",
        rate="control",
        params={"attack": "0.1", "decay": "0.1", "sustain": "0.7", "release": "0.2"},
        outputs=[GraphIROutputPort(name="out", type="Ctl")]
    )
    graph.nodes.append(adsr_node)

    # Output
    graph.outputs = {"envelope_out": ["env1:out"]}

    # Validate
    errors = graph.validate()
    if errors:
        print(f"❌ Validation errors: {errors}")
        return False

    print(f"✅ Graph validated")

    # Create scheduler
    scheduler = SimplifiedScheduler(graph, sample_rate=48000)
    print(f"✅ Scheduler created")

    # Execute for 1.0 second
    output = scheduler.execute(duration_seconds=1.0)

    # Check output
    if "envelope_out" not in output:
        print(f"❌ No envelope_out in outputs")
        return False

    envelope = output["envelope_out"]
    print(f"✅ Generated {len(envelope)} samples (control rate)")
    print(f"   Max: {np.max(envelope):.4f} (expected ~1.0 at attack peak)")
    print(f"   Min: {np.min(envelope):.4f}")
    print(f"   Final: {envelope[-1]:.4f} (should be near 0 after release)")

    # Check envelope properties
    max_val = np.max(envelope)
    if not (0.9 < max_val < 1.1):
        print(f"❌ Max envelope value out of range: {max_val}")
        return False

    print(f"✅ ADSR operator test passed!")
    return True


def test_filter_operator():
    """Test lowpass filter operator."""
    print("\n=== Test 3: Lowpass Filter ===")

    # Create graph: saw -> lowpass
    graph = GraphIR(version="1.0")

    # Sawtooth oscillator
    saw_node = GraphIRNode(
        id="saw1",
        op="saw",
        rate="audio",
        params={"freq": "110Hz"},
        outputs=[GraphIROutputPort(name="out", type="Sig")]
    )
    graph.nodes.append(saw_node)

    # Lowpass filter
    lpf_node = GraphIRNode(
        id="lpf1",
        op="lowpass",
        rate="audio",
        params={"cutoff": "1000Hz", "q": "0.707"},
        outputs=[GraphIROutputPort(name="out", type="Sig")]
    )
    graph.nodes.append(lpf_node)

    # Connect
    graph.edges.append(GraphIREdge(
        from_port="saw1:out",
        to_port="lpf1:signal",
        type="Sig"
    ))

    # Output
    graph.outputs = {"audio_out": ["lpf1:out"]}

    # Validate
    errors = graph.validate()
    if errors:
        print(f"❌ Validation errors: {errors}")
        return False

    print(f"✅ Graph validated")

    # Create scheduler
    scheduler = SimplifiedScheduler(graph, sample_rate=48000)
    print(f"✅ Scheduler created")

    # Execute for 0.5 seconds
    output = scheduler.execute(duration_seconds=0.5)

    # Check output
    if "audio_out" not in output:
        print(f"❌ No audio_out in outputs")
        return False

    audio = output["audio_out"]
    print(f"✅ Generated {len(audio)} samples")
    print(f"   RMS: {np.sqrt(np.mean(audio**2)):.4f}")
    print(f"   Peak: {np.max(np.abs(audio)):.4f}")

    print(f"✅ Lowpass filter test passed!")
    return True


if __name__ == "__main__":
    print("="*60)
    print("Testing Morphogen Operator Integration")
    print("="*60)

    results = []

    # Run tests
    try:
        results.append(("Sine", test_sine_operator()))
    except Exception as e:
        print(f"❌ Sine test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Sine", False))

    try:
        results.append(("ADSR", test_adsr_operator()))
    except Exception as e:
        print(f"❌ ADSR test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("ADSR", False))

    try:
        results.append(("Filter", test_filter_operator()))
    except Exception as e:
        print(f"❌ Filter test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Filter", False))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{name:20s} {status}")

    total_passed = sum(1 for _, p in results if p)
    total_tests = len(results)
    print(f"\nPassed: {total_passed}/{total_tests}")

    sys.exit(0 if all(p for _, p in results) else 1)
