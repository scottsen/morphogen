"""
Morphogen Scheduler Demo: End-to-End Multirate Execution

This example demonstrates:
1. Creating a multirate graph (audio + control)
2. Cross-rate connections with automatic resampling
3. Executing the graph with the SimplifiedScheduler
4. Analyzing the output
5. Saving audio to WAV file (optional)

Graph topology:
  sine(440Hz, audio) → multiply ← adsr(control)
                         ↓
                      output (mono)

The ADSR envelope runs at control rate (1kHz) and is automatically
upsampled to audio rate (48kHz) for multiplication.
"""

import numpy as np
from morphogen.graph_ir import GraphIR, GraphIROutputPort
from morphogen.scheduler import SimplifiedScheduler


def create_modulated_synth():
    """Create a simple modulated synth graph"""

    graph = GraphIR(
        sample_rate=48000,
        seed=1337,
        profile="repro",
    )

    # Metadata
    graph.metadata = {
        "source": "scheduler_demo.py",
        "description": "Multirate synth with envelope modulation",
        "author": "Morphogen",
    }

    # 1. Sine oscillator (audio rate: 48kHz)
    graph.add_node(
        id="osc1",
        op="sine",
        outputs=[GraphIROutputPort(name="out", type="Sig")],
        rate="audio",
        params={"freq": "440Hz"},
    )

    # 2. ADSR envelope (control rate: 1kHz)
    graph.add_node(
        id="env1",
        op="adsr",
        outputs=[GraphIROutputPort(name="out", type="Ctl")],
        rate="control",
        params={
            "attack": "0.01s",
            "decay": "0.1s",
            "sustain": 0.7,
            "release": "0.3s",
        },
    )

    # 3. Multiply for amplitude modulation (audio rate)
    graph.add_node(
        id="mul1",
        op="multiply",
        outputs=[GraphIROutputPort(name="out", type="Sig")],
        rate="audio",
    )

    # Create connections
    graph.add_edge(from_port="osc1:out", to_port="mul1:in1", type="Sig")
    graph.add_edge(from_port="env1:out", to_port="mul1:in2", type="Ctl")  # Cross-rate!

    # Define output
    graph.add_output("mono", ["mul1:out"])

    return graph


def analyze_output(output: np.ndarray, sample_rate: int):
    """Analyze output signal"""
    duration = len(output) / sample_rate

    print("Output Analysis:")
    print(f"  Duration:    {duration:.3f} seconds")
    print(f"  Samples:     {len(output)}")
    print(f"  Sample Rate: {sample_rate} Hz")
    print(f"  Min value:   {np.min(output):.4f}")
    print(f"  Max value:   {np.max(output):.4f}")
    print(f"  Mean:        {np.mean(output):.4f}")
    print(f"  RMS:         {np.sqrt(np.mean(output**2)):.4f}")
    print(f"  Peak:        {np.max(np.abs(output)):.4f}")


def save_to_wav(output: np.ndarray, filename: str, sample_rate: int = 48000):
    """Save output to WAV file (requires scipy)"""
    try:
        from scipy.io import wavfile

        # Normalize to int16 range
        normalized = np.int16(output / np.max(np.abs(output)) * 32767)

        wavfile.write(filename, sample_rate, normalized)
        print(f"\n✓ Audio saved to {filename}")
        return True
    except ImportError:
        print("\n⚠ scipy not available - skipping WAV export")
        print("  Install with: pip install scipy")
        return False


def main():
    """Main demo"""
    print("=" * 70)
    print("Morphogen Scheduler Demo: Multirate Execution")
    print("=" * 70)
    print()

    # Create the graph
    print("Step 1: Creating multirate graph...")
    graph = create_modulated_synth()
    print(f"✓ Graph created with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    print()

    # Display graph structure
    print("Graph Structure:")
    print("\nNodes:")
    for node in graph.nodes:
        rate_info = f"{node.rate} ({SimplifiedScheduler.DEFAULT_RATES[node.rate]} Hz)"
        print(f"  {node.id:10s} [{rate_info:20s}] {node.op}")

    print("\nEdges (connections):")
    for edge in graph.edges:
        from_node = graph.get_node(edge.from_node)
        to_node = graph.get_node(edge.to_node)
        from_rate = from_node.rate if from_node else "?"
        to_rate = to_node.rate if to_node else "?"

        cross_rate = " (CROSS-RATE!)" if from_rate != to_rate else ""
        print(f"  {edge.from_port:15s} → {edge.to_port:15s} [{from_rate} → {to_rate}]{cross_rate}")
    print()

    # Validate
    print("Step 2: Validating graph...")
    errors = graph.validate()
    if errors:
        print("✗ Validation FAILED:")
        for err in errors:
            print(f"  - {err}")
        return
    else:
        print("✓ Graph is valid")
    print()

    # Create scheduler
    print("Step 3: Initializing scheduler...")
    scheduler = SimplifiedScheduler(graph, hop_size=128)
    print("✓ Scheduler initialized")
    print()

    # Display scheduler info
    info = scheduler.get_info()
    print("Scheduler Configuration:")
    print(f"  Sample Rate:  {info['sample_rate']} Hz")
    print(f"  Hop Size:     {info['hop_size']} samples")
    print(f"  Master Clock: {info['master_rate']} Hz (GCD of all rates)")
    print()

    print("Rate Groups:")
    for rg in info['rate_groups']:
        print(f"  {rg['rate']:10s} @ {rg['rate_hz']:6.0f} Hz (multiplier: {rg['multiplier']:2d})")
        print(f"    Operators: {', '.join(rg['operators'])}")
    print()

    # Execute
    print("Step 4: Executing graph...")
    duration_seconds = 1.0
    print(f"  Executing for {duration_seconds} second(s)...")

    outputs = scheduler.execute(duration_seconds=duration_seconds)

    print(f"✓ Execution complete")
    print()

    # Analyze output
    print("Step 5: Analyzing output...")
    mono_output = outputs["mono"]
    analyze_output(mono_output, info['sample_rate'])
    print()

    # Save to WAV
    print("Step 6: Saving output...")
    save_to_wav(mono_output, "scheduler_demo.wav", info['sample_rate'])
    print()

    # Summary
    print("=" * 70)
    print("Demo Summary")
    print("=" * 70)
    print()
    print("✓ Multirate graph created (audio @ 48kHz + control @ 1kHz)")
    print("✓ Cross-rate connection handled (control → audio with linear resampling)")
    print("✓ Graph validated successfully")
    print("✓ Scheduler executed graph for 1 second")
    print(f"✓ Generated {len(mono_output)} audio samples")
    print()
    print("Key Features Demonstrated:")
    print("  • GCD master clock (1kHz = GCD(48kHz, 1kHz))")
    print("  • Rate group partitioning (audio and control separated)")
    print("  • Topological execution order (osc → multiply ← env)")
    print("  • Linear resampling (control 1kHz → audio 48kHz)")
    print("  • Hop-based execution (128-sample blocks)")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
