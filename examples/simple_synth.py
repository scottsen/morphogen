"""
Example: Creating a simple synth using GraphIR

This example demonstrates:
1. Creating a graph programmatically
2. Using multirate (audio + control)
3. Cross-rate connections (ADSR envelope → audio multiply)
4. JSON serialization
5. Validation

Graph topology:
  sine(440Hz) → lpf(2kHz) → multiply ← adsr(envelope)
                               ↓
                            output (mono)
"""

from morphogen.graph_ir import GraphIR, GraphIROutputPort


def create_simple_synth():
    """Create a simple synth graph"""

    # Create graph
    graph = GraphIR(
        sample_rate=48000,
        seed=1337,
        profile="repro",
    )

    # Add metadata
    graph.metadata = {
        "source": "example script",
        "description": "Simple subtractive synth with ADSR envelope",
        "author": "Morphogen",
    }

    # 1. Sine oscillator (audio rate)
    graph.add_node(
        id="osc1",
        op="sine",
        outputs=[GraphIROutputPort(name="out", type="Sig")],
        rate="audio",
        params={"freq": "440Hz", "phase": 0.0},
    )

    # 2. Low-pass filter (audio rate)
    graph.add_node(
        id="lpf1",
        op="lpf",
        outputs=[GraphIROutputPort(name="out", type="Sig")],
        rate="audio",
        params={"cutoff": "2kHz", "q": 0.707},
    )

    # 3. ADSR envelope (control rate)
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

    # 4. Multiply for amplitude modulation (audio rate)
    graph.add_node(
        id="mul1",
        op="multiply",
        outputs=[GraphIROutputPort(name="out", type="Sig")],
        rate="audio",
        params={},
    )

    # Create connections
    graph.add_edge(from_port="osc1:out", to_port="lpf1:in", type="Sig")
    graph.add_edge(from_port="lpf1:out", to_port="mul1:in1", type="Sig")
    graph.add_edge(from_port="env1:out", to_port="mul1:in2", type="Ctl")  # Cross-rate!

    # Define output
    graph.add_output("mono", ["mul1:out"])

    return graph


def main():
    """Main example"""
    print("=" * 70)
    print("Morphogen GraphIR Example: Simple Synth")
    print("=" * 70)
    print()

    # Create the graph
    print("Creating graph...")
    graph = create_simple_synth()
    print(f"✓ Graph created with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    print()

    # Validate
    print("Validating graph...")
    errors = graph.validate()
    if errors:
        print("✗ Validation FAILED:")
        for err in errors:
            print(f"  - {err}")
        return
    else:
        print("✓ Validation PASSED")
    print()

    # Display graph info
    print("Graph Information:")
    print(f"  Version:     {graph.version}")
    print(f"  Profile:     {graph.profile}")
    print(f"  Sample Rate: {graph.sample_rate} Hz")
    print(f"  Seed:        {graph.seed}")
    print()

    print("Nodes:")
    for node in graph.nodes:
        print(f"  {node.id:10s} [{node.rate:8s}] {node.op}")
        for param, value in node.params.items():
            print(f"    - {param}: {value}")
    print()

    print("Edges:")
    for edge in graph.edges:
        print(f"  {edge.from_port:15s} → {edge.to_port:15s} ({edge.type})")
    print()

    print("Outputs:")
    for name, refs in graph.outputs.items():
        print(f"  {name}: {', '.join(refs)}")
    print()

    # Save to JSON
    output_file = "simple_synth.morph.json"
    print(f"Saving to {output_file}...")
    graph.to_json(output_file)
    print(f"✓ Saved to {output_file}")
    print()

    # Load back and verify
    print("Loading from file to verify...")
    loaded = GraphIR.from_json(output_file)
    print(f"✓ Loaded graph with {len(loaded.nodes)} nodes")
    print()

    # Validate loaded graph
    print("Validating loaded graph...")
    errors = loaded.validate()
    if errors:
        print("✗ Validation FAILED")
    else:
        print("✓ Validation PASSED")
    print()

    print("=" * 70)
    print(f"Example complete! Check {output_file} for the JSON output")
    print("=" * 70)


if __name__ == "__main__":
    main()
