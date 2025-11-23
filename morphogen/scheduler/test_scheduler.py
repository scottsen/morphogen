"""
Comprehensive tests for SimplifiedScheduler

Test coverage:
1. GCD master clock computation
2. Rate group partitioning
3. Topological sort integration
4. Linear resampling
5. Execution loop
6. Multirate graphs
7. Cross-rate connections
8. Edge cases and error handling
"""

import pytest
import numpy as np
from math import gcd

from ..graph_ir import GraphIR, GraphIROutputPort
from .simplified import SimplifiedScheduler, RateGroup


class TestMasterClock:
    """Test GCD master clock computation"""

    def test_single_rate_audio(self):
        """Test with single audio rate"""
        graph = GraphIR(sample_rate=48000)
        graph.add_node(
            id="osc1",
            op="sine",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
        )

        scheduler = SimplifiedScheduler(graph)
        assert scheduler.master_rate == 48000.0

    def test_audio_and_control(self):
        """Test GCD with audio (48kHz) and control (1kHz)"""
        graph = GraphIR(sample_rate=48000)

        graph.add_node(
            id="osc1",
            op="sine",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
        )

        graph.add_node(
            id="env1",
            op="adsr",
            outputs=[GraphIROutputPort(name="out", type="Ctl")],
            rate="control",
        )

        scheduler = SimplifiedScheduler(graph)

        # GCD(48000, 1000) = 1000
        assert scheduler.master_rate == 1000.0

    def test_custom_rates(self):
        """Test GCD with custom rate overrides"""
        graph = GraphIR()
        graph.add_node(
            id="osc1",
            op="sine",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
        )
        graph.add_node(
            id="ctrl1",
            op="lfo",
            outputs=[GraphIROutputPort(name="out", type="Ctl")],
            rate="control",
        )

        scheduler = SimplifiedScheduler(
            graph,
            rate_overrides={"audio": 44100, "control": 900}
        )

        # GCD(44100, 900) = 300
        expected_gcd = float(gcd(44100, 900))
        assert scheduler.master_rate == expected_gcd


class TestRateGroups:
    """Test rate group partitioning"""

    def test_single_rate_group(self):
        """Test with all nodes at same rate"""
        graph = GraphIR()
        graph.add_node(
            id="osc1",
            op="sine",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
        )
        graph.add_node(
            id="lpf1",
            op="lpf",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
        )

        scheduler = SimplifiedScheduler(graph)

        assert len(scheduler.rate_groups) == 1
        assert scheduler.rate_groups[0].rate == "audio"
        assert len(scheduler.rate_groups[0].operators) == 2

    def test_multiple_rate_groups(self):
        """Test with audio and control rates"""
        graph = GraphIR()

        graph.add_node(
            id="osc1",
            op="sine",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
        )

        graph.add_node(
            id="env1",
            op="adsr",
            outputs=[GraphIROutputPort(name="out", type="Ctl")],
            rate="control",
        )

        scheduler = SimplifiedScheduler(graph)

        assert len(scheduler.rate_groups) == 2

        # Rate groups should be sorted by rate (highest first)
        assert scheduler.rate_groups[0].rate == "audio"
        assert scheduler.rate_groups[1].rate == "control"

    def test_rate_multipliers(self):
        """Test that rate multipliers are correct"""
        graph = GraphIR(sample_rate=48000)

        graph.add_node(
            id="osc1",
            op="sine",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
        )

        graph.add_node(
            id="env1",
            op="adsr",
            outputs=[GraphIROutputPort(name="out", type="Ctl")],
            rate="control",
        )

        scheduler = SimplifiedScheduler(graph)

        # Master clock = 1000 Hz (GCD of 48000 and 1000)
        # Audio multiplier = 48000 / 1000 = 48
        # Control multiplier = 1000 / 1000 = 1

        audio_group = next(g for g in scheduler.rate_groups if g.rate == "audio")
        control_group = next(g for g in scheduler.rate_groups if g.rate == "control")

        assert audio_group.multiplier == 48
        assert control_group.multiplier == 1


class TestTopologicalSort:
    """Test topological sort integration"""

    def test_simple_chain(self):
        """Test simple A -> B -> C chain"""
        graph = GraphIR()

        graph.add_node(
            id="osc1",
            op="sine",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
        )

        graph.add_node(
            id="lpf1",
            op="lpf",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
        )

        graph.add_node(
            id="gain1",
            op="multiply",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
        )

        graph.add_edge(from_port="osc1:out", to_port="lpf1:in", type="Sig")
        graph.add_edge(from_port="lpf1:out", to_port="gain1:in1", type="Sig")

        scheduler = SimplifiedScheduler(graph)

        audio_group = scheduler.rate_groups[0]
        operators = audio_group.operators

        # osc1 must come before lpf1, lpf1 before gain1
        assert operators.index("osc1") < operators.index("lpf1")
        assert operators.index("lpf1") < operators.index("gain1")

    def test_diamond_dependency(self):
        """Test diamond: A -> B, A -> C, B -> D, C -> D"""
        graph = GraphIR()

        for node_id in ["A", "B", "C", "D"]:
            graph.add_node(
                id=node_id,
                op="sine",
                outputs=[GraphIROutputPort(name="out", type="Sig")],
                rate="audio",
            )

        graph.add_edge(from_port="A:out", to_port="B:in", type="Sig")
        graph.add_edge(from_port="A:out", to_port="C:in", type="Sig")
        graph.add_edge(from_port="B:out", to_port="D:in1", type="Sig")
        graph.add_edge(from_port="C:out", to_port="D:in2", type="Sig")

        scheduler = SimplifiedScheduler(graph)

        audio_group = scheduler.rate_groups[0]
        operators = audio_group.operators

        # A must come before B and C
        assert operators.index("A") < operators.index("B")
        assert operators.index("A") < operators.index("C")

        # B and C must come before D
        assert operators.index("B") < operators.index("D")
        assert operators.index("C") < operators.index("D")

    def test_cycle_detection(self):
        """Test that cycles are rejected"""
        graph = GraphIR()

        graph.add_node(
            id="node1",
            op="sine",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
        )

        graph.add_node(
            id="node2",
            op="lpf",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
        )

        # Create cycle
        graph.add_edge(from_port="node1:out", to_port="node2:in", type="Sig")
        graph.add_edge(from_port="node2:out", to_port="node1:in", type="Sig")

        with pytest.raises(ValueError, match="cycle"):
            SimplifiedScheduler(graph)


class TestLinearResampling:
    """Test linear resampling"""

    def test_upsample_2x(self):
        """Test upsampling by 2x"""
        scheduler = SimplifiedScheduler(GraphIR())

        input_buffer = np.array([0.0, 1.0, 0.0, -1.0])
        output = scheduler._linear_resample(input_buffer, 1000.0, 2000.0)

        # Should have 8 samples (2x)
        assert len(output) == 8

        # Check first and last values match
        assert output[0] == pytest.approx(0.0)
        assert output[-1] == pytest.approx(-1.0)

    def test_downsample_2x(self):
        """Test downsampling by 2x"""
        scheduler = SimplifiedScheduler(GraphIR())

        input_buffer = np.array([0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0, -0.5])
        output = scheduler._linear_resample(input_buffer, 2000.0, 1000.0)

        # Should have 4 samples (0.5x)
        assert len(output) == 4

    def test_same_rate(self):
        """Test that same rate returns copy"""
        scheduler = SimplifiedScheduler(GraphIR())

        input_buffer = np.array([1.0, 2.0, 3.0, 4.0])
        output = scheduler._linear_resample(input_buffer, 1000.0, 1000.0)

        assert len(output) == len(input_buffer)
        np.testing.assert_array_equal(output, input_buffer)

    def test_upsample_control_to_audio(self):
        """Test realistic control -> audio resampling"""
        scheduler = SimplifiedScheduler(GraphIR())

        # Control signal (1kHz): 10 samples
        input_buffer = np.linspace(0, 1, 10)

        # Resample to audio (48kHz)
        output = scheduler._linear_resample(input_buffer, 1000.0, 48000.0)

        # Should have ~480 samples
        assert len(output) == 480

        # Check interpolation is smooth
        assert output[0] == pytest.approx(0.0)
        assert output[-1] == pytest.approx(1.0)


class TestExecutionLoop:
    """Test execution loop"""

    def test_single_sine_oscillator(self):
        """Test executing single sine oscillator"""
        graph = GraphIR(sample_rate=48000)

        graph.add_node(
            id="osc1",
            op="sine",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
            params={"freq": "440Hz"},
        )

        graph.add_output("mono", ["osc1:out"])

        scheduler = SimplifiedScheduler(graph, hop_size=128)

        # Execute 1 second
        outputs = scheduler.execute(duration_samples=48000)

        assert "mono" in outputs
        assert len(outputs["mono"]) == 48000

        # Check it's actually a sine wave (non-zero, bounded)
        assert np.max(np.abs(outputs["mono"])) > 0.5
        assert np.max(np.abs(outputs["mono"])) <= 1.0

    def test_execution_duration_seconds(self):
        """Test execution with duration in seconds"""
        graph = GraphIR(sample_rate=48000)

        graph.add_node(
            id="osc1",
            op="sine",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
        )

        graph.add_output("mono", ["osc1:out"])

        scheduler = SimplifiedScheduler(graph)

        outputs = scheduler.execute(duration_seconds=0.5)

        # 0.5 seconds @ 48kHz = 24000 samples
        assert len(outputs["mono"]) == 24000

    def test_multiple_hops(self):
        """Test that hop-based execution produces correct output length"""
        graph = GraphIR(sample_rate=48000)

        graph.add_node(
            id="osc1",
            op="sine",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
        )

        graph.add_output("mono", ["osc1:out"])

        # Small hop size to force multiple hops
        scheduler = SimplifiedScheduler(graph, hop_size=100)

        outputs = scheduler.execute(duration_samples=1000)

        # Should be exactly 1000 samples despite hop size
        assert len(outputs["mono"]) == 1000


class TestMultirateExecution:
    """Test multirate graph execution"""

    def test_audio_plus_control(self):
        """Test graph with both audio and control rates"""
        graph = GraphIR(sample_rate=48000)

        # Audio oscillator
        graph.add_node(
            id="osc1",
            op="sine",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
            params={"freq": "440Hz"},
        )

        # Control envelope
        graph.add_node(
            id="env1",
            op="adsr",
            outputs=[GraphIROutputPort(name="out", type="Ctl")],
            rate="control",
        )

        # Multiply (audio × control)
        graph.add_node(
            id="mul1",
            op="multiply",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
        )

        # Connections
        graph.add_edge(from_port="osc1:out", to_port="mul1:in1", type="Sig")
        graph.add_edge(from_port="env1:out", to_port="mul1:in2", type="Ctl")

        graph.add_output("mono", ["mul1:out"])

        scheduler = SimplifiedScheduler(graph)

        outputs = scheduler.execute(duration_samples=48000)

        assert "mono" in outputs
        assert len(outputs["mono"]) == 48000

        # Output should be modulated (non-constant)
        assert np.std(outputs["mono"]) > 0.01

    def test_cross_rate_resampling(self):
        """Test that cross-rate connections are properly resampled"""
        graph = GraphIR(sample_rate=48000)

        # Control signal
        graph.add_node(
            id="ctrl1",
            op="adsr",
            outputs=[GraphIROutputPort(name="out", type="Ctl")],
            rate="control",
        )

        # Audio multiply
        graph.add_node(
            id="mul1",
            op="multiply",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
        )

        # Cross-rate connection
        graph.add_edge(from_port="ctrl1:out", to_port="mul1:in1", type="Ctl")

        graph.add_output("mono", ["mul1:out"])

        scheduler = SimplifiedScheduler(graph, hop_size=480)

        # This should not crash and should produce output
        outputs = scheduler.execute(duration_samples=4800)

        assert len(outputs["mono"]) == 4800


class TestSchedulerInfo:
    """Test scheduler information reporting"""

    def test_get_info(self):
        """Test that get_info returns correct information"""
        graph = GraphIR(sample_rate=48000)

        graph.add_node(
            id="osc1",
            op="sine",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
        )

        graph.add_node(
            id="env1",
            op="adsr",
            outputs=[GraphIROutputPort(name="out", type="Ctl")],
            rate="control",
        )

        scheduler = SimplifiedScheduler(graph, hop_size=256)

        info = scheduler.get_info()

        assert info["sample_rate"] == 48000
        assert info["hop_size"] == 256
        assert info["master_rate"] == 1000.0
        assert "audio" in info["active_rates"]
        assert "control" in info["active_rates"]
        assert len(info["rate_groups"]) == 2


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_graph(self):
        """Test with empty graph"""
        graph = GraphIR()

        scheduler = SimplifiedScheduler(graph)

        # Should not crash
        outputs = scheduler.execute(duration_samples=1000)

    def test_zero_duration(self):
        """Test with zero duration"""
        graph = GraphIR()
        graph.add_node(
            id="osc1",
            op="sine",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
        )

        scheduler = SimplifiedScheduler(graph)

        outputs = scheduler.execute(duration_samples=0)

        # Should produce empty or zero output
        for output in outputs.values():
            assert len(output) == 0

    def test_disconnected_nodes(self):
        """Test graph with disconnected nodes"""
        graph = GraphIR()

        # Two unconnected nodes
        graph.add_node(
            id="osc1",
            op="sine",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
        )

        graph.add_node(
            id="osc2",
            op="sine",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
        )

        graph.add_output("mono", ["osc1:out"])

        scheduler = SimplifiedScheduler(graph)

        # Should execute without error
        outputs = scheduler.execute(duration_samples=1000)

        assert len(outputs["mono"]) == 1000


class TestIntegration:
    """End-to-end integration tests"""

    def test_simple_synth(self):
        """Test complete simple synth: sine -> lpf -> multiply <- adsr"""
        graph = GraphIR(sample_rate=48000, seed=1337)

        # Sine oscillator
        graph.add_node(
            id="osc1",
            op="sine",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
            params={"freq": "440Hz"},
        )

        # Low-pass filter (mock)
        graph.add_node(
            id="lpf1",
            op="lpf",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
            params={"cutoff": "2kHz"},
        )

        # ADSR envelope
        graph.add_node(
            id="env1",
            op="adsr",
            outputs=[GraphIROutputPort(name="out", type="Ctl")],
            rate="control",
            params={"attack": "0.01s"},
        )

        # Multiply
        graph.add_node(
            id="mul1",
            op="multiply",
            outputs=[GraphIROutputPort(name="out", type="Sig")],
            rate="audio",
        )

        # Connections
        graph.add_edge(from_port="osc1:out", to_port="lpf1:in", type="Sig")
        graph.add_edge(from_port="lpf1:out", to_port="mul1:in1", type="Sig")
        graph.add_edge(from_port="env1:out", to_port="mul1:in2", type="Ctl")

        graph.add_output("mono", ["mul1:out"])

        # Validate graph
        errors = graph.validate()
        assert len(errors) == 0

        # Execute
        scheduler = SimplifiedScheduler(graph, hop_size=128)
        outputs = scheduler.execute(duration_seconds=1.0)

        assert "mono" in outputs
        assert len(outputs["mono"]) == 48000

        # Check output is reasonable
        assert np.max(np.abs(outputs["mono"])) > 0.1
        assert np.max(np.abs(outputs["mono"])) <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
