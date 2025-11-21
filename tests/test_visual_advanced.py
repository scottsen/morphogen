"""Unit tests for advanced visualization operations (v0.11.0)."""

import pytest
import numpy as np
import tempfile
import os
from morphogen.stdlib.audio import AudioBuffer
from morphogen.stdlib.visual import visual, Visual
from morphogen.stdlib.agents import agents
from morphogen.stdlib.graph import graph


class TestSpectrogramVisualization:
    """Tests for visual.spectrogram operation."""

    def test_spectrogram_basic(self):
        """Test basic spectrogram creation."""
        # Generate simple sine wave
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(duration * sample_rate))
        signal = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        audio_buffer = AudioBuffer(signal, sample_rate=sample_rate)
        spec_vis = visual.spectrogram(audio_buffer)

        assert isinstance(spec_vis, Visual)
        assert spec_vis.width > 0
        assert spec_vis.height > 0

    def test_spectrogram_with_ndarray(self):
        """Test spectrogram with numpy array input."""
        signal = np.random.randn(44100).astype(np.float32)
        spec_vis = visual.spectrogram(signal, sample_rate=44100)

        assert isinstance(spec_vis, Visual)

    def test_spectrogram_palettes(self):
        """Test different color palettes."""
        signal = np.random.randn(22050).astype(np.float32)

        for palette in ["grayscale", "fire", "viridis", "coolwarm"]:
            spec_vis = visual.spectrogram(
                signal,
                sample_rate=22050,
                palette=palette
            )
            assert isinstance(spec_vis, Visual)

    def test_spectrogram_window_sizes(self):
        """Test different window sizes."""
        signal = np.random.randn(44100).astype(np.float32)

        for window_size in [512, 1024, 2048, 4096]:
            spec_vis = visual.spectrogram(
                signal,
                sample_rate=44100,
                window_size=window_size,
                hop_size=window_size // 4
            )
            assert isinstance(spec_vis, Visual)

    def test_spectrogram_log_scale(self):
        """Test logarithmic vs linear scale."""
        signal = np.random.randn(22050).astype(np.float32)

        # Log scale
        spec_log = visual.spectrogram(signal, log_scale=True)
        assert isinstance(spec_log, Visual)

        # Linear scale
        spec_linear = visual.spectrogram(signal, log_scale=False)
        assert isinstance(spec_linear, Visual)

    def test_spectrogram_freq_range(self):
        """Test frequency range filtering."""
        signal = np.random.randn(44100).astype(np.float32)

        spec_vis = visual.spectrogram(
            signal,
            sample_rate=44100,
            freq_range=(200, 2000)
        )
        assert isinstance(spec_vis, Visual)

    def test_spectrogram_stereo_audio(self):
        """Test spectrogram with stereo audio (should flatten)."""
        stereo_signal = np.random.randn(2, 22050).astype(np.float32)
        audio_buffer = AudioBuffer(stereo_signal, sample_rate=22050)

        spec_vis = visual.spectrogram(audio_buffer)
        assert isinstance(spec_vis, Visual)

    def test_spectrogram_invalid_input(self):
        """Test that invalid input raises error."""
        with pytest.raises(TypeError):
            visual.spectrogram("not an array")


class TestGraphVisualization:
    """Tests for visual.graph operation."""

    def test_graph_basic(self):
        """Test basic graph visualization."""
        g = graph.create(10)
        for i in range(9):
            g = graph.add_edge(g, i, i + 1, 1.0)

        vis = visual.graph(g)
        assert isinstance(vis, Visual)
        assert vis.width == 800
        assert vis.height == 800

    def test_graph_empty(self):
        """Test visualization of empty graph."""
        g = graph.create(0)
        vis = visual.graph(g)
        assert isinstance(vis, Visual)

    def test_graph_single_node(self):
        """Test visualization of graph with single node."""
        g = graph.create(1)
        vis = visual.graph(g)
        assert isinstance(vis, Visual)

    def test_graph_layouts(self):
        """Test different layout algorithms."""
        g = graph.create(8)
        # Create ring
        for i in range(8):
            g = graph.add_edge(g, i, (i + 1) % 8, 1.0)

        for layout in ["force", "circular", "grid"]:
            vis = visual.graph(g, layout=layout)
            assert isinstance(vis, Visual)

    def test_graph_force_directed_iterations(self):
        """Test force-directed layout with different iteration counts."""
        g = graph.create(10)
        for i in range(9):
            g = graph.add_edge(g, i, i + 1, 1.0)

        for iterations in [10, 50, 100]:
            vis = visual.graph(g, layout="force", iterations=iterations)
            assert isinstance(vis, Visual)

    def test_graph_centrality_coloring(self):
        """Test centrality-based node coloring."""
        # Create star network (clear centrality)
        g = graph.create(10)
        for i in range(1, 10):
            g = graph.add_edge(g, 0, i, 1.0)

        vis = visual.graph(g, color_by_centrality=True, palette="fire")
        assert isinstance(vis, Visual)

    def test_graph_custom_colors(self):
        """Test custom node and edge colors."""
        g = graph.create(5)
        g = graph.add_edge(g, 0, 1, 1.0)
        g = graph.add_edge(g, 1, 2, 1.0)

        vis = visual.graph(
            g,
            node_color=(1.0, 0.0, 0.0),
            edge_color=(0.0, 1.0, 0.0),
            node_size=12.0,
            edge_width=2.0
        )
        assert isinstance(vis, Visual)

    def test_graph_custom_dimensions(self):
        """Test custom image dimensions."""
        g = graph.create(5)
        g = graph.add_edge(g, 0, 1, 1.0)

        vis = visual.graph(g, width=400, height=600)
        assert vis.width == 400
        assert vis.height == 600

    def test_graph_background_color(self):
        """Test custom background color."""
        g = graph.create(5)
        g = graph.add_edge(g, 0, 1, 1.0)

        vis = visual.graph(g, background=(0.1, 0.2, 0.3))
        assert isinstance(vis, Visual)

    def test_graph_invalid_layout(self):
        """Test that invalid layout raises error."""
        g = graph.create(5)
        with pytest.raises(ValueError, match="Unknown layout"):
            visual.graph(g, layout="invalid")

    def test_graph_invalid_input(self):
        """Test that invalid input raises error."""
        with pytest.raises(TypeError):
            visual.graph("not a graph")


class TestPhaseSpaceVisualization:
    """Tests for visual.phase_space operation."""

    def test_phase_space_basic(self):
        """Test basic phase space visualization."""
        n_agents = 100
        positions = np.random.randn(n_agents, 2)
        velocities = np.random.randn(n_agents, 2)

        particles = agents.create(n_agents, pos=positions)
        particles = agents.set(particles, 'vel', velocities)

        vis = visual.phase_space(particles)
        assert isinstance(vis, Visual)

    def test_phase_space_1d(self):
        """Test phase space with 1D positions/velocities."""
        n_agents = 100
        positions = np.random.randn(n_agents, 1)
        velocities = np.random.randn(n_agents, 1)

        particles = agents.create(n_agents, pos=positions)
        particles = agents.set(particles, 'vel', velocities)

        vis = visual.phase_space(particles)
        assert isinstance(vis, Visual)

    def test_phase_space_color_property(self):
        """Test phase space with property-based coloring."""
        n_agents = 100
        positions = np.random.randn(n_agents, 2)
        velocities = np.random.randn(n_agents, 2)
        energy = np.random.rand(n_agents)

        particles = agents.create(n_agents, pos=positions)
        particles = agents.set(particles, 'vel', velocities)
        particles = agents.set(particles, 'energy', energy)

        vis = visual.phase_space(
            particles,
            color_property='energy',
            palette='fire'
        )
        assert isinstance(vis, Visual)

    def test_phase_space_palettes(self):
        """Test different color palettes."""
        n_agents = 50
        positions = np.random.randn(n_agents, 2)
        velocities = np.random.randn(n_agents, 2)

        particles = agents.create(n_agents, pos=positions)
        particles = agents.set(particles, 'vel', velocities)

        for palette in ["grayscale", "fire", "viridis", "coolwarm"]:
            vis = visual.phase_space(particles, palette=palette)
            assert isinstance(vis, Visual)

    def test_phase_space_trajectories(self):
        """Test phase space with trajectory lines."""
        n_agents = 20
        positions = np.random.randn(n_agents, 2)
        velocities = np.random.randn(n_agents, 2)

        particles = agents.create(n_agents, pos=positions)
        particles = agents.set(particles, 'vel', velocities)

        vis = visual.phase_space(particles, show_trajectories=True)
        assert isinstance(vis, Visual)

    def test_phase_space_custom_dimensions(self):
        """Test custom image dimensions."""
        n_agents = 50
        positions = np.random.randn(n_agents, 2)
        velocities = np.random.randn(n_agents, 2)

        particles = agents.create(n_agents, pos=positions)
        particles = agents.set(particles, 'vel', velocities)

        vis = visual.phase_space(particles, width=400, height=600)
        assert vis.width == 400
        assert vis.height == 600

    def test_phase_space_point_styling(self):
        """Test custom point size and alpha."""
        n_agents = 50
        positions = np.random.randn(n_agents, 2)
        velocities = np.random.randn(n_agents, 2)

        particles = agents.create(n_agents, pos=positions)
        particles = agents.set(particles, 'vel', velocities)

        vis = visual.phase_space(
            particles,
            point_size=5.0,
            alpha=0.8
        )
        assert isinstance(vis, Visual)

    def test_phase_space_background_color(self):
        """Test custom background color."""
        n_agents = 50
        positions = np.random.randn(n_agents, 2)
        velocities = np.random.randn(n_agents, 2)

        particles = agents.create(n_agents, pos=positions)
        particles = agents.set(particles, 'vel', velocities)

        vis = visual.phase_space(particles, background=(0.1, 0.1, 0.2))
        assert isinstance(vis, Visual)

    def test_phase_space_custom_properties(self):
        """Test custom property names."""
        n_agents = 50
        x = np.random.randn(n_agents, 2)
        v = np.random.randn(n_agents, 2)

        particles = agents.create(n_agents, x=x)
        particles = agents.set(particles, 'v', v)

        vis = visual.phase_space(
            particles,
            position_property='x',
            velocity_property='v'
        )
        assert isinstance(vis, Visual)

    def test_phase_space_invalid_input(self):
        """Test that invalid input raises error."""
        with pytest.raises(TypeError):
            visual.phase_space("not agents")


class TestMetricsDashboard:
    """Tests for visual.add_metrics operation."""

    def test_add_metrics_basic(self):
        """Test basic metrics overlay."""
        from morphogen.stdlib.field import field

        f = field.random((64, 64), seed=0)
        vis = visual.colorize(f, palette="fire")

        metrics = {"Frame": 42, "FPS": 59.8}
        vis_with_metrics = visual.add_metrics(vis, metrics)

        assert isinstance(vis_with_metrics, Visual)
        assert vis_with_metrics.shape == vis.shape

    def test_add_metrics_all_positions(self):
        """Test all dashboard positions."""
        from morphogen.stdlib.field import field

        f = field.random((128, 128), seed=0)
        vis = visual.colorize(f, palette="fire")

        metrics = {"Test": 123}

        for position in ["top-left", "top-right", "bottom-left", "bottom-right"]:
            vis_with_metrics = visual.add_metrics(vis, metrics, position=position)
            assert isinstance(vis_with_metrics, Visual)

    def test_add_metrics_multiple_values(self):
        """Test metrics with multiple values."""
        from morphogen.stdlib.field import field

        f = field.random((64, 64), seed=0)
        vis = visual.colorize(f, palette="fire")

        metrics = {
            "Integer": 42,
            "Float": 3.14159,
            "String": "Running",
            "Mixed": "T=273.15K"
        }

        vis_with_metrics = visual.add_metrics(vis, metrics)
        assert isinstance(vis_with_metrics, Visual)

    def test_add_metrics_custom_styling(self):
        """Test custom font and color styling."""
        from morphogen.stdlib.field import field

        f = field.random((64, 64), seed=0)
        vis = visual.colorize(f, palette="fire")

        metrics = {"Test": 123}

        vis_with_metrics = visual.add_metrics(
            vis,
            metrics,
            font_size=18,
            text_color=(1.0, 1.0, 0.0),
            bg_color=(0.2, 0.2, 0.2),
            bg_alpha=0.9
        )
        assert isinstance(vis_with_metrics, Visual)

    def test_add_metrics_empty_dict(self):
        """Test with empty metrics dictionary."""
        from morphogen.stdlib.field import field

        f = field.random((64, 64), seed=0)
        vis = visual.colorize(f, palette="fire")

        metrics = {}
        vis_with_metrics = visual.add_metrics(vis, metrics)
        assert isinstance(vis_with_metrics, Visual)

    def test_add_metrics_preserves_original(self):
        """Test that add_metrics doesn't modify original visual."""
        from morphogen.stdlib.field import field

        f = field.random((64, 64), seed=0)
        vis = visual.colorize(f, palette="fire")
        original_data = vis.data.copy()

        metrics = {"Test": 123}
        vis_with_metrics = visual.add_metrics(vis, metrics)

        # Original should be unchanged
        assert np.array_equal(vis.data, original_data)
        # New one should be different (metrics added)
        assert not np.array_equal(vis_with_metrics.data, original_data)

    def test_add_metrics_invalid_position(self):
        """Test that invalid position raises error."""
        from morphogen.stdlib.field import field

        f = field.random((64, 64), seed=0)
        vis = visual.colorize(f, palette="fire")

        metrics = {"Test": 123}
        with pytest.raises(ValueError, match="Unknown position"):
            visual.add_metrics(vis, metrics, position="invalid")

    def test_add_metrics_invalid_visual(self):
        """Test that invalid visual input raises error."""
        metrics = {"Test": 123}
        with pytest.raises(TypeError):
            visual.add_metrics("not a visual", metrics)


class TestIntegration:
    """Integration tests for combined visualizations."""

    def test_spectrogram_with_metrics(self):
        """Test spectrogram with metrics overlay."""
        signal = np.random.randn(22050).astype(np.float32)
        spec_vis = visual.spectrogram(signal, sample_rate=22050)

        metrics = {
            "Duration": "0.5 s",
            "Sample Rate": "22.05 kHz",
            "Window": 2048
        }

        final_vis = visual.add_metrics(spec_vis, metrics, position="top-left")
        assert isinstance(final_vis, Visual)

    def test_graph_with_metrics(self):
        """Test graph visualization with metrics overlay."""
        g = graph.create(10)
        for i in range(9):
            g = graph.add_edge(g, i, i + 1, 1.0)

        graph_vis = visual.graph(g, color_by_centrality=True)

        n_edges = np.sum(g.adj > 0) // 2
        metrics = {
            "Nodes": g.n_nodes,
            "Edges": n_edges
        }

        final_vis = visual.add_metrics(graph_vis, metrics, position="top-left")
        assert isinstance(final_vis, Visual)

    def test_phase_space_with_metrics(self):
        """Test phase space with metrics overlay."""
        n_agents = 100
        positions = np.random.randn(n_agents, 2)
        velocities = np.random.randn(n_agents, 2)

        particles = agents.create(n_agents, pos=positions)
        particles = agents.set(particles, 'vel', velocities)

        phase_vis = visual.phase_space(particles, palette='viridis')

        metrics = {
            "Particles": n_agents,
            "System": "Test"
        }

        final_vis = visual.add_metrics(phase_vis, metrics, position="top-right")
        assert isinstance(final_vis, Visual)

    def test_output_advanced_visualizations(self):
        """Test saving advanced visualizations to file."""
        # Test spectrogram output
        signal = np.random.randn(11025).astype(np.float32)
        spec_vis = visual.spectrogram(signal, sample_rate=11025)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            spec_path = tmp.name

        try:
            visual.output(spec_vis, spec_path)
            assert os.path.exists(spec_path)
            assert os.path.getsize(spec_path) > 0
        finally:
            if os.path.exists(spec_path):
                os.unlink(spec_path)

        # Test graph output
        g = graph.create(5)
        g = graph.add_edge(g, 0, 1, 1.0)
        graph_vis = visual.graph(g)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            graph_path = tmp.name

        try:
            visual.output(graph_vis, graph_path)
            assert os.path.exists(graph_path)
            assert os.path.getsize(graph_path) > 0
        finally:
            if os.path.exists(graph_path):
                os.unlink(graph_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
