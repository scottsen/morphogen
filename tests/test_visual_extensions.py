"""Tests for visual dialect extensions (v0.6.0)."""

import pytest
import numpy as np
import tempfile
import os

from morphogen.stdlib.visual import Visual, VisualOperations, visual
from morphogen.stdlib.agents import Agents, agents
from morphogen.stdlib.field import field


class TestVisualAgents:
    """Test agent visualization."""

    def test_agents_basic(self):
        """Test basic agent rendering."""
        # Create simple agent system
        positions = np.array([
            [0.5, 0.5],  # Center
            [0.2, 0.3],
            [0.7, 0.8],
        ])

        test_agents = Agents(
            count=3,
            properties={'pos': positions}
        )

        # Render agents
        vis = visual.agents(test_agents, width=128, height=128)

        # Verify output
        assert isinstance(vis, Visual)
        assert vis.shape == (128, 128)

        # Check that some pixels are non-black (agents rendered)
        assert np.any(vis.data > 0)

    def test_agents_with_color_property(self):
        """Test agent rendering with color-by-property."""
        positions = np.random.rand(10, 2)
        velocities = np.random.rand(10, 2) * 0.1

        test_agents = Agents(
            count=10,
            properties={'pos': positions, 'vel': velocities}
        )

        # Render with velocity coloring
        vis = visual.agents(
            test_agents,
            color_property='vel',
            palette='viridis'
        )

        assert isinstance(vis, Visual)
        assert np.any(vis.data > 0)

    def test_agents_with_size_property(self):
        """Test agent rendering with size-by-property."""
        positions = np.random.rand(10, 2)
        masses = np.random.rand(10) * 2.0

        test_agents = Agents(
            count=10,
            properties={'pos': positions, 'mass': masses}
        )

        # Render with mass-based sizing
        vis = visual.agents(
            test_agents,
            size_property='mass',
            size=3.0
        )

        assert isinstance(vis, Visual)

    def test_agents_custom_bounds(self):
        """Test agent rendering with custom bounds."""
        positions = np.array([[0.0, 0.0], [1.0, 1.0]])

        test_agents = Agents(
            count=2,
            properties={'pos': positions}
        )

        # Render with custom bounds
        vis = visual.agents(
            test_agents,
            bounds=((-1.0, 2.0), (-1.0, 2.0))
        )

        assert isinstance(vis, Visual)

    def test_agents_custom_color(self):
        """Test agent rendering with custom color."""
        positions = np.random.rand(5, 2)

        test_agents = Agents(
            count=5,
            properties={'pos': positions}
        )

        # Render with red color
        vis = visual.agents(
            test_agents,
            color=(1.0, 0.0, 0.0),
            size=5.0
        )

        assert isinstance(vis, Visual)

        # Check that red channel has data
        red_channel = vis.data[:, :, 0]
        assert np.any(red_channel > 0)

    def test_agents_background_color(self):
        """Test agent rendering with custom background."""
        positions = np.array([[0.5, 0.5]])

        test_agents = Agents(
            count=1,
            properties={'pos': positions}
        )

        # Render with blue background
        vis = visual.agents(
            test_agents,
            background=(0.0, 0.0, 0.5)
        )

        # Check background is blue
        blue_channel = vis.data[:, :, 2]
        assert np.mean(blue_channel) > 0.4

    def test_agents_invalid_type(self):
        """Test error handling for invalid input type."""
        with pytest.raises(TypeError, match="Expected Agents"):
            visual.agents("not agents")

    def test_agents_invalid_position_shape(self):
        """Test error handling for invalid position shape."""
        # Create agents with 1D positions (invalid)
        test_agents = Agents(
            count=5,
            properties={'pos': np.random.rand(5)}  # Should be (5, 2)
        )

        with pytest.raises(ValueError, match="Position property must be"):
            visual.agents(test_agents)

    def test_agents_many_particles(self):
        """Test rendering many agents."""
        # Create 1000 agents
        positions = np.random.rand(1000, 2)

        test_agents = Agents(
            count=1000,
            properties={'pos': positions}
        )

        vis = visual.agents(test_agents, width=256, height=256, size=2.0)

        assert isinstance(vis, Visual)
        assert vis.shape == (256, 256)


class TestVisualLayers:
    """Test layer operations."""

    def test_layer_create_empty(self):
        """Test creating empty layer."""
        layer = visual.layer(width=128, height=128)

        assert isinstance(layer, Visual)
        assert layer.shape == (128, 128)

        # Should be black (default background)
        assert np.all(layer.data == 0.0)

    def test_layer_create_with_background(self):
        """Test creating layer with custom background."""
        layer = visual.layer(
            width=64,
            height=64,
            background=(0.5, 0.5, 0.5)
        )

        # Should be gray
        assert np.allclose(layer.data, 0.5)

    def test_layer_from_visual(self):
        """Test creating layer from existing visual."""
        # Create test visual
        temp = field.random((64, 64), seed=42)
        original = visual.colorize(temp)

        # Convert to layer (should copy)
        layer = visual.layer(original)

        assert isinstance(layer, Visual)
        assert layer.shape == original.shape
        assert np.allclose(layer.data, original.data)

        # Verify it's a copy
        layer.data[0, 0, 0] = 1.0
        assert not np.allclose(layer.data, original.data)

    def test_layer_invalid_type(self):
        """Test error handling for invalid input."""
        with pytest.raises(TypeError, match="Expected Visual"):
            visual.layer("not a visual")


class TestVisualComposite:
    """Test layer compositing."""

    def test_composite_two_layers(self):
        """Test compositing two layers."""
        # Create two layers
        layer1 = visual.layer(width=64, height=64, background=(1.0, 0.0, 0.0))
        layer2 = visual.layer(width=64, height=64, background=(0.0, 0.0, 1.0))

        # Composite with over mode and opacity list
        result = visual.composite(layer1, layer2, mode="over", opacity=[1.0, 0.5])

        assert isinstance(result, Visual)
        assert result.shape == (64, 64)

        # With over mode: result = layer1 * 1.0, then blend with layer2 at 0.5 opacity
        # result = (1,0,0) * (1-0.5) + (0,0,1) * 0.5 = (0.5, 0, 0.5)
        mean_color = np.mean(result.data, axis=(0, 1))
        assert mean_color[0] > 0.4  # Red (should be ~0.5)
        assert mean_color[2] > 0.4  # Blue (should be ~0.5)
        assert mean_color[1] < 0.1  # Green (should be ~0)

    def test_composite_mode_add(self):
        """Test additive compositing."""
        layer1 = visual.layer(width=32, height=32, background=(0.3, 0.0, 0.0))
        layer2 = visual.layer(width=32, height=32, background=(0.0, 0.3, 0.0))

        result = visual.composite(layer1, layer2, mode="add")

        # Should add colors
        mean_color = np.mean(result.data, axis=(0, 1))
        assert mean_color[0] > 0.25  # Red preserved
        assert mean_color[1] > 0.25  # Green added

    def test_composite_mode_multiply(self):
        """Test multiply compositing."""
        layer1 = visual.layer(width=32, height=32, background=(0.8, 0.8, 0.8))
        layer2 = visual.layer(width=32, height=32, background=(0.5, 0.5, 0.5))

        result = visual.composite(layer1, layer2, mode="multiply")

        # Should darken (multiply)
        mean_value = np.mean(result.data)
        assert mean_value < 0.7  # Darker than both inputs

    def test_composite_mode_screen(self):
        """Test screen compositing."""
        layer1 = visual.layer(width=32, height=32, background=(0.5, 0.5, 0.5))
        layer2 = visual.layer(width=32, height=32, background=(0.5, 0.5, 0.5))

        result = visual.composite(layer1, layer2, mode="screen")

        # Should brighten
        mean_value = np.mean(result.data)
        assert mean_value > 0.5

    def test_composite_mode_overlay(self):
        """Test overlay compositing."""
        layer1 = visual.layer(width=32, height=32, background=(0.3, 0.3, 0.3))
        layer2 = visual.layer(width=32, height=32, background=(0.7, 0.7, 0.7))

        result = visual.composite(layer1, layer2, mode="overlay")

        assert isinstance(result, Visual)

    def test_composite_multiple_layers(self):
        """Test compositing multiple layers."""
        layers = [
            visual.layer(width=32, height=32, background=(1.0, 0.0, 0.0)),
            visual.layer(width=32, height=32, background=(0.0, 1.0, 0.0)),
            visual.layer(width=32, height=32, background=(0.0, 0.0, 1.0)),
        ]

        result = visual.composite(*layers, mode="add", opacity=0.5)

        assert isinstance(result, Visual)

    def test_composite_opacity_list(self):
        """Test compositing with per-layer opacity."""
        layer1 = visual.layer(width=32, height=32, background=(1.0, 0.0, 0.0))
        layer2 = visual.layer(width=32, height=32, background=(0.0, 1.0, 0.0))

        result = visual.composite(
            layer1, layer2,
            mode="over",
            opacity=[1.0, 0.3]  # Second layer less opaque
        )

        assert isinstance(result, Visual)

    def test_composite_no_layers(self):
        """Test error handling for no layers."""
        with pytest.raises(ValueError, match="At least one layer"):
            visual.composite()

    def test_composite_invalid_mode(self):
        """Test error handling for invalid mode."""
        layer1 = visual.layer(width=32, height=32)
        layer2 = visual.layer(width=32, height=32)

        with pytest.raises(ValueError, match="Unknown blending mode"):
            visual.composite(layer1, layer2, mode="invalid")

    def test_composite_size_mismatch(self):
        """Test error handling for mismatched layer sizes."""
        layer1 = visual.layer(width=32, height=32)
        layer2 = visual.layer(width=64, height=64)

        with pytest.raises(ValueError, match="shape"):
            visual.composite(layer1, layer2)

    def test_composite_opacity_length_mismatch(self):
        """Test error handling for opacity list length mismatch."""
        layer1 = visual.layer(width=32, height=32)
        layer2 = visual.layer(width=32, height=32)

        with pytest.raises(ValueError, match="opacity list length"):
            visual.composite(layer1, layer2, opacity=[1.0])  # Should be 2


@pytest.mark.skip(reason="Video generation not fully implemented - planned for v1.0")
class TestVisualVideo:
    """Test video export."""

    def test_video_from_list_mp4(self):
        """Test video export from list of frames (MP4)."""
        pytest.importorskip("imageio")

        # Create test frames
        frames = []
        for i in range(10):
            temp = field.random((64, 64), seed=i)
            frames.append(visual.colorize(temp, palette="fire"))

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            path = f.name

        try:
            visual.video(frames, path, fps=10)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_video_from_list_gif(self):
        """Test video export from list of frames (GIF)."""
        pytest.importorskip("imageio")

        frames = []
        for i in range(5):
            temp = field.random((32, 32), seed=i)
            frames.append(visual.colorize(temp))

        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as f:
            path = f.name

        try:
            visual.video(frames, path, fps=5)
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_video_auto_format(self):
        """Test automatic format detection."""
        pytest.importorskip("imageio")

        frames = [visual.layer(width=32, height=32) for _ in range(3)]

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            path = f.name

        try:
            visual.video(frames, path, format="auto")
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_video_invalid_format(self):
        """Test error handling for invalid format."""
        pytest.importorskip("imageio")

        frames = [visual.layer(width=32, height=32)]

        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name

        try:
            with pytest.raises(ValueError, match="Unsupported format"):
                visual.video(frames, path, format="avi")
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_video_no_frames(self):
        """Test error handling for empty frame list."""
        pytest.importorskip("imageio")

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            path = f.name

        try:
            with pytest.raises(ValueError, match="No frames"):
                visual.video([], path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_video_invalid_frame_type(self):
        """Test error handling for invalid frame type."""
        pytest.importorskip("imageio")

        frames = [visual.layer(width=32, height=32), "not a visual"]

        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            path = f.name

        try:
            with pytest.raises(TypeError, match="not a Visual"):
                visual.video(frames, path)
        finally:
            if os.path.exists(path):
                os.unlink(path)


class TestIntegrationCompositions:
    """Test integration of fields, agents, and visual composition."""

    def test_field_and_agents_composite(self):
        """Test compositing field and agent visualizations."""
        # Create field
        temp = field.random((64, 64), seed=42)
        field_vis = visual.colorize(temp, palette="fire")

        # Create agents
        positions = np.random.rand(10, 2)
        test_agents = Agents(count=10, properties={'pos': positions})
        agent_vis = visual.agents(test_agents, width=64, height=64, color=(1, 1, 1))

        # Composite
        result = visual.composite(field_vis, agent_vis, mode="add")

        assert isinstance(result, Visual)
        assert result.shape == (64, 64)

    def test_multi_agent_layers(self):
        """Test multiple agent layers with different colors."""
        # Create two agent populations
        pos1 = np.random.rand(20, 2)
        pos2 = np.random.rand(20, 2)

        agents1 = Agents(count=20, properties={'pos': pos1})
        agents2 = Agents(count=20, properties={'pos': pos2})

        # Render separately
        vis1 = visual.agents(agents1, width=128, height=128, color=(1, 0, 0))
        vis2 = visual.agents(agents2, width=128, height=128, color=(0, 1, 0))

        # Composite
        result = visual.composite(vis1, vis2, mode="add")

        assert isinstance(result, Visual)

    def test_animated_composite(self):
        """Test creating animated composition."""
        pytest.importorskip("imageio")

        frames = []
        for i in range(5):
            # Evolving field
            temp = field.random((32, 32), seed=i)
            field_vis = visual.colorize(temp, palette="viridis")

            # Moving agents
            positions = np.random.rand(5, 2)
            test_agents = Agents(count=5, properties={'pos': positions})
            agent_vis = visual.agents(test_agents, width=32, height=32)

            # Composite
            composite = visual.composite(field_vis, agent_vis, mode="add")
            frames.append(composite)

        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as f:
            path = f.name

        try:
            visual.video(frames, path, fps=5)
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)
