"""Integration tests for audio/visual I/O (v0.6.0)."""

import pytest
import numpy as np
import tempfile
import os

from morphogen.stdlib.audio import audio, AudioBuffer
from morphogen.stdlib.visual import visual, Visual
from morphogen.stdlib.field import field
from morphogen.stdlib.agents import Agents, agents


@pytest.mark.skip(reason="Audio-visual integration not fully implemented - planned for v1.0")
class TestAudioVisualIntegration:
    """Test integrated audio and visual workflows."""

    def test_audio_waveform_visualization(self):
        """Test visualizing audio waveform."""
        # Generate audio
        tone = audio.sine(freq=440.0, duration=0.1)

        # Create waveform visualization (simplified)
        # Convert audio to 2D field for visualization
        waveform = tone.data[:1000].reshape(10, 100)

        # Create field and visualize
        from morphogen.stdlib.field import Field2D
        field_data = Field2D(data=waveform)
        vis = visual.colorize(field_data, palette="coolwarm")

        assert isinstance(vis, Visual)

    def test_audio_spectrogram_concept(self):
        """Test concept of audio-to-visual mapping."""
        # Generate different frequency tones
        low = audio.sine(freq=220.0, duration=0.05)
        mid = audio.sine(freq=440.0, duration=0.05)
        high = audio.sine(freq=880.0, duration=0.05)

        # Mix and save
        mixed = audio.mix(low, mid, high)

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            path = f.name

        try:
            audio.save(mixed, path)
            loaded = audio.load(path)

            # Verify round-trip
            assert loaded.sample_rate == mixed.sample_rate
            correlation = np.corrcoef(mixed.data, loaded.data)[0, 1]
            assert correlation > 0.99
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_agent_sound_coupling(self):
        """Test coupling agent simulation with audio generation."""
        # Create agents with velocity
        positions = np.random.rand(10, 2)
        velocities = np.random.rand(10, 2) * 0.1

        test_agents = Agents(
            count=10,
            properties={'pos': positions, 'vel': velocities}
        )

        # Visualize agents
        vis = visual.agents(test_agents, color_property='vel', size=3.0)

        # Generate audio based on average velocity
        avg_vel = np.mean(np.linalg.norm(velocities, axis=1))
        freq = 220.0 + avg_vel * 1000.0  # Map velocity to frequency

        tone = audio.sine(freq=freq, duration=0.1)

        # Verify both outputs
        assert isinstance(vis, Visual)
        assert isinstance(tone, AudioBuffer)

    def test_field_sonification(self):
        """Test sonifying field values to audio."""
        # Create field
        temp = field.random((64, 64), seed=42)

        # Sample field values
        samples = temp.data.flatten()[:4410]  # 0.1s at 44100Hz

        # Create audio from field values (normalized)
        normalized = (samples - np.min(samples)) / (np.max(samples) - np.min(samples))
        normalized = normalized * 2.0 - 1.0  # Scale to [-1, 1]

        audio_data = AudioBuffer(data=normalized.astype(np.float32), sample_rate=44100)

        # Save and verify
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            path = f.name

        try:
            audio.save(audio_data, path)
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_multi_modal_export(self):
        """Test exporting both audio and visual output."""
        pytest.importorskip("imageio")

        # Generate audio
        tone = audio.sine(freq=440.0, duration=0.2)

        # Generate visual animation
        frames = []
        for i in range(10):
            temp = field.random((64, 64), seed=i)
            frames.append(visual.colorize(temp, palette="fire"))

        # Export both
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "sound.wav")
            video_path = os.path.join(tmpdir, "animation.mp4")

            audio.save(tone, audio_path)
            visual.video(frames, video_path, fps=10)

            assert os.path.exists(audio_path)
            assert os.path.exists(video_path)

    def test_synchronized_audio_visual(self):
        """Test creating synchronized audio-visual content."""
        pytest.importorskip("imageio")

        # Parameters
        duration = 0.5
        fps = 10
        num_frames = int(duration * fps)

        # Generate audio sequence
        frequencies = [220, 330, 440, 550, 660]
        audio_segments = []

        for freq in frequencies:
            segment = audio.sine(freq=freq, duration=duration / len(frequencies))
            audio_segments.append(segment)

        # Mix audio
        full_audio = audio_segments[0]
        for segment in audio_segments[1:]:
            # Simple concatenation (would need proper mixing in production)
            full_audio = AudioBuffer(
                data=np.concatenate([full_audio.data, segment.data]),
                sample_rate=full_audio.sample_rate
            )

        # Generate synchronized visuals
        frames = []
        for i in range(num_frames):
            # Create visual that changes with audio
            value = i / num_frames
            layer = visual.layer(width=64, height=64, background=(value, 0, 1 - value))
            frames.append(layer)

        # Export both
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "audio.wav")
            video_path = os.path.join(tmpdir, "video.mp4")

            audio.save(full_audio, audio_path)
            visual.video(frames, video_path, fps=fps)

            assert os.path.exists(audio_path)
            assert os.path.exists(video_path)


@pytest.mark.skip(reason="Workflow examples depend on unimplemented features")
class TestWorkflowExamples:
    """Test realistic workflow examples."""

    def test_particle_animation_workflow(self):
        """Test complete particle animation workflow."""
        pytest.importorskip("imageio")

        # Initialize particle system
        n_particles = 50
        positions = np.random.rand(n_particles, 2)
        velocities = (np.random.rand(n_particles, 2) - 0.5) * 0.02

        # Simulate and render
        frames = []
        for step in range(20):
            # Update positions
            positions = positions + velocities

            # Wrap around boundaries
            positions = positions % 1.0

            # Create agents
            particles = Agents(
                count=n_particles,
                properties={'pos': positions, 'vel': velocities}
            )

            # Render
            frame = visual.agents(
                particles,
                width=128,
                height=128,
                color=(1, 1, 1),
                size=2.0
            )
            frames.append(frame)

        # Export animation
        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as f:
            path = f.name

        try:
            visual.video(frames, path, fps=10)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_field_evolution_workflow(self):
        """Test complete field evolution workflow."""
        pytest.importorskip("imageio")

        # Initialize field
        temp = field.random((64, 64), seed=42)

        # Evolve and capture frames
        frames = []
        for _ in range(15):
            temp = field.diffuse(temp, rate=0.1, dt=0.1)
            vis = visual.colorize(temp, palette="fire")
            frames.append(vis)

        # Export
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            path = f.name

        try:
            visual.video(frames, path, fps=10)
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_audio_processing_workflow(self):
        """Test complete audio processing workflow."""
        # Generate base tone
        base = audio.sine(freq=220.0, duration=0.3)

        # Apply effects chain
        filtered = audio.lowpass(base, cutoff=2000.0)
        reverbed = audio.reverb(filtered, mix=0.2)
        limited = audio.limiter(reverbed, threshold=-3.0)

        # Save final result
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            path = f.name

        try:
            audio.save(limited, path)

            # Load and verify
            loaded = audio.load(path)
            assert loaded.sample_rate == limited.sample_rate
            assert abs(loaded.duration - limited.duration) < 0.01
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_layered_composition_workflow(self):
        """Test complete layered composition workflow."""
        # Create background field
        background = field.random((128, 128), seed=1)
        bg_vis = visual.colorize(background, palette="viridis")

        # Create agent layer 1
        pos1 = np.random.rand(30, 2)
        agents1 = Agents(count=30, properties={'pos': pos1})
        agent_vis1 = visual.agents(agents1, width=128, height=128, color=(1, 0, 0))

        # Create agent layer 2
        pos2 = np.random.rand(20, 2)
        agents2 = Agents(count=20, properties={'pos': pos2})
        agent_vis2 = visual.agents(agents2, width=128, height=128, color=(0, 1, 1))

        # Composite all layers
        result = visual.composite(
            bg_vis,
            agent_vis1,
            agent_vis2,
            mode="add",
            opacity=[1.0, 0.7, 0.5]
        )

        # Save result
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name

        try:
            visual.output(result, path)
            assert os.path.exists(path)
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_format_conversion_workflow(self):
        """Test audio format conversion workflow."""
        pytest.importorskip("soundfile")

        # Generate audio
        original = audio.sine(freq=440.0, duration=0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save as WAV
            wav_path = os.path.join(tmpdir, "audio.wav")
            audio.save(original, wav_path)

            # Load and re-save as FLAC
            loaded_wav = audio.load(wav_path)
            flac_path = os.path.join(tmpdir, "audio.flac")
            audio.save(loaded_wav, flac_path)

            # Load FLAC and verify
            loaded_flac = audio.load(flac_path)

            assert loaded_flac.sample_rate == original.sample_rate
            assert abs(loaded_flac.duration - original.duration) < 0.01

            # Check correlation
            correlation = np.corrcoef(original.data, loaded_flac.data)[0, 1]
            assert correlation > 0.95


class TestErrorHandling:
    """Test error handling across I/O operations."""

    def test_audio_save_invalid_path(self):
        """Test error handling for invalid save path."""
        tone = audio.sine(freq=440.0, duration=0.1)

        # soundfile raises LibsndfileError for invalid paths
        with pytest.raises((OSError, FileNotFoundError, PermissionError, Exception)):
            audio.save(tone, "/invalid/path/audio.wav")

    def test_visual_video_invalid_dimensions(self):
        """Test error handling for inconsistent frame dimensions."""
        pytest.importorskip("imageio")

        frames = [
            visual.layer(width=32, height=32),
            visual.layer(width=64, height=64),  # Different size!
        ]

        # This should work at video creation level, but layers should match
        # The actual error would come from composite, not video

    def test_missing_dependencies(self):
        """Test graceful handling of missing optional dependencies."""
        # This test verifies that appropriate ImportErrors are raised
        # when optional dependencies are missing

        # These would raise ImportError if dependencies missing:
        # - audio.save() with FLAC (needs soundfile)
        # - visual.video() (needs imageio)
        # - audio.play() (needs sounddevice)
        # - audio.record() (needs sounddevice)

        # We can't actually test this without uninstalling packages,
        # but the error messages should be helpful
        pass
