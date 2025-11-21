"""Integration tests for audio operations with runtime and compositions."""

import pytest
import numpy as np
from morphogen.stdlib.audio import audio, AudioBuffer
from morphogen.runtime.runtime import Runtime
from morphogen.parser.parser import Parser
from morphogen.lexer.lexer import Lexer


class TestRuntimeIntegration:
    """Tests for audio operations in runtime."""

    def test_audio_namespace_available(self):
        """Test that audio namespace is available in runtime."""
        runtime = Runtime()
        assert runtime.context.get_variable("audio") is not None

    def test_sine_in_runtime(self):
        """Test calling audio.sine from runtime."""
        runtime = Runtime()
        audio_ns = runtime.context.get_variable("audio")

        sig = audio_ns.sine(freq=440.0, duration=0.1)
        assert isinstance(sig, AudioBuffer)
        assert sig.num_samples == 4410

    def test_noise_in_runtime(self):
        """Test calling audio.noise from runtime."""
        runtime = Runtime()
        audio_ns = runtime.context.get_variable("audio")

        noise = audio_ns.noise(noise_type="white", seed=42, duration=0.1)
        assert isinstance(noise, AudioBuffer)

    def test_filter_in_runtime(self):
        """Test calling audio.lowpass from runtime."""
        runtime = Runtime()
        audio_ns = runtime.context.get_variable("audio")

        sig = audio_ns.sine(freq=440.0, duration=0.1)
        filtered = audio_ns.lowpass(sig, cutoff=2000.0)

        assert isinstance(filtered, AudioBuffer)

    def test_composition_in_runtime(self):
        """Test composing operations in runtime."""
        runtime = Runtime()
        audio_ns = runtime.context.get_variable("audio")

        # Create a simple composition
        osc = audio_ns.sine(freq=440.0, duration=0.5)
        env = audio_ns.adsr(attack=0.01, decay=0.1, sustain=0.7, release=0.2, duration=0.5)

        # Apply envelope
        shaped = AudioBuffer(data=osc.data * env.data, sample_rate=44100)

        assert shaped.num_samples == osc.num_samples


class TestAudioCompositions:
    """Tests for complete audio compositions."""

    def test_simple_tone(self):
        """Test simple synthesized tone."""
        # Sine wave with envelope
        tone = audio.sine(freq=440.0, duration=1.0)
        env = audio.adsr(attack=0.01, decay=0.1, sustain=0.7, release=0.3, duration=1.0)

        # Apply envelope
        shaped = AudioBuffer(data=tone.data * env.data, sample_rate=44100)

        assert shaped.num_samples == tone.num_samples
        assert not np.any(np.isnan(shaped.data))

    def test_plucked_string(self):
        """Test plucked string synthesis."""
        # Excitation: filtered noise with envelope
        exc = audio.noise(noise_type="white", seed=1, duration=0.01)
        exc = audio.lowpass(exc, cutoff=6000.0)
        env = audio.envexp(time_constant=0.005, duration=0.01)
        exc_shaped = AudioBuffer(data=exc.data * env.data, sample_rate=44100)

        # String resonance
        pluck = audio.string(exc_shaped, freq=220.0, t60=1.5, damping=0.3)

        # Add reverb
        final = audio.reverb(pluck, mix=0.12, size=0.8)

        assert final.num_samples > 0
        assert not np.any(np.isnan(final.data))

    def test_bell_sound(self):
        """Test bell synthesis with modal synthesis."""
        # Impulse excitation
        exc = audio.impulse(rate=1.0, duration=0.001)

        # Modal synthesis with inharmonic partials
        bell = audio.modal(
            exc,
            frequencies=[200, 370, 550, 720, 890, 1050],
            decays=[3.0, 2.5, 2.0, 1.5, 1.2, 1.0],
            amplitudes=[1.0, 0.8, 0.6, 0.5, 0.3, 0.2]
        )

        # Large reverb for realistic bell
        final = audio.reverb(bell, mix=0.4, size=0.95)

        assert final.num_samples > 0

    def test_bass_synth(self):
        """Test bass synthesizer patch."""
        # Saw wave for bass
        bass = audio.saw(freq=55.0, duration=1.0, blep=True)

        # Filter with envelope
        filtered = audio.lowpass(bass, cutoff=800.0, q=2.0)

        # Envelope
        env = audio.adsr(attack=0.01, decay=0.2, sustain=0.4, release=0.3, duration=1.0)

        # Apply envelope
        shaped = AudioBuffer(data=filtered.data * env.data, sample_rate=44100)

        # Drive for warmth
        final = audio.drive(shaped, amount=0.3, shape="tanh")

        assert final.num_samples > 0

    def test_pad_sound(self):
        """Test pad/ambient sound."""
        # Multiple detuned saws
        saw1 = audio.saw(freq=220.0, duration=2.0, blep=True)
        saw2 = audio.saw(freq=220.5, duration=2.0, blep=True)  # Slightly detuned
        saw3 = audio.saw(freq=219.5, duration=2.0, blep=True)  # Slightly detuned

        # Mix
        mixed = audio.mix(saw1, saw2, saw3)

        # Filter
        filtered = audio.lowpass(mixed, cutoff=2000.0, q=0.707)

        # Slow envelope
        env = audio.adsr(attack=0.5, decay=0.5, sustain=0.8, release=1.0, duration=2.0)
        shaped = AudioBuffer(data=filtered.data * env.data, sample_rate=44100)

        # Chorus and reverb
        chorused = audio.chorus(shaped, rate=0.3, depth=0.008, mix=0.4)
        final = audio.reverb(chorused, mix=0.3, size=0.85)

        assert final.num_samples > 0

    def test_kick_drum(self):
        """Test kick drum synthesis."""
        # Pitch envelope (from high to low)
        duration = 0.5

        # Simple sine sweep for kick
        t = np.arange(int(duration * 44100)) / 44100
        freq_sweep = 150 * np.exp(-t * 8) + 50  # 150Hz -> 50Hz

        # Generate with frequency modulation (simplified)
        kick = audio.sine(freq=80.0, duration=duration)

        # Exponential envelope
        env = audio.envexp(time_constant=0.08, duration=duration)
        shaped = AudioBuffer(data=kick.data * env.data, sample_rate=44100)

        # Drive for punch
        final = audio.drive(shaped, amount=0.4, shape="tanh")

        assert final.num_samples > 0

    def test_snare_drum(self):
        """Test snare drum synthesis."""
        duration = 0.3

        # Noise component
        noise = audio.noise(noise_type="white", seed=42, duration=duration)
        noise_filtered = audio.highpass(noise, cutoff=200.0)

        # Tone component (modes)
        exc = audio.impulse(rate=1.0, duration=0.001)
        tone = audio.modal(
            exc,
            frequencies=[180, 330, 450],
            decays=[0.15, 0.12, 0.10]
        )

        # Mix tone and noise
        mixed = AudioBuffer(
            data=tone.data[:noise_filtered.num_samples] * 0.3 + noise_filtered.data * 0.7,
            sample_rate=44100
        )

        # Envelope
        env = audio.envexp(time_constant=0.05, duration=duration)
        final = AudioBuffer(data=mixed.data * env.data, sample_rate=44100)

        assert final.num_samples > 0


class TestAudioChains:
    """Tests for effect chains and signal processing."""

    def test_guitar_chain(self):
        """Test guitar processing chain."""
        # Plucked string
        exc = audio.noise(noise_type="white", seed=1, duration=0.01)
        exc = audio.lowpass(exc, cutoff=8000.0)
        guitar = audio.string(exc, freq=329.63, t60=2.0, damping=0.2)

        # EQ
        eqed = audio.eq3(guitar, bass=-3.0, mid=3.0, treble=0.0)

        # Drive
        driven = audio.drive(eqed, amount=0.5, shape="tanh")

        # Delay
        delayed = audio.delay(driven, time=0.375, feedback=0.35, mix=0.25)

        # Reverb
        final = audio.reverb(delayed, mix=0.15, size=0.7)

        assert final.num_samples > 0
        assert not np.any(np.isnan(final.data))

    def test_vocal_chain(self):
        """Test vocal processing chain."""
        # Sine as voice proxy
        vocal = audio.sine(freq=200.0, duration=1.0)
        env = audio.adsr(attack=0.1, decay=0.1, sustain=0.8, release=0.2, duration=1.0)
        vocal = AudioBuffer(data=vocal.data * env.data, sample_rate=44100)

        # EQ
        eqed = audio.eq3(vocal, bass=-6.0, mid=3.0, treble=6.0)

        # Compression (limiter)
        compressed = audio.limiter(eqed, threshold=-6.0, release=0.1)

        # Chorus
        chorused = audio.chorus(compressed, rate=0.5, depth=0.005, mix=0.2)

        # Reverb
        final = audio.reverb(chorused, mix=0.25, size=0.8)

        assert final.num_samples > 0

    def test_mastering_chain(self):
        """Test mastering chain."""
        # Source material (sine as proxy)
        source = audio.sine(freq=440.0, duration=1.0)
        source.data *= 0.7  # Not at full level

        # EQ
        eqed = audio.eq3(source, bass=1.0, mid=0.0, treble=2.0)

        # Limiter
        limited = audio.limiter(eqed, threshold=-0.5, release=0.05)

        # Normalize to target
        final = audio.normalize(limited, target=0.98)

        assert final.num_samples > 0
        assert np.max(np.abs(final.data)) <= 1.0


class TestDeterministicCompositions:
    """Tests for deterministic composition rendering."""

    def test_composition_deterministic(self):
        """Test that composition renders deterministically."""
        # Build same composition twice
        def build_composition():
            exc = audio.noise(noise_type="white", seed=42, duration=0.01)
            exc = audio.lowpass(exc, cutoff=6000.0)
            pluck = audio.string(exc, freq=220.0, t60=1.0, damping=0.3)
            return audio.reverb(pluck, mix=0.15, size=0.8)

        comp1 = build_composition()
        comp2 = build_composition()

        assert np.allclose(comp1.data, comp2.data)

    def test_complex_composition_deterministic(self):
        """Test complex composition is deterministic."""
        def build_complex():
            # Oscillators
            osc1 = audio.saw(freq=220.0, duration=1.0)
            osc2 = audio.square(freq=220.0, pwm=0.3, duration=1.0)

            # Mix
            mixed = audio.mix(osc1, osc2)

            # Filter
            filtered = audio.lowpass(mixed, cutoff=1500.0, q=2.0)

            # Envelope
            env = audio.adsr(attack=0.01, decay=0.2, sustain=0.6, release=0.3, duration=1.0)
            shaped = AudioBuffer(data=filtered.data * env.data, sample_rate=44100)

            # Effects
            chorused = audio.chorus(shaped, rate=0.3, depth=0.008, mix=0.3)
            final = audio.reverb(chorused, mix=0.2, size=0.8)

            return final

        comp1 = build_complex()
        comp2 = build_complex()

        assert np.allclose(comp1.data, comp2.data)


class TestAudioPerformance:
    """Tests for audio performance characteristics."""

    def test_long_buffer_generation(self):
        """Test generating longer audio buffers."""
        # 10 seconds
        sig = audio.sine(freq=440.0, duration=10.0)
        assert sig.num_samples == 441000

    def test_complex_chain_performance(self):
        """Test complex effect chain doesn't crash."""
        sig = audio.sine(freq=440.0, duration=1.0)

        # Chain many effects
        processed = sig
        processed = audio.lowpass(processed, cutoff=2000.0)
        processed = audio.drive(processed, amount=0.3, shape="tanh")
        processed = audio.chorus(processed, rate=0.3, depth=0.008, mix=0.3)
        processed = audio.delay(processed, time=0.3, feedback=0.3, mix=0.25)
        processed = audio.reverb(processed, mix=0.2, size=0.8)
        processed = audio.limiter(processed, threshold=-3.0, release=0.05)

        assert processed.num_samples > 0
        assert not np.any(np.isnan(processed.data))


class TestAudioUtilityFunctions:
    """Tests for utility functions in compositions."""

    def test_stereo_mixing(self):
        """Test mixing stereo signals."""
        # Create stereo signals via panning
        sig1 = audio.sine(freq=440.0, duration=0.5)
        sig2 = audio.sine(freq=880.0, duration=0.5)

        left = audio.pan(sig1, position=-0.5)
        right = audio.pan(sig2, position=0.5)

        # Mix stereo (simple addition)
        mixed = AudioBuffer(
            data=left.data + right.data,
            sample_rate=44100
        )

        assert mixed.is_stereo

    def test_gain_staging(self):
        """Test gain staging in mix."""
        sig1 = audio.sine(freq=440.0, duration=0.5)
        sig2 = audio.sine(freq=880.0, duration=0.5)
        sig3 = audio.sine(freq=1320.0, duration=0.5)

        # Apply different gains
        sig1_gain = audio.gain(sig1, amount_db=-6.0)
        sig2_gain = audio.gain(sig2, amount_db=0.0)
        sig3_gain = audio.gain(sig3, amount_db=-12.0)

        # Mix
        mixed = audio.mix(sig1_gain, sig2_gain, sig3_gain)

        assert mixed.num_samples > 0

    def test_normalization_in_chain(self):
        """Test normalization at end of chain."""
        sig = audio.sine(freq=440.0, duration=0.5)

        # Process with effects (might change level)
        processed = audio.drive(sig, amount=0.5, shape="tanh")
        processed = audio.reverb(processed, mix=0.3, size=0.8)

        # Normalize to target
        final = audio.normalize(processed, target=0.95)

        # Should be at target
        assert abs(np.max(np.abs(final.data)) - 0.95) < 0.01
