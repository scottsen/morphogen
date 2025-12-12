"""Smoke tests for audio_analysis domain operators.

These tests verify operators are registered and callable.
Comprehensive functional tests should be added based on actual use cases.
"""

import pytest
import numpy as np
from morphogen.core.domain_registry import DomainRegistry


class TestAudioAnalysisDomainRegistration:
    """Verify audio_analysis domain is properly registered."""

    def test_domain_registered(self):
        """Test audio_analysis domain is registered."""
        DomainRegistry.initialize()
        domain = DomainRegistry.get('audio_analysis')

        assert domain is not None

    def test_all_operators_registered(self):
        """Test all 9 operators are registered."""
        DomainRegistry.initialize()
        domain = DomainRegistry.get('audio_analysis')
        operators = domain.list_operators()

        expected_ops = [
            'track_fundamental',
            'track_partials',
            'spectral_envelope',
            'analyze_modes',
            'fit_exponential_decay',
            'measure_t60',
            'measure_inharmonicity',
            'deconvolve',
            'model_noise'
        ]

        for op in expected_ops:
            assert op in operators, f"Operator {op} not registered"

        assert len(operators) == 9


class TestOperatorSmokeTests:
    """Basic smoke tests to verify operators are callable."""

    @pytest.fixture
    def test_signal(self):
        """Create simple test signal."""
        sr = 48000
        duration = 0.1  # Short signal for speed
        t = np.linspace(0, duration, int(sr * duration))
        return np.sin(2 * np.pi * 440 * t), sr

    def test_track_fundamental_callable(self, test_signal):
        """Verify track_fundamental is callable."""
        from morphogen.stdlib import audio_analysis
        signal, sr = test_signal

        result = audio_analysis.track_fundamental(signal, sr)

        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_deconvolve_callable(self, test_signal):
        """Verify deconvolve is callable."""
        from morphogen.stdlib import audio_analysis
        signal, sr = test_signal

        result = audio_analysis.deconvolve(signal, sr)

        assert result is not None

    def test_model_noise_callable(self, test_signal):
        """Verify model_noise is callable."""
        from morphogen.stdlib import audio_analysis
        signal, sr = test_signal

        result = audio_analysis.model_noise(signal, sr)

        assert result is not None


# Note: Comprehensive functional tests should be added based on:
# - Actual use cases from examples/
# - Expected behavior documented in operator metadata
# - Cross-domain integration scenarios
