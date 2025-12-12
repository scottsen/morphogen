"""Smoke tests for instrument_model domain operators.

These tests verify operators are registered and callable.
Comprehensive functional tests should be added based on actual use cases.
"""

import pytest
import numpy as np
from morphogen.core.domain_registry import DomainRegistry


class TestInstrumentModelDomainRegistration:
    """Verify instrument_model domain is properly registered."""

    def test_domain_registered(self):
        """Test instrument_model domain is registered."""
        DomainRegistry.initialize()
        domain = DomainRegistry.get('instrument_model')

        assert domain is not None

    def test_all_operators_registered(self):
        """Test all 12 operators are registered (5 core + 7 re-exported)."""
        DomainRegistry.initialize()
        domain = DomainRegistry.get('instrument_model')
        operators = domain.list_operators()

        # Core instrument_model operators
        core_ops = [
            'analyze_instrument',
            'synthesize_note',
            'morph_instruments',
            'save_instrument',
            'load_instrument'
        ]

        # Re-exported from audio_analysis
        reexported_ops = [
            'track_fundamental',
            'track_partials',
            'analyze_modes',
            'fit_exponential_decay',
            'measure_inharmonicity',
            'deconvolve',
            'model_noise'
        ]

        all_ops = core_ops + reexported_ops

        for op in all_ops:
            assert op in operators, f"Operator {op} not registered"

        assert len(operators) == 12


class TestOperatorSmokeTests:
    """Basic smoke tests to verify core operators are callable."""

    @pytest.fixture
    def test_signal(self):
        """Create simple test audio signal."""
        sr = 48000
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        # Simple tone with decay
        envelope = np.exp(-t / 0.2)
        signal = envelope * np.sin(2 * np.pi * 440 * t)
        return signal, sr

    def test_analyze_instrument_callable(self, test_signal):
        """Verify analyze_instrument is callable."""
        from morphogen.stdlib import instrument_model
        signal, sr = test_signal

        result = instrument_model.analyze_instrument(signal, sr)

        assert result is not None


# Note: Comprehensive functional tests should be added based on:
# - Actual use cases from examples/
# - Physical modeling synthesis workflows
# - Instrument morphing scenarios
# - Cross-domain audio integration
