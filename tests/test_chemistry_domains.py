"""
Tests for Phase 2 chemistry domain operator metadata.

Validates that all chemistry domains migrated in Phase 2 have:
- Correct @operator decorators applied
- Proper metadata (domain, category, signature)
- All expected operators present and discoverable
"""

import pytest
from morphogen.core.domain_registry import DomainRegistry
from morphogen.core.operator import get_operator_metadata, OpCategory, is_operator


class TestKineticsDomain:
    """Test kinetics domain operators and metadata."""

    def setup_method(self):
        """Initialize registry before each test."""
        DomainRegistry.clear()
        DomainRegistry.initialize()

    def test_kinetics_domain_registered(self):
        """Test that kinetics domain is registered."""
        assert DomainRegistry.has_domain('kinetics')
        domain = DomainRegistry.get('kinetics')
        assert domain.name == 'kinetics'
        assert 'kinetics' in domain.module_path

    def test_kinetics_operator_count(self):
        """Test that kinetics has expected number of operators."""
        domain = DomainRegistry.get('kinetics')
        # Phase 2 migration: 11 operators
        assert len(domain.operators) == 11

    def test_kinetics_expected_operators_present(self):
        """Test that all expected kinetics operators are present."""
        domain = DomainRegistry.get('kinetics')
        expected = [
            'arrhenius', 'modified_arrhenius', 'vant_hoff',
            'reaction_rates', 'integrate_ode',
            'batch_reactor', 'cstr', 'pfr', 'pfr_with_dispersion',
            'mass_transfer_limited', 'create_reaction'
        ]

        actual = domain.list_operators()
        for op_name in expected:
            assert op_name in actual, f"Expected operator '{op_name}' not found"

    def test_kinetics_operators_have_metadata(self):
        """Test that all kinetics operators have proper metadata."""
        domain = DomainRegistry.get('kinetics')

        for op_name in domain.list_operators():
            op = domain.get_operator(op_name)

            # Check operator is decorated
            assert is_operator(op), f"{op_name} missing @operator decorator"

            # Check metadata exists and is correct
            metadata = get_operator_metadata(op)
            assert metadata is not None, f"{op_name} has no metadata"
            assert metadata.domain == "kinetics", f"{op_name} has wrong domain"
            assert isinstance(metadata.category, OpCategory), f"{op_name} has invalid category"
            assert metadata.signature, f"{op_name} missing signature"

    def test_kinetics_integrate_category(self):
        """Test that reactor integration operators have INTEGRATE category."""
        domain = DomainRegistry.get('kinetics')

        # These operators perform time/space integration
        integrate_ops = ['batch_reactor', 'pfr', 'pfr_with_dispersion', 'integrate_ode']

        for op_name in integrate_ops:
            metadata = domain.get_operator_metadata(op_name)
            assert metadata.category == OpCategory.INTEGRATE, \
                f"{op_name} should be INTEGRATE category, got {metadata.category.name}"

    def test_kinetics_operators_callable(self):
        """Test that kinetics operators are callable."""
        domain = DomainRegistry.get('kinetics')

        # Just verify operators are callable functions
        for op_name in domain.list_operators():
            op = domain.get_operator(op_name)
            assert callable(op), f"{op_name} should be callable"


class TestElectrochemDomain:
    """Test electrochem domain operators and metadata."""

    def setup_method(self):
        """Initialize registry before each test."""
        DomainRegistry.clear()
        DomainRegistry.initialize()

    def test_electrochem_domain_registered(self):
        """Test that electrochem domain is registered."""
        assert DomainRegistry.has_domain('electrochem')
        domain = DomainRegistry.get('electrochem')
        assert domain.name == 'electrochem'

    def test_electrochem_operator_count(self):
        """Test that electrochem has expected number of operators."""
        domain = DomainRegistry.get('electrochem')
        # Phase 2 migration: 13 operators
        assert len(domain.operators) == 13

    def test_electrochem_expected_operators_present(self):
        """Test that all expected electrochem operators are present."""
        domain = DomainRegistry.get('electrochem')
        expected = [
            'butler_volmer', 'tafel_equation', 'nernst', 'limiting_current',
            'battery_discharge', 'battery_charge', 'battery_cycle_life',
            'fuel_cell_voltage', 'fuel_cell_efficiency',
            'water_electrolysis_voltage', 'faraday_efficiency',
            'corrosion_current', 'corrosion_rate'
        ]

        actual = domain.list_operators()
        for op_name in expected:
            assert op_name in actual, f"Expected operator '{op_name}' not found"

    def test_electrochem_operators_have_metadata(self):
        """Test that all electrochem operators have proper metadata."""
        domain = DomainRegistry.get('electrochem')

        for op_name in domain.list_operators():
            op = domain.get_operator(op_name)
            assert is_operator(op), f"{op_name} missing @operator decorator"

            metadata = get_operator_metadata(op)
            assert metadata is not None, f"{op_name} has no metadata"
            assert metadata.domain == "electrochem", f"{op_name} has wrong domain"
            assert isinstance(metadata.category, OpCategory)


class TestCatalysisDomain:
    """Test catalysis domain operators and metadata."""

    def setup_method(self):
        """Initialize registry before each test."""
        DomainRegistry.clear()
        DomainRegistry.initialize()

    def test_catalysis_domain_registered(self):
        """Test that catalysis domain is registered."""
        assert DomainRegistry.has_domain('catalysis')

    def test_catalysis_operator_count(self):
        """Test that catalysis has expected number of operators."""
        domain = DomainRegistry.get('catalysis')
        # Phase 2 migration: 11 operators
        assert len(domain.operators) == 11

    def test_catalysis_expected_operators_present(self):
        """Test that all expected catalysis operators are present."""
        domain = DomainRegistry.get('catalysis')
        expected = [
            'langmuir_hinshelwood', 'eley_rideal', 'surface_coverage_step',
            'langmuir_adsorption', 'competitive_adsorption',
            'bet_surface_area', 'pore_size_distribution',
            'turnover_frequency', 'catalyst_selectivity', 'catalyst_deactivation',
            'microkinetic_steady_state'
        ]

        actual = domain.list_operators()
        for op_name in expected:
            assert op_name in actual, f"Expected operator '{op_name}' not found"

    def test_catalysis_operators_have_metadata(self):
        """Test that all catalysis operators have proper metadata."""
        domain = DomainRegistry.get('catalysis')

        for op_name in domain.list_operators():
            op = domain.get_operator(op_name)
            assert is_operator(op), f"{op_name} missing @operator decorator"

            metadata = get_operator_metadata(op)
            assert metadata.domain == "catalysis"


class TestTransportDomain:
    """Test transport domain operators and metadata."""

    def setup_method(self):
        """Initialize registry before each test."""
        DomainRegistry.clear()
        DomainRegistry.initialize()

    def test_transport_domain_registered(self):
        """Test that transport domain is registered."""
        assert DomainRegistry.has_domain('transport')

    def test_transport_operator_count(self):
        """Test that transport has expected number of operators."""
        domain = DomainRegistry.get('transport')
        # Phase 2 migration: 17 operators (class methods skipped)
        assert len(domain.operators) == 17

    def test_transport_expected_operators_present(self):
        """Test that all expected transport operators are present."""
        domain = DomainRegistry.get('transport')
        expected = [
            'conduction', 'convection', 'radiation',
            'nusselt_correlation', 'heat_transfer_coefficient',
            'fickian_diffusion', 'knudsen_diffusion', 'convective_mass_transfer',
            'sherwood_correlation', 'mass_transfer_coefficient',
            'effective_diffusivity', 'darcy_flow', 'carman_kozeny',
            'reynolds_number', 'prandtl_number', 'schmidt_number', 'peclet_number'
        ]

        actual = domain.list_operators()
        for op_name in expected:
            assert op_name in actual, f"Expected operator '{op_name}' not found"

    def test_transport_operators_have_metadata(self):
        """Test that all transport operators have proper metadata."""
        domain = DomainRegistry.get('transport')

        for op_name in domain.list_operators():
            op = domain.get_operator(op_name)
            assert is_operator(op), f"{op_name} missing @operator decorator"

            metadata = get_operator_metadata(op)
            assert metadata.domain == "transport"

    def test_transport_class_methods_not_decorated(self):
        """Test that class methods were correctly skipped during migration."""
        # This tests that we only decorated standalone functions, not class methods
        domain = DomainRegistry.get('transport')

        # These are the standalone functions (should be 17)
        assert len(domain.operators) == 17

        # Verify no duplicate names from class methods
        operator_names = domain.list_operators()
        assert len(operator_names) == len(set(operator_names)), "Duplicate operator names found"


class TestMultiphaseDomain:
    """Test multiphase domain operators and metadata."""

    def setup_method(self):
        """Initialize registry before each test."""
        DomainRegistry.clear()
        DomainRegistry.initialize()

    def test_multiphase_domain_registered(self):
        """Test that multiphase domain is registered."""
        assert DomainRegistry.has_domain('multiphase')

    def test_multiphase_operator_count(self):
        """Test that multiphase has expected number of operators."""
        domain = DomainRegistry.get('multiphase')
        # Phase 2 migration: 8 operators (nested function skipped)
        assert len(domain.operators) == 8

    def test_multiphase_expected_operators_present(self):
        """Test that all expected multiphase operators are present."""
        domain = DomainRegistry.get('multiphase')
        expected = [
            'antoine_equation', 'vle_flash', 'bubble_point', 'dew_point',
            'volumetric_mass_transfer', 'gas_liquid_reaction',
            'gas_absorption', 'two_phase_pressure_drop'
        ]

        actual = domain.list_operators()
        for op_name in expected:
            assert op_name in actual, f"Expected operator '{op_name}' not found"

    def test_multiphase_operators_have_metadata(self):
        """Test that all multiphase operators have proper metadata."""
        domain = DomainRegistry.get('multiphase')

        for op_name in domain.list_operators():
            op = domain.get_operator(op_name)
            assert is_operator(op), f"{op_name} missing @operator decorator"

            metadata = get_operator_metadata(op)
            assert metadata.domain == "multiphase"


class TestCombustionDomain:
    """Test combustion domain operators and metadata."""

    def setup_method(self):
        """Initialize registry before each test."""
        DomainRegistry.clear()
        DomainRegistry.initialize()

    def test_combustion_domain_registered(self):
        """Test that combustion domain is registered."""
        assert DomainRegistry.has_domain('combustion')

    def test_combustion_operator_count(self):
        """Test that combustion has expected number of operators."""
        domain = DomainRegistry.get('combustion')
        # Phase 2 migration: 7 operators
        assert len(domain.operators) == 7

    def test_combustion_expected_operators_present(self):
        """Test that all expected combustion operators are present."""
        domain = DomainRegistry.get('combustion')
        expected = [
            'equivalence_ratio', 'adiabatic_flame_temperature', 'zone_temperature',
            'smoke_reduction', 'combustion_efficiency', 'emissions_index',
            'analyze_fire_pit_combustion'
        ]

        actual = domain.list_operators()
        for op_name in expected:
            assert op_name in actual, f"Expected operator '{op_name}' not found"

    def test_combustion_operators_have_metadata(self):
        """Test that all combustion operators have proper metadata."""
        domain = DomainRegistry.get('combustion')

        for op_name in domain.list_operators():
            op = domain.get_operator(op_name)
            assert is_operator(op), f"{op_name} missing @operator decorator"

            metadata = get_operator_metadata(op)
            assert metadata.domain == "combustion"

    def test_combustion_module_path_correct(self):
        """Test that combustion uses combustion_light module path."""
        domain = DomainRegistry.get('combustion')
        # Domain name is "combustion" but file is combustion_light.py
        assert 'combustion_light' in domain.module_path


class TestPhase2Summary:
    """Summary tests for all Phase 2 chemistry domains."""

    def setup_method(self):
        """Initialize registry before each test."""
        DomainRegistry.clear()
        DomainRegistry.initialize()

    def test_all_phase2_domains_registered(self):
        """Test that all Phase 2 chemistry domains are registered."""
        phase2_domains = [
            'kinetics', 'electrochem', 'catalysis',
            'transport', 'multiphase', 'combustion'
        ]

        registered = DomainRegistry.list_domains()
        for domain_name in phase2_domains:
            assert domain_name in registered, f"Phase 2 domain '{domain_name}' not registered"

    def test_phase2_total_operator_count(self):
        """Test total operator count across all Phase 2 domains."""
        phase2_counts = {
            'kinetics': 11,
            'electrochem': 13,
            'catalysis': 11,
            'transport': 17,
            'multiphase': 8,
            'combustion': 7
        }

        total = 0
        for domain_name, expected_count in phase2_counts.items():
            domain = DomainRegistry.get(domain_name)
            actual_count = len(domain.operators)
            assert actual_count == expected_count, \
                f"{domain_name}: expected {expected_count} operators, got {actual_count}"
            total += actual_count

        # Phase 2 migration total: 67 operators
        assert total == 67, f"Expected 67 total Phase 2 operators, got {total}"

    def test_all_phase2_operators_have_metadata(self):
        """Test that every Phase 2 operator has valid metadata."""
        phase2_domains = [
            'kinetics', 'electrochem', 'catalysis',
            'transport', 'multiphase', 'combustion'
        ]

        total_operators = 0
        for domain_name in phase2_domains:
            domain = DomainRegistry.get(domain_name)

            for op_name in domain.list_operators():
                op = domain.get_operator(op_name)

                # Verify operator is decorated
                assert is_operator(op), \
                    f"{domain_name}.{op_name} missing @operator decorator"

                # Verify metadata is complete
                metadata = get_operator_metadata(op)
                assert metadata is not None
                assert metadata.domain == domain_name
                assert isinstance(metadata.category, OpCategory)
                assert metadata.signature is not None
                assert isinstance(metadata.deterministic, bool)

                total_operators += 1

        assert total_operators == 67, "Metadata check should cover all 67 Phase 2 operators"

    def test_phase2_operator_categories_valid(self):
        """Test that all Phase 2 operators use valid OpCategory values."""
        phase2_domains = [
            'kinetics', 'electrochem', 'catalysis',
            'transport', 'multiphase', 'combustion'
        ]

        valid_categories = set(OpCategory.__members__.values())

        for domain_name in phase2_domains:
            domain = DomainRegistry.get(domain_name)

            for op_name in domain.list_operators():
                metadata = domain.get_operator_metadata(op_name)
                assert metadata.category in valid_categories, \
                    f"{domain_name}.{op_name} has invalid category: {metadata.category}"
