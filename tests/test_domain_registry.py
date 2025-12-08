"""
Tests for the domain registry system.

Tests cover:
- Domain registration
- Operator discovery
- Metadata extraction
- Registry queries
"""

import pytest
from morphogen.core.domain_registry import DomainRegistry, DomainDescriptor, register_stdlib_domains
from morphogen.core.operator import operator, OpCategory, get_operator_metadata, is_operator


class TestOperatorDecorator:
    """Test the @operator decorator."""

    def test_operator_decorator_attaches_metadata(self):
        """Test that @operator attaches metadata to functions."""
        @operator(
            domain="test",
            category=OpCategory.CONSTRUCT,
            signature="() -> int",
            deterministic=True,
            doc="Test function"
        )
        def test_func():
            return 42

        assert hasattr(test_func, '_operator_metadata')
        metadata = test_func._operator_metadata
        assert metadata.domain == "test"
        assert metadata.category == OpCategory.CONSTRUCT
        assert metadata.signature == "() -> int"
        assert metadata.deterministic is True
        assert metadata.doc == "Test function"

    def test_operator_preserves_function_behavior(self):
        """Test that @operator doesn't change function behavior."""
        @operator(
            domain="test",
            category=OpCategory.CONSTRUCT,
            signature="(x: int) -> int",
            deterministic=True
        )
        def double(x: int) -> int:
            return x * 2

        assert double(5) == 10
        assert double(0) == 0

    def test_get_operator_metadata(self):
        """Test get_operator_metadata helper."""
        @operator(
            domain="test",
            category=OpCategory.QUERY,
            signature="() -> bool",
            deterministic=True
        )
        def is_ready():
            return True

        metadata = get_operator_metadata(is_ready)
        assert metadata is not None
        assert metadata.domain == "test"
        assert metadata.category == OpCategory.QUERY

    def test_get_operator_metadata_returns_none_for_non_operators(self):
        """Test that get_operator_metadata returns None for non-operators."""
        def regular_function():
            return 42

        metadata = get_operator_metadata(regular_function)
        assert metadata is None

    def test_is_operator_helper(self):
        """Test is_operator helper function."""
        @operator(
            domain="test",
            category=OpCategory.QUERY,
            signature="() -> bool",
            deterministic=True
        )
        def decorated():
            return True

        def undecorated():
            return False

        assert is_operator(decorated) is True
        assert is_operator(undecorated) is False

    def test_all_op_categories(self):
        """Test that all OpCategory values work with decorator."""
        categories = [
            OpCategory.CONSTRUCT,
            OpCategory.TRANSFORM,
            OpCategory.QUERY,
            OpCategory.INTEGRATE,
            OpCategory.COMPOSE,
            OpCategory.MUTATE,
            OpCategory.SAMPLE,
            OpCategory.RENDER
        ]

        for category in categories:
            @operator(
                domain="test",
                category=category,
                signature="() -> int",
                deterministic=True
            )
            def test_func():
                return 42

            metadata = get_operator_metadata(test_func)
            assert metadata.category == category, f"Category {category.name} not working"

    def test_optional_metadata_fields(self):
        """Test optional metadata fields: pure, stateful, vectorized."""
        @operator(
            domain="test",
            category=OpCategory.TRANSFORM,
            signature="(x: int) -> int",
            deterministic=False,
            pure=False,
            stateful=True,
            vectorized=True
        )
        def stateful_op(x):
            return x * 2

        metadata = get_operator_metadata(stateful_op)
        assert metadata.deterministic is False
        assert metadata.pure is False
        assert metadata.stateful is True
        assert metadata.vectorized is True

    def test_optional_metadata_defaults(self):
        """Test that optional metadata fields have correct defaults."""
        @operator(
            domain="test",
            category=OpCategory.QUERY,
            signature="() -> bool"
        )
        def default_op():
            return True

        metadata = get_operator_metadata(default_op)
        # Check defaults
        assert metadata.deterministic is True  # default
        assert metadata.pure is True  # default
        assert metadata.stateful is False  # default
        assert metadata.vectorized is False  # default

    def test_doc_fallback_to_function_docstring(self):
        """Test that doc falls back to function __doc__ if not provided."""
        @operator(
            domain="test",
            category=OpCategory.QUERY,
            signature="() -> str"
        )
        def documented_func():
            """This is the docstring."""
            return "result"

        metadata = get_operator_metadata(documented_func)
        assert metadata.doc == "This is the docstring."

    def test_doc_explicit_overrides_docstring(self):
        """Test that explicit doc parameter overrides function docstring."""
        @operator(
            domain="test",
            category=OpCategory.QUERY,
            signature="() -> str",
            doc="Explicit documentation"
        )
        def documented_func():
            """This is the docstring."""
            return "result"

        metadata = get_operator_metadata(documented_func)
        assert metadata.doc == "Explicit documentation"

    def test_doc_empty_fallback(self):
        """Test that doc falls back to empty string if no doc provided."""
        @operator(
            domain="test",
            category=OpCategory.QUERY,
            signature="() -> str"
        )
        def undocumented_func():
            return "result"

        metadata = get_operator_metadata(undocumented_func)
        assert metadata.doc == ""


class TestOperatorMetadataValidation:
    """Test validation and error handling for operator metadata."""

    def test_empty_domain_name_still_works(self):
        """Test that empty domain name is allowed (though not recommended)."""
        # Note: Empty domain is technically allowed by current implementation
        @operator(
            domain="",
            category=OpCategory.QUERY,
            signature="() -> int"
        )
        def empty_domain_op():
            return 42

        metadata = get_operator_metadata(empty_domain_op)
        assert metadata.domain == ""

    def test_empty_signature_allowed(self):
        """Test that empty signature is allowed."""
        @operator(
            domain="test",
            category=OpCategory.QUERY,
            signature=""
        )
        def empty_sig_op():
            return 42

        metadata = get_operator_metadata(empty_sig_op)
        assert metadata.signature == ""

    def test_operator_with_args_and_kwargs(self):
        """Test that decorated functions work with *args and **kwargs."""
        @operator(
            domain="test",
            category=OpCategory.TRANSFORM,
            signature="(*args, **kwargs) -> Any",
            deterministic=False
        )
        def flexible_op(*args, **kwargs):
            return sum(args) + sum(kwargs.values())

        # Test that function still works
        result = flexible_op(1, 2, 3, x=4, y=5)
        assert result == 15

        # Test that metadata is attached
        metadata = get_operator_metadata(flexible_op)
        assert metadata is not None
        assert metadata.domain == "test"


class TestDomainDescriptor:
    """Test DomainDescriptor class."""

    def test_create_descriptor(self):
        """Test creating a domain descriptor."""
        descriptor = DomainDescriptor(
            name="test",
            module_path="test.module",
            version="1.0.0",
            description="Test domain"
        )

        assert descriptor.name == "test"
        assert descriptor.module_path == "test.module"
        assert descriptor.version == "1.0.0"
        assert descriptor.description == "Test domain"

    def test_get_operator(self):
        """Test getting an operator from descriptor."""
        def my_op():
            return 42

        descriptor = DomainDescriptor(
            name="test",
            module_path="test.module",
            operators={"my_op": my_op}
        )

        op = descriptor.get_operator("my_op")
        assert op is my_op
        assert op() == 42

    def test_get_nonexistent_operator_raises(self):
        """Test that getting nonexistent operator raises ValueError."""
        descriptor = DomainDescriptor(
            name="test",
            module_path="test.module"
        )

        with pytest.raises(ValueError, match="Operator 'missing' not found"):
            descriptor.get_operator("missing")

    def test_list_operators(self):
        """Test listing operators."""
        def op1():
            pass

        def op2():
            pass

        descriptor = DomainDescriptor(
            name="test",
            module_path="test.module",
            operators={"op2": op2, "op1": op1}
        )

        ops = descriptor.list_operators()
        assert ops == ["op1", "op2"]  # Should be sorted


class TestDomainRegistry:
    """Test DomainRegistry class."""

    def setup_method(self):
        """Clear registry before each test."""
        DomainRegistry.clear()

    def teardown_method(self):
        """Clear registry after each test."""
        DomainRegistry.clear()

    def test_register_domain_manually(self):
        """Test manually registering a domain."""
        # Create a simple test module inline
        import types
        test_module = types.ModuleType("test_domain")

        @operator(
            domain="test",
            category=OpCategory.CONSTRUCT,
            signature="() -> int",
            deterministic=True
        )
        def create():
            return 42

        test_module.create = create

        # Register by importing the module
        import sys
        sys.modules["test_domain"] = test_module

        try:
            descriptor = DomainRegistry.register(
                "test",
                "test_domain",
                "Test domain"
            )

            assert descriptor.name == "test"
            assert descriptor.module_path == "test_domain"
            assert "create" in descriptor.operators
        finally:
            # Cleanup
            del sys.modules["test_domain"]

    def test_get_domain(self):
        """Test getting a domain from registry."""
        import types
        test_module = types.ModuleType("test_domain2")

        @operator(
            domain="test2",
            category=OpCategory.QUERY,
            signature="() -> bool",
            deterministic=True
        )
        def check():
            return True

        test_module.check = check

        import sys
        sys.modules["test_domain2"] = test_module

        try:
            DomainRegistry.register("test2", "test_domain2")
            descriptor = DomainRegistry.get("test2")

            assert descriptor.name == "test2"
            assert "check" in descriptor.operators
        finally:
            del sys.modules["test_domain2"]

    def test_get_nonexistent_domain_raises(self):
        """Test that getting nonexistent domain raises ValueError."""
        with pytest.raises(ValueError, match="Domain 'missing' not registered"):
            DomainRegistry.get("missing")

    def test_list_domains(self):
        """Test listing registered domains."""
        import types
        import sys

        # Create two test modules
        for i in [1, 2]:
            mod = types.ModuleType(f"test_domain{i}")
            sys.modules[f"test_domain{i}"] = mod

        try:
            DomainRegistry.register("domain1", "test_domain1")
            DomainRegistry.register("domain2", "test_domain2")

            domains = DomainRegistry.list_domains()
            assert domains == ["domain1", "domain2"]  # Should be sorted
        finally:
            del sys.modules["test_domain1"]
            del sys.modules["test_domain2"]

    def test_has_domain(self):
        """Test checking if domain is registered."""
        import types
        import sys

        mod = types.ModuleType("test_domain3")
        sys.modules["test_domain3"] = mod

        try:
            assert not DomainRegistry.has_domain("test3")

            DomainRegistry.register("test3", "test_domain3")
            assert DomainRegistry.has_domain("test3")
        finally:
            del sys.modules["test_domain3"]

    def test_unregister_domain(self):
        """Test unregistering a domain."""
        import types
        import sys

        mod = types.ModuleType("test_domain4")
        sys.modules["test_domain4"] = mod

        try:
            DomainRegistry.register("test4", "test_domain4")
            assert DomainRegistry.has_domain("test4")

            DomainRegistry.unregister("test4")
            assert not DomainRegistry.has_domain("test4")
        finally:
            if "test_domain4" in sys.modules:
                del sys.modules["test_domain4"]

    def test_get_operator_from_registry(self):
        """Test getting an operator through the registry."""
        import types
        import sys

        mod = types.ModuleType("test_domain5")

        @operator(
            domain="test5",
            category=OpCategory.TRANSFORM,
            signature="(x: int) -> int",
            deterministic=True
        )
        def triple(x: int) -> int:
            return x * 3

        mod.triple = triple
        sys.modules["test_domain5"] = mod

        try:
            DomainRegistry.register("test5", "test_domain5")
            op = DomainRegistry.get_operator("test5", "triple")

            assert op(4) == 12
        finally:
            del sys.modules["test_domain5"]

    def test_operator_discovery_skips_private_attrs(self):
        """Test that operator discovery skips private attributes."""
        import types
        import sys

        mod = types.ModuleType("test_domain6")

        @operator(
            domain="test6",
            category=OpCategory.CONSTRUCT,
            signature="() -> int",
            deterministic=True
        )
        def public_op():
            return 1

        # This should be skipped
        def _private_function():
            return 2

        mod.public_op = public_op
        mod._private_function = _private_function

        sys.modules["test_domain6"] = mod

        try:
            descriptor = DomainRegistry.register("test6", "test_domain6")

            assert "public_op" in descriptor.operators
            assert "_private_function" not in descriptor.operators
        finally:
            del sys.modules["test_domain6"]

    def test_register_duplicate_domain_raises(self):
        """Test that registering duplicate domain raises ValueError."""
        import types
        import sys

        mod = types.ModuleType("test_domain7")
        sys.modules["test_domain7"] = mod

        try:
            DomainRegistry.register("test7", "test_domain7")

            with pytest.raises(ValueError, match="already registered"):
                DomainRegistry.register("test7", "test_domain7")
        finally:
            del sys.modules["test_domain7"]


class TestStdlibRegistration:
    """Test stdlib domain registration."""

    def setup_method(self):
        """Clear registry before each test."""
        DomainRegistry.clear()

    def teardown_method(self):
        """Clear registry after each test."""
        DomainRegistry.clear()

    def test_register_stdlib_domains(self):
        """Test that stdlib domains can be registered."""
        # This will attempt to register all 23 domains
        # Some may fail if they don't have @operator decorators yet
        register_stdlib_domains()

        # Just verify the function runs without crashing
        domains = DomainRegistry.list_domains()
        assert isinstance(domains, list)

    def test_initialize_registry(self):
        """Test initializing the registry."""
        assert not DomainRegistry._initialized

        DomainRegistry.initialize()
        assert DomainRegistry._initialized

        # Should be idempotent
        DomainRegistry.initialize()
        assert DomainRegistry._initialized


class TestDomainIntegration:
    """End-to-end integration tests for real domain usage."""

    def setup_method(self):
        """Clear registry before each test."""
        DomainRegistry.clear()

    def teardown_method(self):
        """Clear registry after each test."""
        DomainRegistry.clear()

    def test_end_to_end_domain_workflow(self):
        """Test complete workflow: register -> discover -> get metadata -> call operator."""
        # Initialize with real domains
        DomainRegistry.initialize()

        # Verify field domain is registered (from Phase 1)
        assert DomainRegistry.has_domain('field')

        # Get domain
        domain = DomainRegistry.get('field')
        assert domain.name == 'field'

        # List operators
        operators = domain.list_operators()
        assert len(operators) > 0
        assert isinstance(operators, list)

        # Get a specific operator
        if 'create_empty' in operators:
            op = domain.get_operator('create_empty')
            assert callable(op)
            assert is_operator(op)

            # Get metadata
            metadata = domain.get_operator_metadata('create_empty')
            assert metadata.domain == 'field'
            assert isinstance(metadata.category, OpCategory)

    def test_cross_domain_operator_access(self):
        """Test accessing operators from multiple domains."""
        DomainRegistry.initialize()

        # Get operators from different domains
        domains_to_test = ['field', 'audio', 'kinetics']

        for domain_name in domains_to_test:
            if DomainRegistry.has_domain(domain_name):
                domain = DomainRegistry.get(domain_name)
                operators = domain.list_operators()

                # Verify all operators have metadata
                for op_name in operators:
                    metadata = domain.get_operator_metadata(op_name)
                    assert metadata.domain == domain_name

    def test_get_operator_through_registry(self):
        """Test the convenience method for getting operators."""
        DomainRegistry.initialize()

        if DomainRegistry.has_domain('kinetics'):
            # Test DomainRegistry.get_operator shortcut
            op = DomainRegistry.get_operator('kinetics', 'arrhenius')
            assert callable(op)
            assert is_operator(op)

            # Verify it's the same as getting through domain
            domain = DomainRegistry.get('kinetics')
            op2 = domain.get_operator('arrhenius')
            assert op is op2

    def test_operator_metadata_consistency(self):
        """Test that metadata is consistent across access methods."""
        DomainRegistry.initialize()

        if DomainRegistry.has_domain('field'):
            domain = DomainRegistry.get('field')

            for op_name in domain.list_operators():
                # Get operator
                op = domain.get_operator(op_name)

                # Get metadata through domain
                metadata1 = domain.get_operator_metadata(op_name)

                # Get metadata directly from operator
                metadata2 = get_operator_metadata(op)

                # Should be the same object
                assert metadata1 is metadata2

    def test_domain_error_messages(self):
        """Test that error messages are helpful."""
        DomainRegistry.initialize()

        # Test helpful error when domain doesn't exist
        with pytest.raises(ValueError) as excinfo:
            DomainRegistry.get('nonexistent_domain')

        error_msg = str(excinfo.value)
        assert 'nonexistent_domain' in error_msg
        assert 'not registered' in error_msg
        assert 'Available domains:' in error_msg

    def test_domain_operator_not_found_error(self):
        """Test error when operator doesn't exist in domain."""
        DomainRegistry.initialize()

        if DomainRegistry.has_domain('field'):
            domain = DomainRegistry.get('field')

            with pytest.raises(ValueError) as excinfo:
                domain.get_operator('nonexistent_operator')

            error_msg = str(excinfo.value)
            assert 'nonexistent_operator' in error_msg
            assert 'not found' in error_msg
            assert 'field' in error_msg
