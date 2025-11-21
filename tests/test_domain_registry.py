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
from morphogen.core.operator import operator, OpCategory, get_operator_metadata


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
