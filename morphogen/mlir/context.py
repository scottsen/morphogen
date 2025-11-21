"""MLIR Context Management for Morphogen v0.7.0

This module provides the foundational MLIR context management for Morphogen's
real MLIR integration. It handles dialect registration, context lifecycle,
and provides a clean API for MLIR operations.

Note: This requires mlir Python bindings to be installed.
Install via: pip install mlir -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest
"""

from typing import Optional
import warnings


# Check if MLIR is available
MLIR_AVAILABLE = False
try:
    from mlir import ir
    from mlir.dialects import builtin, func, arith, scf, memref

    MLIR_AVAILABLE = True
except ImportError:
    warnings.warn(
        "MLIR Python bindings not found. v0.7.0 real MLIR integration requires "
        "MLIR to be installed. Falling back to legacy text-based IR generation. "
        "Install with: pip install mlir -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest",
        ImportWarning
    )
    # Create placeholder classes for graceful degradation
    ir = None


class MorphogenMLIRContext:
    """Manages MLIR context for Morphogen compilation.

    This class provides a centralized context for all MLIR operations in Morphogen,
    including dialect registration, module creation, and resource management.

    Example:
        >>> with MorphogenMLIRContext() as ctx:
        ...     module = ctx.create_module()
        ...     # Build IR using module
    """

    def __init__(self):
        """Initialize Morphogen MLIR context."""
        if not MLIR_AVAILABLE:
            raise RuntimeError(
                "MLIR Python bindings are required for v0.7.0 but not installed. "
                "Please install: pip install mlir -f https://github.com/makslevental/mlir-wheels/releases/expanded_assets/latest"
            )

        self.ctx = ir.Context()
        self.ctx.allow_unregistered_dialects = True  # For development
        self._register_dialects()

    def _register_dialects(self):
        """Register standard and custom Morphogen dialects.

        Standard dialects:
        - builtin: Core MLIR types and operations
        - func: Function definition and calls
        - arith: Arithmetic operations
        - scf: Structured control flow
        - memref: Memory reference operations

        Custom dialects:
        - morphogen.field: Field operations dialect (Phase 2, implemented)
        - morphogen.temporal: Temporal execution dialect (Phase 3, implemented)
        - morphogen.agent: Agent-based modeling dialect (Phase 4, implemented)
        - morphogen.audio: Audio synthesis dialect (Phase 5, implemented)
        - morphogen.visual: Visual rendering dialect (Phase 6, planned)
        """
        with self.ctx:
            # Load standard dialects
            # Note: In MLIR Python bindings, dialects are loaded on-demand
            # We just ensure the context is set up correctly
            # Custom Morphogen dialects are defined in morphogen.mlir.dialects
            pass

    def create_module(self, name: Optional[str] = None) -> "ir.Module":
        """Create a new MLIR module in this context.

        Args:
            name: Optional module name

        Returns:
            MLIR Module object

        Example:
            >>> module = ctx.create_module("my_morphogen_program")
        """
        with self.ctx:
            location = ir.Location.unknown()
            if name:
                module = ir.Module.create(location)
                module.operation.attributes["sym_name"] = ir.StringAttr.get(name)
                return module
            else:
                return ir.Module.create(location)

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        # Context cleanup handled by MLIR bindings
        pass


def get_mlir_context() -> MorphogenMLIRContext:
    """Get or create the global Morphogen MLIR context.

    Returns:
        Shared MorphogenMLIRContext instance

    Note:
        This provides a singleton-like pattern for the MLIR context.
        In most cases, you should use this instead of creating contexts directly.
    """
    global _global_context
    if _global_context is None:
        _global_context = MorphogenMLIRContext()
    return _global_context


def is_mlir_available() -> bool:
    """Check if MLIR Python bindings are available.

    Returns:
        True if MLIR is installed and available
    """
    return MLIR_AVAILABLE


# Global context instance
_global_context: Optional[MorphogenMLIRContext] = None
