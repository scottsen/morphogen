"""Code Generation for Kairo (v0.7.4 Phase 6)

This package provides JIT and AOT compilation capabilities for Kairo programs
using MLIR's ExecutionEngine with LLVM backend.

Components:
- jit.py: JIT compilation and execution with caching
- aot.py: Ahead-of-time compilation to native binaries/shared libraries
- executor.py: High-level ExecutionEngine API with memory management
- scf_to_llvm.py: SCF/Arith/Func to LLVM lowering (in lowering package)

Features (Phase 6):
- JIT compilation with caching (in-memory and disk)
- AOT compilation to executables, shared libs, object files
- LLVM optimization levels 0-3
- Memory management and automatic cleanup
- Thread-safe execution
- Cross-compilation support
"""

from .jit import KairoJIT, CompilationCache, create_jit, MLIR_AVAILABLE
from .aot import KairoAOT, OutputFormat, create_aot
from .executor import ExecutionEngine, ExecutionMode, MemoryBuffer, create_execution_engine

__all__ = [
    # JIT compilation
    "KairoJIT",
    "CompilationCache",
    "create_jit",

    # AOT compilation
    "KairoAOT",
    "OutputFormat",
    "create_aot",

    # Unified execution engine
    "ExecutionEngine",
    "ExecutionMode",
    "MemoryBuffer",
    "create_execution_engine",

    # Availability flag
    "MLIR_AVAILABLE",
]
