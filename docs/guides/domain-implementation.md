# Morphogen Domain Implementation Guide

**Version:** 1.0
**Date:** 2025-11-15
**Audience:** Domain Developers, Core Contributors
**Related:** ADR-002 (Cross-Domain Patterns), ../architecture/domain-architecture.md

---

## Overview

This guide provides a **step-by-step process** for implementing a new domain in Morphogen. It incorporates proven patterns from TiaCAD (geometry), RiffStack (audio), and Strudel (patterns).

**What you'll learn:**
1. Domain design checklist
2. File structure and organization
3. Reference system design
4. Operator implementation (4-layer model)
5. Pass system development
6. Testing and validation
7. Documentation requirements

**Time estimate:** 2-4 weeks for a well-scoped domain (depending on complexity)

---

## Prerequisites

Before implementing a domain, you should:
- ✅ Read `../architecture/domain-architecture.md` (understand domain classification)
- ✅ Read `ADR-002` (cross-domain architectural patterns)
- ✅ Review existing domains: `morphogen/stdlib/audio.py`, `morphogen/stdlib/field.py`
- ✅ Understand MLIR basics (dialects, operations, lowering)
- ✅ Familiarize yourself with Morphogen's type system (`docs/../specifications/type-system.md`)

---

## Part 1: Domain Design Checklist

Before writing code, answer these design questions:

### 1.1 Domain Scope

- [ ] **Domain name**: What is the domain called? (e.g., Physics, Audio, Finance)
- [ ] **Domain purpose**: What problems does it solve? (1-2 sentences)
- [ ] **Core primitives**: What are the 5-10 fundamental types? (e.g., Body, Force, Field)
- [ ] **Key operations**: What are the most common operations? (e.g., integrate, apply_force)
- [ ] **Success criteria**: How do you know the domain works? (e.g., conservation of energy)

**Example: PhysicsDomain**
```
Name: PhysicsDomain
Purpose: Simulate physical systems (particles, rigid bodies, forces, integrators)
Core primitives: Body, Particle, Force, Integrator, SpatialPartition
Key operations: integrate, apply_force, detect_collision, compute_energy
Success criteria: Energy conservation (symplectic integrators), correct trajectories vs. analytical solutions
```

---

### 1.2 Reference System Design

**Critical decision:** What is your domain's **primary reference type**?

- [ ] **Reference name**: `{Domain}Ref` (e.g., PhysicsRef, AudioRef, FinanceRef)
- [ ] **What it references**: What does the reference point to? (e.g., a body, a node, a curve)
- [ ] **Auto-anchors**: What are 5-10 automatic access points? (e.g., `center_of_mass`, `input[0]`)
- [ ] **Frame semantics**: Does your domain use spatial frames? Temporal frames? Custom frames?

**Example: PhysicsDomain Reference Design**
```python
# Primary reference type
BodyRef
  - References: A rigid body or particle in the simulation
  - Auto-anchors:
      • .center_of_mass: Vector3
      • .local_axes.{x,y,z}: Basis vectors
      • .collision_normal: Vector3 (updated on collision)
      • .velocity: Vector3
      • .angular_momentum: Vector3
  - Frame: Spatial frame (inherits from ../specifications/coordinate-frames.md)
```

**Design principle:** One reference type to rule them all (avoid fragmentation).

---

### 1.3 Operator Taxonomy

**Plan your 4-layer operator hierarchy:**

| Layer | Type | Complexity | Example (PhysicsDomain) |
|-------|------|------------|-------------------------|
| **Layer 1** | Atomic | Single operation, no dependencies | `gravity_force_pair`, `euler_step` |
| **Layer 2** | Composite | Combines 2-5 atomic ops | `barnes_hut_force`, `rk4_integrator` |
| **Layer 3** | Constructs | Domain-specific patterns (10-50 ops) | `n_body_system`, `rigid_body_dynamics` |
| **Layer 4** | Presets | Pre-configured systems (50+ ops) | `solar_system`, `molecular_dynamics` |

- [ ] List 10-20 **Layer 1** atomic operators
- [ ] List 5-10 **Layer 2** composite operators
- [ ] List 3-5 **Layer 3** constructs
- [ ] List 1-3 **Layer 4** presets (can be added later)

**Design principle:** Start with Layer 1 (atomic ops), build upward.

---

### 1.4 Pass System Design

**Domain-specific optimization and lowering passes:**

- [ ] **Validation passes**: What domain invariants must be checked? (e.g., positive masses, valid time steps)
- [ ] **Optimization passes**: What domain-specific optimizations exist? (e.g., Barnes-Hut, filter merging)
- [ ] **Lowering passes**: How does domain IR lower to MLIR? (e.g., PhysicsDialect → SCF/Linalg)
- [ ] **Backend passes**: What CPU/GPU optimizations are needed? (e.g., vectorization, CUDA kernels)

**Example: PhysicsDomain Passes**
```
Validation:
  - PositiveMassCheck: Ensure all masses > 0
  - TimeStepStability: Ensure dt < critical value

Optimization:
  - SymplecticEnforcement: Replace Euler → Verlet for Hamiltonian systems
  - SpatialPartitioningOptimization: Choose grid vs. octree vs. BVH based on density

Lowering:
  - NBodyToBarnesHut: O(N²) → O(N log N)
  - IntegratorToSCF: For loops + arithmetic ops

Backend:
  - VectorizationPass: SIMD for force calculations
  - CUDALowering: GPU kernels for large particle counts
```

**Design principle:** Passes encode domain expertise, not just compilation.

---

## Part 2: File Structure and Organization

### 2.1 Directory Layout

Create the following structure for your domain (example: `physics`):

```
morphogen/
├── stdlib/
│   └── physics.py                    # High-level Python API (Layer 3-4 operators)
├── mlir/
│   ├── dialects/
│   │   └── physics.py                # MLIR dialect definition (PhysicsDialect)
│   ├── lowering/
│   │   ├── physics_to_scf.py         # Lower to SCF dialect
│   │   └── physics_to_linalg.py      # Lower to Linalg dialect (if needed)
│   └── passes/
│       └── physics/
│           ├── __init__.py
│           ├── symplectic_enforcement.py
│           ├── barnes_hut_optimization.py
│           └── vectorization.py
├── runtime/
│   └── physics_runtime.py            # Runtime support functions (if needed)
└── tests/
    ├── test_physics_ops.py           # Unit tests for operators
    ├── test_physics_passes.py        # Tests for passes
    └── test_physics_integration.py   # End-to-end integration tests

docs/
├── ../specifications/physics-domains.md                   # Domain specification
├── adr/
│   └── 003-physics-domain-design.md  # Design decisions
└── examples/
    └── physics/
        ├── n_body_solar_system.kairo
        └── rigid_body_collision.kairo
```

---

### 2.2 File-by-File Implementation Order

Implement in this order (dependencies flow downward):

1. **docs/../specifications/physics-domains.md** - Specification first (design before code)
2. **morphogen/mlir/dialects/physics.py** - MLIR dialect (types + operations)
3. **morphogen/stdlib/physics.py** - Python API (user-facing operators)
4. **morphogen/mlir/lowering/physics_to_scf.py** - Lowering to MLIR
5. **morphogen/mlir/passes/physics/*.py** - Optimization passes
6. **tests/test_physics_*.py** - Tests (continuous validation)
7. **docs/examples/physics/*.kairo** - Example programs

---

## Part 3: Implementation Steps

### Step 1: Write the Domain Specification

**File:** `docs/specifications/{domain}.md`

**Template:**
```markdown
# specifications/{domain}: {Full Domain Name}

**Status:** RFC | APPROVED | IMPLEMENTED
**Version:** 0.1
**Last Updated:** YYYY-MM-DD

## 1. Overview

[1-2 paragraphs describing the domain]

## 2. Core Types

### 2.1 Reference Type
[Define your {Domain}Ref]

### 2.2 Primitive Types
[List domain-specific types: Body, Force, Integrator, etc.]

## 3. Operators

### 3.1 Layer 1: Atomic Operators
[List with signatures]

### 3.2 Layer 2: Composite Operators
[List with signatures]

### 3.3 Layer 3: Constructs
[List with signatures]

## 4. Auto-Anchors

[Table of auto-generated anchors on reference types]

## 5. Passes

### 5.1 Validation Passes
### 5.2 Optimization Passes
### 5.3 Lowering Passes

## 6. Determinism Profile

[How does this domain ensure determinism?]

## 7. Examples

[2-3 code examples]

## 8. References

[Links to papers, existing systems, etc.]
```

**See:** `docs/../specifications/geometry.md` as a reference implementation.

---

### Step 2: Define MLIR Dialect

**File:** `morphogen/mlir/dialects/{domain}.py`

**Template:**
```python
"""
MLIR Dialect for {Domain}Domain.
Defines types and operations for {domain-specific purpose}.
"""

from typing import List, Optional
from dataclasses import dataclass
from morphogen.mlir.ir import (
    Dialect, Operation, Type, Attribute,
    OpView, register_op, register_type
)


# ============================================================================
# Dialect Definition
# ============================================================================

class {Domain}Dialect(Dialect):
    """MLIR dialect for {domain} operations."""

    name = "{domain}"
    namespace = "morphogen.{domain}"


# ============================================================================
# Types
# ============================================================================

@register_type("{domain}")
class {Primary}Type(Type):
    """Type representing a {primary object} in {domain}."""

    def __init__(self, *, metadata: Optional[dict] = None):
        self.metadata = metadata or {}

    def __str__(self):
        return f"!{domain}.{primary}"


@register_type("{domain}")
class {Secondary}Type(Type):
    """Type representing a {secondary object} in {domain}."""
    # ...


# ============================================================================
# Operations (Layer 1: Atomic)
# ============================================================================

@register_op("{domain}")
class {AtomicOp}Op(Operation):
    """
    {Description of atomic operation}.

    Signature:
        {signature}

    Determinism: DETERMINISTIC | NON_DETERMINISTIC
    Complexity: O(...)
    """

    name = "{domain}.{atomic_op}"

    def __init__(self, *args, **kwargs):
        # Implementation
        ...

    def verify(self):
        """Verify operation invariants."""
        # Check constraints
        ...


# ============================================================================
# Operations (Layer 2: Composite)
# ============================================================================

@register_op("{domain}")
class {CompositeOp}Op(Operation):
    """
    {Description of composite operation}.
    Composed of: {list of atomic ops}
    """

    name = "{domain}.{composite_op}"

    def build_from_atomic(self, ...):
        """Build this op from Layer 1 atomic ops."""
        # ...


# ============================================================================
# Helper Functions
# ============================================================================

def create_{domain}_context():
    """Create MLIR context with {domain} dialect registered."""
    from morphogen.mlir.ir import MLIRContext
    ctx = MLIRContext()
    ctx.register_dialect({Domain}Dialect)
    return ctx
```

**Reference:** See `morphogen/mlir/dialects/audio.py` for a complete example.

---

### Step 3: Implement Python API (stdlib)

**File:** `morphogen/stdlib/{domain}.py`

**Template:**
```python
"""
{Domain}Domain: High-level Python API for {domain description}.

This module provides Layer 3 and Layer 4 operators for {domain}.
"""

from typing import List, Optional, Union
from dataclasses import dataclass
from morphogen.ast.nodes import Node, Ref
from morphogen.ast.types import Type
from morphogen.stdlib.registry import operator, domain


# ============================================================================
# Reference Types
# ============================================================================

@domain("{domain}")
class {Primary}Ref(Ref):
    """
    Reference to a {primary object} in {domain}.

    Auto-anchors:
        - .{anchor1}: {type}
        - .{anchor2}: {type}
        - .{anchor3}: {type}
    """

    def __init__(self, name: str, **kwargs):
        super().__init__(name, domain="{domain}", **kwargs)

    @property
    def {anchor1}(self):
        """Auto-generated anchor: {description}."""
        return self._get_anchor("{anchor1}")

    @property
    def {anchor2}(self):
        """Auto-generated anchor: {description}."""
        return self._get_anchor("{anchor2}")


# ============================================================================
# Layer 1: Atomic Operators
# ============================================================================

@operator(
    domain="{domain}",
    layer=1,
    category="atomic",
    deterministic=True,
    tags=["core"]
)
def {atomic_op}(arg1: Type1, arg2: Type2) -> ReturnType:
    """
    {Description of atomic operation}.

    Args:
        arg1: {description}
        arg2: {description}

    Returns:
        {description}

    Example:
        >>> result = {atomic_op}(...)
    """
    # Implementation
    ...


# ============================================================================
# Layer 2: Composite Operators
# ============================================================================

@operator(
    domain="{domain}",
    layer=2,
    category="composite",
    deterministic=True,
    tags=["optimization"]
)
def {composite_op}(...) -> ReturnType:
    """
    {Description of composite operation}.
    Composes: {list of Layer 1 ops}
    """
    # Build from atomic ops
    ...


# ============================================================================
# Layer 3: Constructs
# ============================================================================

@operator(
    domain="{domain}",
    layer=3,
    category="construct",
    deterministic=True,
    tags=["high-level"]
)
def {construct}(...) -> ReturnType:
    """
    {Description of high-level construct}.

    This is a domain-specific pattern that combines multiple
    composite operators to achieve {goal}.
    """
    # Build complex system
    ...


# ============================================================================
# Layer 4: Presets
# ============================================================================

def preset_{name}(**overrides) -> {Primary}Ref:
    """
    Preset configuration for {use case}.

    Default parameters:
        - param1: value1
        - param2: value2

    Args:
        **overrides: Override default parameters

    Returns:
        Configured {primary object}

    Example:
        >>> system = preset_{name}(param1=custom_value)
    """
    defaults = {
        "param1": "value1",
        "param2": "value2",
    }
    config = {**defaults, **overrides}
    return {construct}(**config)
```

**Reference:** See `morphogen/stdlib/audio.py` for Layer 3-4 operators.

---

### Step 4: Implement Lowering to MLIR

**File:** `morphogen/mlir/lowering/{domain}_to_scf.py`

**Template:**
```python
"""
Lowering pass: {Domain}Dialect → SCF (Structured Control Flow).

This pass converts high-level {domain} operations to loops,
conditionals, and arithmetic operations.
"""

from morphogen.mlir.dialects.{domain} import {Domain}Dialect, {Op1}Op, {Op2}Op
from morphogen.mlir.dialects.builtin import scf, arith, func
from morphogen.mlir.passes import LoweringPass


class {Domain}ToSCFLoweringPass(LoweringPass):
    """
    Lower {Domain}Dialect operations to SCF dialect.

    Example:
        {domain}.{op} → scf.for + arith.add
    """

    def lower_{op}(self, op: {Op}Op):
        """
        Lower {domain}.{op} to SCF.

        Original:
            %result = {domain}.{op}(%arg1, %arg2)

        Lowered:
            %result = scf.for %i = 0 to %n {
                %tmp = arith.add %arg1, %arg2
                scf.yield %tmp
            }
        """
        # Lowering implementation
        ...

    def run(self, module):
        """Run lowering pass on entire module."""
        for op in module.walk():
            if isinstance(op, {Op1}Op):
                self.lower_{op1}(op)
            elif isinstance(op, {Op2}Op):
                self.lower_{op2}(op)
```

**Reference:** See `morphogen/mlir/lowering/audio_to_scf.py`.

---

### Step 5: Implement Domain Passes

**File:** `morphogen/mlir/passes/{domain}/{pass_name}.py`

**Example: Optimization Pass**
```python
"""
Symplectic Enforcement Pass for PhysicsDomain.

Replaces non-symplectic integrators with symplectic equivalents
for Hamiltonian systems to ensure energy conservation.
"""

from morphogen.mlir.passes import DomainPass
from morphogen.mlir.dialects.physics import IntegratorOp


class SymplecticEnforcementPass(DomainPass):
    """
    Enforce symplectic integration for Hamiltonian systems.

    Transformations:
        - euler + hamiltonian=True → verlet
        - rk4 + hamiltonian=True → yoshida4
    """

    domain = "physics"

    def visit_integrator_op(self, op: IntegratorOp):
        """Visit integrator operations and replace if needed."""

        if op.is_hamiltonian and not op.is_symplectic:
            if op.method == "euler":
                return self.replace_with_verlet(op)
            elif op.method == "rk4":
                return self.replace_with_yoshida(op)

        return op

    def replace_with_verlet(self, euler_op: IntegratorOp):
        """Replace Euler with Verlet integrator."""
        # Implementation
        ...
```

**Reference:** See `morphogen/mlir/passes/` for existing passes.

---

### Step 6: Write Tests

**File:** `tests/test_{domain}_ops.py`

**Template:**
```python
"""
Unit tests for {Domain}Domain operators.
"""

import pytest
from morphogen.stdlib.{domain} import {op1}, {op2}, {Primary}Ref


class Test{Domain}AtomicOps:
    """Tests for Layer 1 atomic operators."""

    def test_{op1}_basic(self):
        """Test {op1} with simple inputs."""
        result = {op1}(arg1=..., arg2=...)
        assert result == expected

    def test_{op1}_edge_cases(self):
        """Test {op1} with edge cases."""
        # Test zero values
        # Test negative values
        # Test large values
        ...

    def test_{op1}_determinism(self):
        """Ensure {op1} is deterministic."""
        result1 = {op1}(seed=42, ...)
        result2 = {op1}(seed=42, ...)
        assert result1 == result2


class Test{Domain}CompositeOps:
    """Tests for Layer 2 composite operators."""

    def test_{op2}_composition(self):
        """Test that {op2} correctly composes atomic ops."""
        # Verify composition
        ...


class Test{Domain}References:
    """Tests for reference system and auto-anchors."""

    def test_{primary}_ref_anchors(self):
        """Test auto-generated anchors on {Primary}Ref."""
        obj = {Primary}Ref("test_obj")

        # Test anchor existence
        assert hasattr(obj, "{anchor1}")
        assert hasattr(obj, "{anchor2}")

        # Test anchor types
        assert isinstance(obj.{anchor1}, ExpectedType)


class Test{Domain}Integration:
    """End-to-end integration tests."""

    def test_full_simulation(self):
        """Test complete {domain} simulation pipeline."""
        # Build system
        # Run simulation
        # Verify results against analytical solution
        ...
```

**Run tests:**
```bash
pytest tests/test_{domain}_ops.py -v
```

---

## Part 4: Documentation Requirements

### 4.1 Required Documentation

Every domain MUST include:

1. **Specification** (`docs/specifications/{domain}.md`)
   - Types, operators, passes, determinism profile
   - See template in Step 1

2. **Architecture Decision Record** (`docs/adr/{number}-{domain}-domain-design.md`)
   - Design rationale
   - Alternatives considered
   - Trade-offs

3. **Examples** (`docs/examples/{domain}/`)
   - At least 3 example programs
   - Cover beginner → intermediate → advanced

4. **API Reference** (auto-generated from docstrings)
   - Ensure all operators have complete docstrings
   - Include type annotations

### 4.2 Documentation Checklist

- [ ] Specification document complete
- [ ] ADR written and reviewed
- [ ] 3+ example programs
- [ ] Docstrings on all operators (with examples)
- [ ] Type annotations on all functions
- [ ] README in `morphogen/stdlib/{domain}/` explaining domain purpose
- [ ] Entry in `docs/../architecture/domain-architecture.md` (update roadmap)

---

## Part 5: Integration and Release

### 5.1 Pre-Release Checklist

Before merging your domain into main:

- [ ] All tests pass (`pytest tests/test_{domain}_*.py`)
- [ ] Type checking passes (`mypy morphogen/stdlib/{domain}.py`)
- [ ] Linting passes (`ruff check morphogen/stdlib/{domain}.py`)
- [ ] Documentation builds without errors
- [ ] Performance benchmarks run (if applicable)
- [ ] Code review completed (2+ reviewers)

### 5.2 Integration with Morphogen Core

Update these core files:

1. **`morphogen/stdlib/__init__.py`**
   ```python
   from morphogen.stdlib.{domain} import *
   ```

2. **`morphogen/mlir/dialects/__init__.py`**
   ```python
   from morphogen.mlir.dialects.{domain} import {Domain}Dialect
   ```

3. **`docs/../architecture/domain-architecture.md`**
   - Add your domain to the appropriate tier (Core/Next-Wave/Advanced)
   - Update roadmap section

4. **`docs/../specifications/operator-registry.md`**
   - Add your domain's operators to the registry tables

### 5.3 Versioning

Domain versions should follow Morphogen's overall version:
- **v0.8**: Core domains (Geometry, Audio, Fields)
- **v0.9**: Next-wave domains (Physics, Finance, Graphics)
- **v1.0**: Stable API for all core + next-wave domains

---

## Part 6: Advanced Topics

### 6.1 Cross-Domain Interfaces

If your domain needs to interact with other domains:

1. **Define interface contract** in `morphogen/interfaces/{source}_to_{target}.py`
   ```python
   class PhysicsToAudioInterface:
       """Map physics events → audio triggers."""

       @staticmethod
       def collision_to_percussion(collision_event):
           """Convert collision impulse to drum trigger."""
           ...
   ```

2. **Document in SPEC**: Add "Cross-Domain Interfaces" section
3. **Write integration tests**: `test_{source}_to_{target}_integration.py`

### 6.2 GPU Acceleration

If your domain benefits from GPU:

1. **CUDA lowering pass**: `morphogen/mlir/passes/{domain}/cuda_lowering.py`
2. **Benchmark**: Compare CPU vs. GPU performance
3. **Document requirements**: CUDA version, memory requirements

### 6.3 Determinism Strategies

Ensure deterministic execution:
- **Fixed RNG seeds**: Use `morphogen.random.deterministic_rng(seed)`
- **Operator ordering**: Document order-of-operations dependencies
- **Floating-point**: Use `morphogen.math.deterministic_sum()` for reductions
- **Profile**: Add determinism profile to SPEC

---

## Conclusion

**You're ready to implement a domain!**

**Recommended first domains to implement** (if not already done):
1. **PhysicsDomain** (N-body, integrators, forces) - Moderate complexity
2. **FinanceDomain** (Monte Carlo, stochastic processes) - High value
3. **PatternDomain** (Strudel-like sequencing) - Unique capability

**Key success factors:**
- ✅ Design before coding (write SPEC first)
- ✅ Follow 4-layer operator model
- ✅ Implement reference system with auto-anchors
- ✅ Write tests continuously (not at the end)
- ✅ Document everything (specs, examples, ADRs)

**Get help:**
- Review existing domains: `morphogen/stdlib/audio.py`, `morphogen/stdlib/field.py`
- Read ADR-002 for architectural patterns
- Ask questions in Morphogen development discussions

---

**Happy domain building! 🚀**
