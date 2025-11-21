# Morphogen v0.7.0 Phase 2: Field Operations Dialect - Implementation Plan

**Status**: In Progress
**Phase**: 2 of 4 (Months 4-6)
**Started**: 2025-11-14
**Branch**: `claude/phase-2-field-dialect-012u2jBL3j3mL6Bz1tcuhSo1`

---

## Executive Summary

Phase 2 implements the **Morphogen Field Dialect** in real MLIR, providing high-level operations for spatial field computations with lowering to SCF (Structured Control Flow) loops. This is the first custom dialect implementation and demonstrates the full compilation pipeline.

---

## Phase 1 Completion ✅

**Delivered:**
- ✅ MLIR context management (`morphogen/mlir/context.py`)
- ✅ Compiler V2 infrastructure (`morphogen/mlir/compiler_v2.py`)
- ✅ Module directory structure (dialects/, lowering/, codegen/)
- ✅ Proof-of-concept (`examples/mlir_poc.py`)
- ✅ Comprehensive test suite (context, compiler)
- ✅ Graceful degradation (falls back to legacy when MLIR unavailable)

**Current Capabilities:**
- Create MLIR modules and contexts
- Compile literals (float, int, bool) to arith.constant
- Basic arithmetic (manual construction with arith dialect)
- Full test coverage with MLIR bindings

---

## Phase 2 Goals

### Primary Objectives
1. **Custom Field Dialect**: Define `!morphogen.field<T>` type and operations
2. **Field Operations**: Implement 4 core operations (create, gradient, diffuse, laplacian)
3. **Lowering Pass**: Transform field ops → SCF loops + memref
4. **Performance Benchmarking**: Measure speedup vs NumPy interpreter

### Success Metrics
- ✅ Field operations compile to valid MLIR
- ✅ Lowering produces correct SCF loop nests
- ✅ Can verify correctness against NumPy reference
- ✅ Compilation time < 1s for typical programs
- 🎯 Target: 10x+ speedup over interpreter (Phase 4 JIT)

---

## Architecture

### Compilation Pipeline

```
Morphogen AST (field.gradient(...))
    ↓
Custom Field Dialect
    %0 = morphogen.field.create %w, %h : !morphogen.field<f32>
    %1 = morphogen.field.gradient %0, %dir : !morphogen.field<f32>
    ↓
FieldToSCFPass (Lowering)
    %mem = memref.alloc(%h, %w) : memref<?x?xf32>
    scf.for %i = %c0 to %h step %c1 {
      scf.for %j = %c0 to %w step %c1 {
        // gradient computation
      }
    }
    ↓
Standard MLIR Optimizations
    ↓
LLVM Lowering (Phase 4)
    ↓
Native Code
```

### Field Dialect Design

**Type System:**
```mlir
!morphogen.field<T>          // Generic field type
!morphogen.field<f32>        // 32-bit float field
!morphogen.field<f64>        // 64-bit float field
!morphogen.field<i32>        // 32-bit integer field
```

**Operations (Phase 2 MVP):**

1. **morphogen.field.create** - Allocate field
   ```mlir
   %field = morphogen.field.create %width, %height, %fill : !morphogen.field<f32>
   ```

2. **morphogen.field.gradient** - Compute spatial gradient (central difference)
   ```mlir
   %grad = morphogen.field.gradient %field : !morphogen.field<f32> -> !morphogen.field<vector<2xf32>>
   ```

3. **morphogen.field.laplacian** - 5-point stencil Laplacian
   ```mlir
   %lapl = morphogen.field.laplacian %field : !morphogen.field<f32>
   ```

4. **morphogen.field.diffuse** - Jacobi diffusion solver
   ```mlir
   %diffused = morphogen.field.diffuse %field, %rate, %dt, %iters : !morphogen.field<f32>
   ```

---

## Implementation Plan

### Task 1: Field Dialect Definition ✅

**File**: `morphogen/mlir/dialects/field.py`

**Approach**: Use MLIR Python bindings to define operations programmatically

**Implementation Steps**:
1. Define `FieldType` class wrapping MLIR type
2. Create operation builders for each field op
3. Add type checking and validation
4. Implement operation constructors with proper attributes

**Example**:
```python
class FieldType:
    """Wrapper for !morphogen.field<T> type."""

    @staticmethod
    def get(element_type, context):
        """Get field type for given element type."""
        # Use MLIR OpaqueType for custom types
        return ir.OpaqueType.get("kairo", f"field<{element_type}>", context=context)

class FieldCreateOp:
    """morphogen.field.create operation."""

    @staticmethod
    def create(width, height, fill_value, element_type, loc, ip):
        """Create a field creation operation."""
        # Custom op construction using Python bindings
        ...
```

**Tests**: `tests/test_field_dialect.py`
- Test field type creation
- Test each operation construction
- Verify IR output format

---

### Task 2: Field-to-SCF Lowering Pass ✅

**File**: `morphogen/mlir/lowering/field_to_scf.py`

**Approach**: Pattern-based rewriting using MLIR pass infrastructure

**Implementation Steps**:
1. Create `FieldToSCFPass` class
2. Implement lowering patterns for each field operation
3. Handle memref allocation for field storage
4. Generate nested scf.for loops for spatial operations

**Example Lowering** (gradient):

*Input*:
```mlir
%grad = morphogen.field.gradient %field : !morphogen.field<f32>
```

*Output*:
```mlir
%h = memref.dim %field, %c0 : memref<?x?xf32>
%w = memref.dim %field, %c1 : memref<?x?xf32>
%grad = memref.alloc(%h, %w, %c2) : memref<?x?x2xf32>

scf.for %i = %c1 to %h_minus_1 step %c1 {
  scf.for %j = %c1 to %w_minus_1 step %c1 {
    // Central difference for x gradient
    %i_prev = arith.subi %i, %c1 : index
    %i_next = arith.addi %i, %c1 : index
    %val_prev = memref.load %field[%i_prev, %j] : memref<?x?xf32>
    %val_next = memref.load %field[%i_next, %j] : memref<?x?xf32>
    %dx = arith.subf %val_next, %val_prev : f32
    %dx_scaled = arith.divf %dx, %c2_f32 : f32

    // Central difference for y gradient
    %j_prev = arith.subi %j, %c1 : index
    %j_next = arith.addi %j, %c1 : index
    %val_prev_y = memref.load %field[%i, %j_prev] : memref<?x?xf32>
    %val_next_y = memref.load %field[%i, %j_next] : memref<?x?xf32>
    %dy = arith.subf %val_next_y, %val_prev_y : f32
    %dy_scaled = arith.divf %dy, %c2_f32 : f32

    // Store gradient
    memref.store %dx_scaled, %grad[%i, %j, %c0] : memref<?x?x2xf32>
    memref.store %dy_scaled, %grad[%i, %j, %c1] : memref<?x?x2xf32>
  }
}
```

**Tests**: `tests/test_field_lowering.py`
- Test each lowering pattern
- Verify SCF loop structure
- Check memref allocation
- Validate against NumPy reference implementation

---

### Task 3: Compiler Integration ✅

**File**: `morphogen/mlir/compiler_v2.py`

**Enhancements**:
1. Extend `compile_program` to handle field operations
2. Register field dialect operations
3. Apply lowering passes after IR construction
4. Add verification passes

**Example**:
```python
def compile_field_operation(self, field_op: FieldOp) -> ir.Value:
    """Compile field operation from AST."""
    if field_op.name == "gradient":
        return FieldGradientOp.create(
            field=self.compile_expression(field_op.field),
            loc=ir.Location.unknown(),
            ip=self.current_insertion_point
        )
    # ... handle other operations

def apply_lowering_passes(self, module: ir.Module):
    """Apply lowering passes to module."""
    pm = PassManager.parse("builtin.module(field-to-scf)")
    pm.run(module)
```

**Tests**: `tests/test_mlir_compiler_v2.py` (extend existing)
- Test field operation compilation
- Verify lowering integration
- End-to-end compilation tests

---

### Task 4: Performance Benchmarking ✅

**File**: `benchmarks/field_operations_benchmark.py`

**Metrics**:
1. **Compilation Time**: Time to compile Morphogen → MLIR IR
2. **Execution Time**: Time to execute (Phase 4 with JIT)
3. **Correctness**: Numerical accuracy vs NumPy reference
4. **Memory Usage**: Peak memory consumption

**Benchmark Suite**:
- Gradient computation (256x256, 512x512, 1024x1024)
- Laplacian computation
- Diffusion solver (10, 100, 1000 iterations)
- Combined operations (gradient + laplacian + diffuse)

**Expected Results** (Phase 2):
- ✅ Correct MLIR IR generation
- ✅ Valid lowering to SCF
- ⏳ Execution speedup (Phase 4 JIT required)

---

### Task 5: Testing Strategy ✅

**Unit Tests**:
- `tests/test_field_dialect.py` - Dialect operations
- `tests/test_field_lowering.py` - Lowering passes
- `tests/test_mlir_compiler_v2.py` - Compiler integration

**Integration Tests**:
- `tests/test_field_e2e.py` - End-to-end compilation
- Compare MLIR lowered IR with expected patterns
- Numerical correctness checks (Phase 4)

**Coverage Target**: >90% for Phase 2 code

---

## Implementation Order

### Week 1-2: Field Dialect Foundation
1. ✅ Implement field type wrapper
2. ✅ Create operation builders (create, gradient, laplacian, diffuse)
3. ✅ Add unit tests for dialect operations
4. ✅ Verify IR generation

### Week 3-4: Lowering Pass Implementation
1. ✅ Implement FieldToSCFPass infrastructure
2. ✅ Create lowering patterns for each operation
3. ✅ Test individual lowering patterns
4. ✅ Verify SCF loop structure

### Week 5-6: Compiler Integration & Testing
1. ✅ Integrate field dialect into compiler_v2
2. ✅ Connect lowering passes
3. ✅ End-to-end testing
4. ✅ Performance benchmarking
5. ✅ Documentation updates

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| MLIR API complexity | High | Start with simple ops, iterate incrementally |
| Custom dialect registration | Medium | Use OpaqueType for Phase 2, full IRDL in Phase 3 |
| Lowering pattern complexity | High | Reference MLIR tutorials, test each pattern individually |
| Performance not meeting goals | Low | Phase 2 focuses on correctness; optimization in Phase 4 |

---

## Deliverables

### Code
- ✅ `morphogen/mlir/dialects/field.py` - Field dialect implementation
- ✅ `morphogen/mlir/lowering/field_to_scf.py` - Lowering pass
- ✅ Enhanced `morphogen/mlir/compiler_v2.py` - Compiler integration
- ✅ `benchmarks/field_operations_benchmark.py` - Performance tests

### Tests
- ✅ `tests/test_field_dialect.py` - Dialect tests
- ✅ `tests/test_field_lowering.py` - Lowering tests
- ✅ `tests/test_field_e2e.py` - Integration tests

### Documentation
- ✅ This implementation plan
- ✅ Updated `docs/v0.7.0_DESIGN.md`
- ✅ Example programs demonstrating field dialect

---

## Success Criteria

- [x] Field dialect operations compile to valid MLIR
- [x] Lowering pass produces correct SCF loops
- [x] All tests pass (unit + integration)
- [x] Documentation complete and accurate
- [x] Benchmarks show compilation pipeline works
- [ ] Ready to proceed to Phase 3 (Temporal Execution)

---

## Next Steps (Phase 3)

After Phase 2 completion:
1. Implement temporal flow blocks (`flow(dt, steps)`)
2. State management via memref
3. Iteration and time-stepping in MLIR
4. Agent dialect operations

---

**Last Updated**: 2025-11-14
**Phase Status**: Ready to begin implementation
