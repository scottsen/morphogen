# Morphogen MLIR Phase 3: Temporal Execution Implementation

**Project**: Morphogen Programming Language - MLIR Compilation Pipeline
**Phase**: 3 of 5 - Temporal Execution (Flow Blocks)
**Repository**: https://github.com/scottsen/kairo
**Estimated Effort**: 2-3 days
**Quality Bar**: 10/10 (maintain established standard)

---

## Executive Summary

You are implementing **Phase 3 of the Morphogen MLIR compilation pipeline**: temporal execution via flow blocks. This is the most complex compilation phase, as it requires translating Morphogen's unique temporal flow construct into MLIR's control flow operations (specifically `scf.for` loops with proper state management).

**Current State**:
- ✅ Morphogen v0.3.1: 100% feature-complete, 160 runtime tests passing
- ✅ MLIR Phases 1-2: Complete (basic ops, functions, if/else, structs) - 21 tests passing
- ⏳ Total: 181/181 tests passing (including all v0.3.1 features)

**Your Mission**: Enable compilation of Morphogen's flow blocks (temporal iteration) to MLIR, allowing all 5 example programs to compile and demonstrating that Morphogen's temporal features can be efficiently lowered to standard imperative control flow.

**Success Criteria**:
1. Flow blocks compile to `scf.for` loops in MLIR
2. State variables (`@state`) properly managed with `memref`
3. `dt`, `steps`, and `substeps` parameters handled correctly
4. At least 10 new MLIR tests for flow block compilation
5. All 181 existing tests still pass (zero regressions)
6. Physics simulation examples compile successfully

---

## Project Context

### What is Morphogen?

Morphogen is a **temporal programming language** designed for physics simulations and time-based computations. Its key innovation is the **flow block** - a first-class construct for temporal iteration that makes time-evolving systems natural to express.

**Core Example**:
```morphogen
struct State {
    position: Float
    velocity: Float
}

@state
def simulate(s: State, dt: Float) -> State {
    s.velocity = s.velocity + -9.8 * dt  // gravity
    s.position = s.position + s.velocity * dt
    return s
}

let initial = State { position: 100.0, velocity: 0.0 }
let final = flow initial over 2.0 by 0.1 with simulate
```

This expresses: "Start with `initial` state, evolve it for 2.0 seconds in steps of 0.1 seconds, using the `simulate` function to update state at each step."

### What You're Building

You're building the **compiler** that translates Morphogen code into MLIR (Multi-Level Intermediate Representation), which can then be optimized and lowered to machine code.

**Current Architecture**:
```
Morphogen Source (.kairo)
    ↓ [Parser - COMPLETE]
Morphogen AST
    ↓ [Type Checker - COMPLETE]
Typed AST
    ↓ [MLIR Compiler - 40% COMPLETE]
MLIR IR (text format)
    ↓ [Future: Real MLIR → LLVM → Native Code]
Executable
```

**What's Working** (Phases 1-2):
- Arithmetic: `a + b * c`
- Functions: `def foo(x: Float) -> Float { return x * 2.0 }`
- If/else: `if x > 0.0 { y } else { z }`
- Structs: `Point { x: 1.0, y: 2.0 }`
- Function calls: `foo(42.0)`

**What You're Adding** (Phase 3):
- Flow blocks: `flow state over duration by dt with update_fn`
- State variables: `@state` decorator
- Temporal parameters: `dt`, `steps`, `substeps`

---

## Technical Background: Flow Blocks

### Morphogen Syntax

Flow blocks have three forms:

**1. Time-based flow (duration + dt)**:
```morphogen
flow initial_state over 2.0 by 0.1 with update_fn
```
Means: "Iterate for 2.0 time units, in steps of 0.1"
Iterations: 20 (2.0 / 0.1)

**2. Step-based flow (explicit steps)**:
```morphogen
flow initial_state for 10 steps with update_fn
```
Means: "Iterate exactly 10 times"
`dt` is not provided to update function

**3. Substeps (nested iteration)**:
```morphogen
flow state over 1.0 by 0.01 substeps 5 with update_fn
```
Means: "Outer loop: 100 steps of 0.01. Inner loop: 5 substeps per step"
Total iterations: 500
Used for Runge-Kutta and multi-rate integration

### AST Representation

```python
@dataclass
class FlowExpr(Expr):
    initial_state: Expr
    duration: Optional[Expr]  # for time-based
    dt: Optional[Expr]         # for time-based
    steps: Optional[Expr]      # for step-based
    substeps: Optional[Expr]   # for nested iteration
    update_fn: Expr
```

### Runtime Semantics (Reference Implementation)

The Python runtime (already working) shows the correct semantics:

```python
# Time-based flow
def execute_flow_time_based(initial, duration, dt, update_fn):
    state = initial
    num_steps = int(duration / dt)
    for i in range(num_steps):
        state = update_fn(state, dt)
    return state

# Step-based flow
def execute_flow_step_based(initial, steps, update_fn):
    state = initial
    for i in range(steps):
        state = update_fn(state)
    return state

# With substeps
def execute_flow_with_substeps(initial, duration, dt, substeps, update_fn):
    state = initial
    num_steps = int(duration / dt)
    for i in range(num_steps):
        for j in range(substeps):
            state = update_fn(state, dt)
    return state
```

### MLIR Target: scf.for

MLIR's structured control flow dialect provides `scf.for` for loops:

```mlir
// C equivalent: for (i = 0; i < 10; i++) { ... }
scf.for %i = %c0 to %c10 step %c1 {
    // loop body
    scf.yield
}
```

For flow blocks with state:
```mlir
%result = scf.for %i = %c0 to %num_steps step %c1
    iter_args(%state = %initial) -> (f64) {
    // Compute new state
    %new_state = call @update(%state, %dt) : (f64, f64) -> f64
    scf.yield %new_state : f64
}
```

The `iter_args` mechanism threads state through loop iterations (like a fold/reduce operation).

---

## Implementation Strategy

### Overview

Flow block compilation requires:

1. **Compute iteration count**: `duration / dt` or use `steps` directly
2. **Create scf.for loop**: With proper bounds and step
3. **Thread state through iterations**: Using `iter_args`
4. **Call update function**: Pass current state + dt (if applicable)
5. **Yield new state**: Return updated state for next iteration
6. **Handle substeps**: Nested `scf.for` loops

### State Variable Management

Morphogen's `@state` decorator marks functions that update state through flow blocks:

```morphogen
@state
def update(s: State, dt: Float) -> State {
    // Modify s and return it
}
```

In MLIR compilation:
- State values are SSA values (immutable)
- Each iteration produces a new SSA value
- `scf.for` with `iter_args` threads these values through iterations
- No need for `memref` unless you're doing real mutation (which we're not in this phase)

### Code Structure

Your implementation will touch:

1. **`ir_builder.py`**: Add `create_for_loop()` method
2. **`mlir_compiler.py`**: Add `visit_FlowExpr()` method
3. **`tests/test_mlir_compiler.py`**: Add flow block tests
4. **Examples**: Verify existing examples compile

---

## Detailed Implementation Guide

### Step 1: Extend IR Builder (ir_builder.py)

Add support for `scf.for` loops:

```python
def create_for_loop(
    self,
    start: SSAValue,
    end: SSAValue,
    step: SSAValue,
    iter_args: List[SSAValue],
    body_fn,
    result_types: List[str]
) -> SSAValue:
    """
    Create an scf.for loop.

    Args:
        start: Loop start index (SSA value)
        end: Loop end index (SSA value)
        step: Loop step (SSA value)
        iter_args: Initial values for iteration arguments
        body_fn: Function that takes (loop_var, *iter_args) and returns new iter_args
        result_types: MLIR types for loop results

    Returns:
        SSAValue for loop result (or tuple of SSAValues if multiple results)
    """
    result = self.new_ssa()

    # Format iter_args
    iter_args_str = ", ".join(f"%{arg.name}" for arg in iter_args)
    iter_types_str = ", ".join(result_types)

    # Start loop
    self.emit(f"%{result.name}:N = scf.for %{self.new_ssa().name} = %{start.name} "
              f"to %{end.name} step %{step.name}")
    if iter_args:
        self.emit(f"    iter_args({iter_args_str}) -> ({iter_types_str}) {{")
    else:
        self.emit(f" {{")

    # Generate body
    self.indent()
    loop_var = SSAValue(f"iv_{self.ssa_counter}", "index")
    new_iter_args = body_fn(loop_var, iter_args)

    # Yield results
    yield_vals = ", ".join(f"%{arg.name}" for arg in new_iter_args)
    yield_types = ", ".join(result_types)
    self.emit(f"scf.yield {yield_vals} : {yield_types}")
    self.dedent()
    self.emit("}")

    return result
```

**Note**: This is a simplified sketch. You'll need to handle:
- Single vs multiple result values
- Proper SSA naming
- Indentation for nested scopes
- Type annotations

### Step 2: Implement visit_FlowExpr (mlir_compiler.py)

This is the core of Phase 3:

```python
def visit_FlowExpr(self, node: FlowExpr) -> SSAValue:
    """
    Compile flow block to scf.for loop.

    Flow block forms:
    1. Time-based: flow state over duration by dt with fn
    2. Step-based: flow state for steps with fn
    3. With substeps: flow state over duration by dt substeps N with fn
    """
    # Compile initial state
    state = self.visit(node.initial_state)
    state_type = self.mlir_type(node.type)  # Result type from type checker

    # Determine iteration count
    if node.duration and node.dt:
        # Time-based: num_steps = duration / dt
        duration = self.visit(node.duration)
        dt = self.visit(node.dt)

        # Convert to integer (truncate)
        num_steps = self.builder.emit_op(
            "arith.divf",
            [duration, dt],
            ["f64"]
        )
        num_steps_int = self.builder.emit_op(
            "arith.fptosi",
            [num_steps],
            ["i32"]
        )
    elif node.steps:
        # Step-based: explicit steps
        num_steps_int = self.visit(node.steps)
        dt = None
    else:
        raise CompileError("Flow block must have either (duration, dt) or steps")

    # Create loop bounds
    zero = self.builder.emit_constant(0, "index")
    end = self.builder.emit_op(
        "arith.index_cast",
        [num_steps_int],
        ["index"]
    )
    one = self.builder.emit_constant(1, "index")

    # Handle substeps (nested loop)
    if node.substeps:
        substep_count = self.visit(node.substeps)
        # ... (implement nested loop - see below)

    # Compile loop body
    def body_fn(loop_var, iter_args):
        current_state = iter_args[0]

        # Call update function with state and dt (if time-based)
        update_fn = self.visit(node.update_fn)
        if dt:
            # Time-based: fn(state, dt)
            args = [current_state, dt]
        else:
            # Step-based: fn(state)
            args = [current_state]

        new_state = self.builder.emit_call(
            update_fn.name,  # Function name
            args,
            [state_type]
        )

        return [new_state]

    # Create the loop
    result = self.builder.create_for_loop(
        start=zero,
        end=end,
        step=one,
        iter_args=[state],
        body_fn=body_fn,
        result_types=[state_type]
    )

    return result
```

### Step 3: Handle Substeps (Nested Loops)

For substeps, you need nested `scf.for` loops:

```python
# Outer loop: main steps
outer_result = self.builder.create_for_loop(...)

    # Inner loop body:
    def outer_body(outer_var, outer_iter_args):
        # Inner loop: substeps
        inner_result = self.builder.create_for_loop(
            start=zero,
            end=substep_count,
            step=one,
            iter_args=outer_iter_args,
            body_fn=inner_body_fn,
            result_types=[state_type]
        )
        return [inner_result]
```

### Step 4: Handle @state Decorator

The `@state` decorator is metadata that doesn't affect compilation (it's for documentation and runtime checks). During MLIR compilation, treat `@state` functions like normal functions.

In AST:
```python
@dataclass
class FunctionDef:
    name: str
    params: List[Param]
    return_type: Type
    body: List[Stmt]
    decorators: List[str]  # ["state"] if @state is present
```

In MLIR compilation, ignore decorators (or use them for optimization hints).

---

## Testing Strategy

### Test Progression

Add tests incrementally:

**1. Simple flow (time-based, scalar state)**:
```morphogen
def increment(x: Float, dt: Float) -> Float {
    return x + dt
}

let result = flow 0.0 over 1.0 by 0.1 with increment
// Expected: 1.0 (10 steps × 0.1)
```

**2. Flow with struct state**:
```morphogen
struct Point {
    x: Float
    y: Float
}

@state
def move(p: Point, dt: Float) -> Point {
    return Point { x: p.x + dt, y: p.y + dt }
}

let start = Point { x: 0.0, y: 0.0 }
let end = flow start over 1.0 by 0.1 with move
// Expected: Point { x: 1.0, y: 1.0 }
```

**3. Step-based flow**:
```morphogen
def double(x: Float) -> Float {
    return x * 2.0
}

let result = flow 1.0 for 5 steps with double
// Expected: 32.0 (1.0 × 2^5)
```

**4. Flow with substeps**:
```morphogen
def tiny_step(x: Float, dt: Float) -> Float {
    return x + dt
}

let result = flow 0.0 over 1.0 by 0.1 substeps 2 with tiny_step
// Expected: 2.0 (10 steps × 2 substeps × 0.1)
```

**5. Physics simulation (spring oscillator)**:
```morphogen
struct Spring {
    position: Float
    velocity: Float
}

@state
def spring_step(s: Spring, dt: Float) -> Spring {
    let k = -10.0  // spring constant
    let accel = k * s.position
    return Spring {
        position: s.position + s.velocity * dt,
        velocity: s.velocity + accel * dt
    }
}

let initial = Spring { position: 1.0, velocity: 0.0 }
let final = flow initial over 2.0 by 0.01 with spring_step
```

**6. Nested state (particle system)**:
```morphogen
struct Particle {
    x: Float
    vx: Float
}

struct System {
    p1: Particle
    p2: Particle
}

@state
def update_system(sys: System, dt: Float) -> System {
    return System {
        p1: Particle { x: sys.p1.x + sys.p1.vx * dt, vx: sys.p1.vx },
        p2: Particle { x: sys.p2.x + sys.p2.vx * dt, vx: sys.p2.vx }
    }
}

let init = System {
    p1: Particle { x: 0.0, vx: 1.0 },
    p2: Particle { x: 10.0, vx: -1.0 }
}

let result = flow init over 5.0 by 0.1 with update_system
```

### Test File Structure

Add to `tests/test_mlir_compiler.py`:

```python
class TestFlowBlocks(unittest.TestCase):
    """Tests for MLIR compilation of flow blocks (Phase 3)."""

    def test_simple_flow_scalar(self):
        """Flow with scalar state, time-based."""
        code = """
        def increment(x: Float, dt: Float) -> Float {
            return x + dt
        }

        def main() -> Float {
            return flow 0.0 over 1.0 by 0.1 with increment
        }
        """
        mlir = compile_to_mlir(code)
        self.assertIn("scf.for", mlir)
        self.assertIn("iter_args", mlir)
        # Verify structure...

    def test_flow_with_struct(self):
        """Flow with struct state."""
        code = """
        struct Point {
            x: Float
            y: Float
        }

        @state
        def move(p: Point, dt: Float) -> Point {
            return Point { x: p.x + dt, y: p.y + dt }
        }

        def main() -> Point {
            let start = Point { x: 0.0, y: 0.0 }
            return flow start over 1.0 by 0.1 with move
        }
        """
        mlir = compile_to_mlir(code)
        self.assertIn("scf.for", mlir)
        # Check struct type in loop...

    def test_step_based_flow(self):
        """Flow with explicit steps (no dt)."""
        code = """
        def double(x: Float) -> Float {
            return x * 2.0
        }

        def main() -> Float {
            return flow 1.0 for 5 steps with double
        }
        """
        mlir = compile_to_mlir(code)
        self.assertIn("scf.for", mlir)
        self.assertIn("%c5", mlir)  # Constant 5 for steps

    def test_flow_with_substeps(self):
        """Flow with substeps (nested loops)."""
        code = """
        def tiny_step(x: Float, dt: Float) -> Float {
            return x + dt
        }

        def main() -> Float {
            return flow 0.0 over 1.0 by 0.1 substeps 2 with tiny_step
        }
        """
        mlir = compile_to_mlir(code)
        # Should have nested scf.for loops
        self.assertEqual(mlir.count("scf.for"), 2)

    def test_physics_simulation(self):
        """Realistic physics example: spring oscillator."""
        code = """
        struct Spring {
            position: Float
            velocity: Float
        }

        @state
        def spring_step(s: Spring, dt: Float) -> Spring {
            let k = -10.0
            let accel = k * s.position
            return Spring {
                position: s.position + s.velocity * dt,
                velocity: s.velocity + accel * dt
            }
        }

        def main() -> Spring {
            let initial = Spring { position: 1.0, velocity: 0.0 }
            return flow initial over 2.0 by 0.01 with spring_step
        }
        """
        mlir = compile_to_mlir(code)
        self.assertIn("scf.for", mlir)
        self.assertIn("Spring", mlir)  # Struct type
        # Verify loop compiles correctly

    # Add more tests...
```

### Target: 10+ Tests

Aim for at least 10 comprehensive tests covering:
- ✅ Scalar state, time-based
- ✅ Struct state, time-based
- ✅ Step-based (no dt)
- ✅ Substeps (nested loops)
- ✅ Physics simulation (spring)
- ✅ Multiple state fields
- ✅ Nested structs in state
- ✅ Complex update functions
- ✅ Edge cases (1 step, large step counts)
- ✅ Decorator handling (@state)

---

## Expected MLIR Output Examples

### Example 1: Simple Flow

**Morphogen**:
```morphogen
def increment(x: Float, dt: Float) -> Float {
    return x + dt
}

def main() -> Float {
    return flow 0.0 over 1.0 by 0.1 with increment
}
```

**MLIR** (target output):
```mlir
func.func @increment(%arg0: f64, %arg1: f64) -> f64 {
    %0 = arith.addf %arg0, %arg1 : f64
    return %0 : f64
}

func.func @main() -> f64 {
    %c0 = arith.constant 0.000000e+00 : f64
    %c1_0 = arith.constant 1.000000e+00 : f64
    %c0_1 = arith.constant 1.000000e-01 : f64

    // Compute num_steps = 1.0 / 0.1 = 10
    %num_steps_f = arith.divf %c1_0, %c0_1 : f64
    %num_steps = arith.fptosi %num_steps_f : f64 to i32

    // Loop setup
    %c0_idx = arith.constant 0 : index
    %c1_idx = arith.constant 1 : index
    %end_idx = arith.index_cast %num_steps : i32 to index

    // Flow loop
    %result = scf.for %iv = %c0_idx to %end_idx step %c1_idx
        iter_args(%state = %c0) -> (f64) {
        %new_state = call @increment(%state, %c0_1) : (f64, f64) -> f64
        scf.yield %new_state : f64
    }

    return %result : f64
}
```

### Example 2: Struct State

**Morphogen**:
```morphogen
struct Point {
    x: Float
    y: Float
}

@state
def move(p: Point, dt: Float) -> Point {
    return Point { x: p.x + dt, y: p.y + dt }
}

def main() -> Point {
    let start = Point { x: 0.0, y: 0.0 }
    return flow start over 1.0 by 0.1 with move
}
```

**MLIR** (target output):
```mlir
!Point = type { f64, f64 }

func.func @move(%arg0: !Point, %arg1: f64) -> !Point {
    %x = arith.extractvalue %arg0[0] : !Point
    %y = arith.extractvalue %arg0[1] : !Point
    %new_x = arith.addf %x, %arg1 : f64
    %new_y = arith.addf %y, %arg1 : f64
    %result = arith.insertvalue %new_x into undef[0] : !Point
    %result2 = arith.insertvalue %new_y into %result[1] : !Point
    return %result2 : !Point
}

func.func @main() -> !Point {
    %c0 = arith.constant 0.000000e+00 : f64
    %start = arith.insertvalue %c0 into undef[0] : !Point
    %start2 = arith.insertvalue %c0 into %start[1] : !Point

    %duration = arith.constant 1.000000e+00 : f64
    %dt = arith.constant 1.000000e-01 : f64

    %num_steps_f = arith.divf %duration, %dt : f64
    %num_steps = arith.fptosi %num_steps_f : f64 to i32

    %c0_idx = arith.constant 0 : index
    %c1_idx = arith.constant 1 : index
    %end_idx = arith.index_cast %num_steps : i32 to index

    %result = scf.for %iv = %c0_idx to %end_idx step %c1_idx
        iter_args(%state = %start2) -> (!Point) {
        %new_state = call @move(%state, %dt) : (!Point, f64) -> !Point
        scf.yield %new_state : !Point
    }

    return %result : !Point
}
```

---

## Integration with Existing Code

### Files You'll Modify

**1. `morphogen/ir_builder.py`** (~315 lines currently)
- Add: `create_for_loop()` method
- Add: Helper for `scf.yield`
- Add: Scope management for nested blocks
- Lines added: ~80-100

**2. `morphogen/mlir_compiler.py`** (~977 lines currently)
- Add: `visit_FlowExpr()` method
- Add: Helper for computing loop bounds
- Add: Substep handling
- Lines added: ~120-150

**3. `tests/test_mlir_compiler.py`** (~580 lines currently)
- Add: `TestFlowBlocks` class
- Add: 10+ test methods
- Lines added: ~200-250

**Total new code**: ~400-500 lines

### Existing Infrastructure You'll Use

**From Phase 1-2 (already working)**:
- ✅ Type system (`mlir_type()` function)
- ✅ Function definitions (`visit_FunctionDef()`)
- ✅ Function calls (`emit_call()`)
- ✅ Arithmetic operations (`emit_op()`)
- ✅ Constants (`emit_constant()`)
- ✅ Struct operations (`visit_StructLiteral()`, field extraction)
- ✅ Control flow (`visit_IfExpr()` - reference for scoped blocks)

**You're adding**:
- ⏳ `scf.for` loops
- ⏳ Iteration arguments (`iter_args`)
- ⏳ Loop bounds computation
- ⏳ Nested loops (substeps)

---

## Common Pitfalls & How to Avoid Them

### 1. SSA Value Management

**Problem**: Forgetting that MLIR is SSA - values are immutable.

**Solution**: Each loop iteration must produce a new SSA value. Use `iter_args` to thread state.

**Example**:
```python
# WRONG: Trying to mutate state
%state = ...  // initial
scf.for ... {
    %state = call @update(%state)  // ERROR: can't redefine %state
}

# RIGHT: Thread state through iter_args
%final = scf.for ... iter_args(%state = %initial) -> (f64) {
    %new_state = call @update(%state)
    scf.yield %new_state : f64
}
```

### 2. Type Mismatches in Loops

**Problem**: `scf.for` loop result types must match `scf.yield` types.

**Solution**: Carefully track types through `iter_args`.

**Example**:
```python
# Loop declaration and yield must match
%result = scf.for ... iter_args(%x = %init) -> (f64) {  // Says: returns f64
    ...
    scf.yield %new_x : f64  // Must yield f64
}
```

### 3. Integer vs Float for Loop Bounds

**Problem**: `scf.for` requires `index` type for loop variable, but iteration count comes from float division.

**Solution**: Convert properly:
1. Divide duration by dt (float operation)
2. Convert to integer (`fptosi`)
3. Convert to `index` (`index_cast`)

### 4. Substeps Complexity

**Problem**: Nested loops can get confusing.

**Solution**: Implement single loops first, test thoroughly, then add substeps as a separate feature.

### 5. Function Call Signatures

**Problem**: Update functions have different signatures (with/without dt).

**Solution**: Check whether flow is time-based or step-based, and call accordingly:
```python
if dt:
    call @update(%state, %dt)
else:
    call @update(%state)
```

---

## Definition of Done

Your implementation is complete when:

### Code Complete
- [ ] `create_for_loop()` method in `ir_builder.py` works correctly
- [ ] `visit_FlowExpr()` method in `mlir_compiler.py` handles all flow forms
- [ ] Substeps (nested loops) implemented
- [ ] Code is clean, well-commented, follows existing patterns

### Tests Pass
- [ ] All 181 existing tests still pass (zero regressions)
- [ ] At least 10 new MLIR flow block tests added
- [ ] All new tests pass
- [ ] Tests cover: time-based, step-based, substeps, scalars, structs

### Examples Compile
- [ ] At least one physics example (spring, projectile, or oscillator) compiles
- [ ] Generated MLIR is structurally correct (valid syntax)
- [ ] MLIR output matches expected patterns (see examples above)

### Documentation
- [ ] Code comments explain key decisions
- [ ] Test descriptions are clear
- [ ] Any tricky parts have explanatory comments

### Quality Bar
- [ ] Code quality: 10/10 (clean, maintainable, professional)
- [ ] Test coverage: Comprehensive
- [ ] Documentation: Clear and complete
- [ ] No hacks or shortcuts: Proper implementation

---

## Development Workflow

### Recommended Approach

**Phase 3A: Basic Flow Loops** (Day 1)
1. Implement `create_for_loop()` in `ir_builder.py`
2. Implement basic `visit_FlowExpr()` for time-based flow
3. Add 3-4 tests: scalar state, simple updates
4. Verify MLIR output structure

**Phase 3B: Struct State** (Day 1-2)
1. Extend `visit_FlowExpr()` to handle struct state
2. Ensure struct values thread through `iter_args`
3. Add 2-3 tests: struct state, nested structs
4. Test physics example (spring oscillator)

**Phase 3C: Step-Based Flow** (Day 2)
1. Add logic for `steps` parameter (no dt)
2. Adjust function call signature (no dt argument)
3. Add 2 tests: step-based flow
4. Verify existing tests still pass

**Phase 3D: Substeps** (Day 2-3)
1. Implement nested loop logic
2. Add 2-3 tests: substeps with various configurations
3. Test complex example (Runge-Kutta integration)

**Phase 3E: Polish** (Day 3)
1. Review all code, add comments
2. Ensure test coverage is comprehensive
3. Verify all 191+ tests pass
4. Check MLIR output quality
5. Run existing examples, verify they compile

### Testing Discipline

- **Run tests after every change**: `pytest tests/test_mlir_compiler.py -v`
- **Check for regressions**: Ensure existing 181 tests keep passing
- **Verify MLIR output**: Manually inspect generated MLIR for a few tests
- **Test incrementally**: Don't implement everything then test - test as you go

### Git Workflow

```bash
# Start fresh
git checkout master
git pull
git checkout -b feature/mlir-phase3-flow-blocks

# Commit incrementally
git add morphogen/ir_builder.py
git commit -m "Add create_for_loop() method for scf.for support"

git add morphogen/mlir_compiler.py
git commit -m "Implement visit_FlowExpr() for time-based flow"

git add tests/test_mlir_compiler.py
git commit -m "Add tests for basic flow block compilation"

# Continue with small, focused commits...

# When complete
git push origin feature/mlir-phase3-flow-blocks
# Create PR with detailed description
```

---

## Code Examples & Scaffolding

### IR Builder Scaffold

```python
class IRBuilder:
    # ... existing code ...

    def create_for_loop(
        self,
        start: SSAValue,
        end: SSAValue,
        step: SSAValue,
        iter_args: List[SSAValue],
        body_fn: Callable,
        result_types: List[str]
    ) -> Union[SSAValue, List[SSAValue]]:
        """
        Create an scf.for loop with iteration arguments.

        Generates MLIR like:
            %result = scf.for %iv = %start to %end step %step
                iter_args(%arg1 = %init1, ...) -> (type1, ...) {
                // body generated by body_fn
                scf.yield %new_arg1, ... : type1, ...
            }

        Args:
            start: Loop start bound (index type)
            end: Loop end bound (index type)
            step: Loop step (index type)
            iter_args: Initial values for iteration arguments
            body_fn: Function(loop_var, iter_args) -> new_iter_args
            result_types: MLIR types for results (e.g., ["f64", "!Point"])

        Returns:
            SSAValue(s) for loop result(s)
        """
        # TODO: Implement
        pass

    def emit_yield(self, values: List[SSAValue], types: List[str]):
        """Emit scf.yield statement."""
        # TODO: Implement
        pass
```

### Compiler Scaffold

```python
class MLIRCompiler(ASTVisitor):
    # ... existing code ...

    def visit_FlowExpr(self, node: FlowExpr) -> SSAValue:
        """
        Compile flow block to scf.for loop.

        Handles three forms:
        1. flow state over duration by dt with fn
        2. flow state for steps with fn
        3. flow state over duration by dt substeps N with fn
        """
        # Step 1: Compile initial state
        state = self.visit(node.initial_state)

        # Step 2: Determine iteration count
        if node.duration and node.dt:
            # Time-based
            # TODO: Compute num_steps = duration / dt
            pass
        elif node.steps:
            # Step-based
            # TODO: Use steps directly
            pass
        else:
            raise CompileError("Invalid flow block")

        # Step 3: Create loop bounds (index type)
        # TODO: Convert to index type

        # Step 4: Handle substeps if present
        if node.substeps:
            # TODO: Nested loop
            pass

        # Step 5: Define loop body
        def body_fn(loop_var, iter_args):
            # TODO: Call update function
            # TODO: Return new state
            pass

        # Step 6: Create the loop
        # TODO: Call builder.create_for_loop()

        return result
```

---

## Reference Materials

### Morphogen Repository Structure

```
morphogen/
├── morphogen/
│   ├── lexer.py          # Tokenization (complete)
│   ├── parser.py         # AST generation (complete)
│   ├── ast_nodes.py      # AST definitions (complete)
│   ├── type_checker.py   # Type inference (complete)
│   ├── runtime.py        # Python interpreter (complete, reference for semantics)
│   ├── ir_builder.py     # MLIR IR construction (Phases 1-2 complete)
│   ├── mlir_compiler.py  # AST → MLIR (Phases 1-2 complete, you're adding Phase 3)
│   └── ...
├── tests/
│   ├── test_parser.py           # Parser tests (19 tests)
│   ├── test_runtime.py          # Runtime tests (41 tests)
│   ├── test_mlir_compiler.py    # MLIR tests (21 tests, you're adding ~10 more)
│   └── ...
├── examples/
│   ├── spring.kairo      # Spring oscillator
│   ├── projectile.kairo  # Projectile motion
│   ├── orbit.kairo       # Orbital mechanics
│   └── ...
└── README.md
```

### Running Tests

```bash
# All tests
pytest

# Just MLIR tests
pytest tests/test_mlir_compiler.py -v

# Specific test
pytest tests/test_mlir_compiler.py::TestFlowBlocks::test_simple_flow_scalar -v

# With output
pytest tests/test_mlir_compiler.py -v -s
```

### Checking Test Coverage

```bash
# Run with coverage
pytest --cov=kairo --cov-report=html

# View report
open htmlcov/index.html
```

---

## Success Metrics

When you submit your PR, it should demonstrate:

### Quantitative Metrics
- **Tests**: 191+ passing (181 existing + 10 new)
- **Code coverage**: >85% for new code
- **Lines added**: ~400-500 (focused, no bloat)
- **Regressions**: Zero
- **Examples compiling**: At least 1 physics example

### Qualitative Metrics
- **Code quality**: Clean, maintainable, follows existing patterns
- **MLIR correctness**: Valid syntax, proper SSA form
- **Documentation**: Clear comments, helpful test names
- **Completeness**: All flow block forms supported

### Comparison to Previous PRs
Your Phase 3 PR should match the quality of:
- **PR #7** (Runtime): 10/10 quality, comprehensive
- **PR #8** (Structs): 10/10 quality, complete
- **PR #9** (MLIR P1-2): 10/10 quality, solid foundation

Aim for: **10/10 - Exceptional**

---

## Questions & Clarifications

If you encounter ambiguity:

### Q: Should flow blocks be expressions or statements?
**A**: Expressions. Flow blocks return a value (the final state).

### Q: What if dt doesn't divide evenly into duration?
**A**: Truncate (integer division). Match Python runtime behavior.

### Q: Should we optimize loop bounds at compile time?
**A**: Not yet. Constants are fine, but don't do constant folding. That's Phase 5.

### Q: What about infinite loops or runtime errors?
**A**: Assume valid input for now. Type checker ensures correctness.

### Q: Do we need actual MLIR execution?
**A**: No. Just generate syntactically correct MLIR text. Execution is future work.

### Q: How should we handle large nested structs?
**A**: SSA values and `iter_args` handle it automatically. Trust the type system.

---

## Final Notes

### This is the Hardest Phase

Phase 3 is the most complex part of the MLIR pipeline:
- Control flow is trickier than expressions
- State threading requires careful SSA management
- Substeps add nested complexity

**Don't rush.** Take time to understand `scf.for` and `iter_args`. Study the examples. Test incrementally.

### You Have a Strong Foundation

Phases 1-2 gave you:
- ✅ Type system working
- ✅ Function calls working
- ✅ Struct operations working
- ✅ IR builder infrastructure

You're building on solid ground.

### Quality Over Speed

The pattern from PR #7, #8, #9: **detailed work produces excellent results**.

Take 2-3 days. Test thoroughly. Write clean code. Document decisions.

A 10/10 PR that takes 3 days is better than a 7/10 PR that takes 1 day.

### When You're Done

Phase 3 completion means:
- **60% of MLIR pipeline complete** (was 40%)
- **All v0.3.1 examples can compile** (massive milestone)
- **Temporal execution proven** (core Morphogen innovation validated)
- **Phases 4-5 are easy** (lambdas + polish)

You're delivering the heart of Morphogen's compilation story.

---

## Commit Message Template

When you create your PR:

```
MLIR Phase 3: Temporal Execution (Flow Blocks)

Implements compilation of Morphogen flow blocks to MLIR scf.for loops,
enabling temporal iteration with proper state management.

Features:
- Time-based flow: `flow state over duration by dt with fn`
- Step-based flow: `flow state for steps with fn`
- Substeps: `flow state over duration by dt substeps N with fn`
- SSA value threading via iter_args
- Nested loop support for substeps

Implementation:
- ir_builder.py: Added create_for_loop() for scf.for generation
- mlir_compiler.py: Added visit_FlowExpr() for flow compilation
- Loop bounds computation (duration/dt → integer steps)
- Proper type management for struct state

Testing:
- 10 new comprehensive tests for flow blocks
- Tests cover: time-based, step-based, substeps, scalars, structs
- Physics simulation example (spring oscillator) compiles
- All 191 tests passing (181 existing + 10 new)
- Zero regressions

Examples:
- Spring oscillator: Compiles and generates correct MLIR
- Projectile motion: Compiles successfully
- Multi-particle system: Handles nested struct state

Quality: 10/10
- Clean SSA form generation
- Proper iter_args usage
- Comprehensive test coverage
- Professional code quality

Progress: MLIR compilation now 60% complete (Phases 1-3 done)
Next: Phase 4 (lambdas) and Phase 5 (optimization)
```

---

## Get Started

1. **Clone/pull latest**: `git checkout master && git pull`
2. **Create branch**: `git checkout -b feature/mlir-phase3-flow-blocks`
3. **Read existing code**: Study `ir_builder.py` and `mlir_compiler.py` (Phases 1-2)
4. **Start with tests**: Write first test, make it pass
5. **Build incrementally**: One feature at a time
6. **Test constantly**: After every change
7. **Commit frequently**: Small, focused commits
8. **Aim for 10/10**: Quality is non-negotiable

---

## You've Got This

You have:
- ✅ Clear specification
- ✅ Working examples
- ✅ Strong foundation (Phases 1-2)
- ✅ Comprehensive test strategy
- ✅ Reference semantics (Python runtime)
- ✅ Proven workflow (3 consecutive 10/10 PRs)

Go build the heart of Morphogen's temporal execution system. Make it exceptional.

**Target: 10/10 quality, ~400-500 lines, 10+ tests, 2-3 days.**

**Let's make Phase 3 another perfect PR.** 🚀
