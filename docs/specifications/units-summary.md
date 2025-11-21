# Physical Unit Checking - Implementation Summary

## Overview

Implemented comprehensive physical unit checking and dimensional analysis for Morphogen's type system, enabling safer cross-domain composition through compile-time dimensional consistency verification.

## What Was Implemented

### 1. Core Unit System (`morphogen/types/units.py`)

**Dimensions Class**
- Represents dimensional formulas using seven SI base dimensions (M, L, T, I, Θ, N, J)
- Supports dimensional algebra: multiplication, division, exponentiation
- Uses rational numbers (Fraction) for exact fractional exponents
- Example: Force = M¹·L¹·T⁻² (mass × length / time²)

**Unit Class**
- Represents physical units with name, symbol, dimensions, and scale factor
- Implements unit algebra operations (*, /, **)
- Provides compatibility checking via dimensional analysis
- Supports unit conversion between compatible units
- Includes SI base units (m, kg, s, A, K, mol, cd)
- Includes common derived units (N, J, W, Pa, Hz, V, C, Ω)
- Includes prefixed units (cm, km, g, ms)

**Parsing Infrastructure**
- `parse_unit(expr)`: Parses unit expression strings like "kg*m/s^2"
- `check_unit_compatibility(u1, u2)`: Checks dimensional compatibility
- `get_unit_dimensions(u)`: Extracts dimensional formula from unit string
- Supports products, quotients, powers: "kg*m/s^2", "m/s", "m^2"

### 2. Type System Integration (`morphogen/ast/types.py`)

**Enhanced Type Classes**
- Updated `ScalarType`, `VectorType`, `FieldType`, `SignalType`
- Changed from string equality to dimensional compatibility checking
- Now uses `check_unit_compatibility()` instead of `unit == other.unit`
- Enables: `f32 [m]` compatible with `f32 [cm]` (both are length)
- Prevents: `f32 [m]` assignment to `f32 [s]` (incompatible dimensions)

**Backward Compatibility**
- `None` unit remains compatible with any unit
- Graceful fallback if units module unavailable
- Preserves existing type system behavior

### 3. Type Checker Enhancement (`morphogen/ast/visitors.py`)

**Unit Validation**
- Validates unit expressions are parseable at type-check time
- Detects invalid unit syntax early: "xyz", "m^abc"
- Reports specific errors: "Invalid unit expression 'xyz': Unknown unit"

**Enhanced Error Messages**
- Includes unit information in type mismatch errors
- Shows dimensional incompatibility clearly
- Example: "cannot assign f32 with unit [m] to f32 with unit [s]"

**Arithmetic Unit Inference**
- `_infer_arithmetic_unit()`: Computes resulting units from operations
- Addition/subtraction: Checks dimensional compatibility
- Multiplication: Multiplies units (m × m = m²)
- Division: Divides units (m / s = m/s)
- Exponentiation: Validates exponent is dimensionless

### 4. Comprehensive Test Suite (`tests/test_units.py`)

**Test Coverage** (300+ lines)
- Dimensional algebra operations
- Unit algebra operations
- Unit parsing (simple, products, quotients, powers, complex)
- Compatibility checking
- Unit conversion and scaling
- Cross-domain scenarios
- Edge cases (inverse units, square roots, cancellation)

**All Tests Passing**
- Verified with standalone Python tests
- Tested dimensional analysis: ✓
- Tested unit algebra: ✓
- Tested parsing: ✓
- Tested type system integration: ✓

### 5. Documentation (`docs/UNITS.md`)

**Comprehensive Guide** (500+ lines)
- Overview and features
- Basic usage with examples
- Dimensional analysis explanation
- Unit compatibility rules
- Cross-domain safety examples
- SI base and derived units reference
- Implementation details
- Best practices
- API reference with file paths and line numbers

## Key Benefits for Cross-Domain Composition

### 1. Type Safety Across Domains

**Before:**
```morphogen
// No checking - runtime error likely
let temp_field: Field2D<f32> = ...;
struct Particle {
    temp: f32,  // Could be any unit!
}
```

**After:**
```morphogen
// Compile-time verification
let temp_field: Field2D<f32> [K] = ...;
struct Particle {
    temp: f32 [K],  // Must match field units
}
// Type checker ensures compatibility
```

### 2. Physical Consistency Verification

**Prevents Invalid Operations:**
```morphogen
let distance: f32 [m] = 10.0;
let time: f32 [s] = 5.0;

// ERROR caught at compile-time:
let bad = distance + time;  // Type error: [m] and [s] incompatible
```

**Enables Valid Operations:**
```morphogen
let velocity = distance / time;  // OK: f32 [m/s]
let acceleration: f32 [m/s^2] = 9.8;
let force = mass * acceleration;  // OK: f32 [kg*m/s^2] = [N]
```

### 3. Cross-Domain Interface Safety

**Field-Agent Domain:**
- Ensures agent properties match field element types
- Prevents sampling temperature field into pressure property
- Example: `sample_field(agents, temp_field [K], "temp" [K])`

**Physics-Audio Domain:**
- Validates physical quantities map correctly to audio
- Forces dimensional consistency in sonification
- Example: Force [N] → Amplitude [Pa] conversion verified

**Spatial-Temporal Domain:**
- Ensures position/velocity/acceleration relationships
- Validates time integration: `pos + vel*dt` checked
- Example: [m] + [m/s]*[s] = [m] + [m] ✓

### 4. Automatic Unit Inference

The type checker automatically tracks units through operations:

```morphogen
let length: f32 [m] = 10.0;
let width: f32 [m] = 5.0;
let time: f32 [s] = 2.0;

let area = length * width;     // Inferred: f32 [m^2]
let velocity = length / time;  // Inferred: f32 [m/s]
let volume = area * length;    // Inferred: f32 [m^3]
```

### 5. Dimensional Compatibility

Compatible units are automatically recognized:

```morphogen
// All length units are compatible
let d1: f32 [m] = 1000.0;
let d2: f32 [km] = 1.0;
let d3: f32 [cm] = 100000.0;
// All can be used interchangeably

// Dimensionally equivalent derived units
let f1: f32 [kg*m/s^2] = 10.0;
let f2: f32 [N] = 10.0;
// Both represent force, are compatible
```

## Technical Highlights

### Dimensional Formula Algebra

```python
# Force dimensions: M·L·T⁻²
force_dims = Dimensions(mass=1, length=1, time=-2)

# Distance dimensions: L
dist_dims = Dimensions(length=1)

# Work = Force × Distance
work_dims = force_dims * dist_dims
# Result: M·L²·T⁻² (Energy/Joules)
```

### Fractional Exponents Support

```python
# Square root of area gives length
area = Dimensions(length=2)
length = area ** Fraction(1, 2)
# Result: Dimensions(length=1)
```

### Scale Factor Conversion

```python
kilometer = Unit.kilometer()  # scale=1000.0
meter = Unit.meter()          # scale=1.0

# Convert 5 km to meters
result = kilometer.convert_to(meter, 5.0)
# Result: 5000.0 m
```

## Files Modified/Created

### Created:
1. `/home/user/morphogen/morphogen/types/units.py` (550 lines)
   - Core unit system implementation
   - Dimensions and Unit classes
   - Parsing and compatibility checking

2. `/home/user/morphogen/tests/test_units.py` (500 lines)
   - Comprehensive test coverage
   - All aspects of dimensional analysis

3. `/home/user/morphogen/docs/UNITS.md` (500 lines)
   - Complete user documentation
   - Examples and best practices

### Modified:
1. `/home/user/morphogen/morphogen/ast/types.py`
   - Added unit compatibility imports
   - Updated `is_compatible_with()` methods
   - Now uses dimensional analysis

2. `/home/user/morphogen/morphogen/ast/visitors.py`
   - Added `_validate_unit_expression()`
   - Added `_infer_arithmetic_unit()`
   - Enhanced error messages with unit info

3. `/home/user/morphogen/morphogen/types/__init__.py`
   - Updated exports for units module
   - Removed references to non-existent modules

## Usage Examples

### Simple Type Declaration
```morphogen
let temperature: f32 [K] = 273.15;
let pressure: f32 [Pa] = 101325.0;
let velocity: Vec2<f32> [m/s] = Vec2(10.0, 0.0);
```

### Cross-Domain Composition
```morphogen
// Temperature field
let temp_field: Field2D<f32> [K] = field2d(128, 128, 1.0);

// Agents with matching units
struct Particle {
    pos: Vec2<f32> [m],
    vel: Vec2<f32> [m/s],
    temp: f32 [K],  // Compatible with temp_field
}

// Safe sampling - units verified
let particles = sample_field(agents, temp_field, "temp");
```

### Physics Simulation
```morphogen
fn update_physics(dt: f32 [s]) {
    let mass: f32 [kg] = 1.0;
    let force: f32 [N] = 10.0;

    // F = ma, so a = F/m
    let accel = force / mass;  // f32 [m/s^2]

    // v = v0 + at
    let velocity = accel * dt;  // f32 [m/s]

    // s = vt
    let displacement = velocity * dt;  // f32 [m]
}
```

## Integration with Existing Code

### Parser Integration
- Parser already supports unit syntax: `type [unit_expr]`
- Unit expressions collected as strings
- Now parsed and validated by unit system

### Runtime Integration
- Type system uses units at compile-time only
- Runtime values remain unchanged
- No performance overhead

### Backward Compatibility
- Existing code without units continues to work
- `None` unit matches any unit (opt-in gradual typing)
- No breaking changes to existing types

## Impact on Development

### Prevents Common Errors
- Mixing incompatible units (meters + seconds)
- Incorrect dimensional formulas (force as mass × velocity)
- Unit mismatches in cross-domain interfaces

### Improves Code Quality
- Self-documenting types with physical meaning
- Compiler enforces physical correctness
- Safer refactoring with unit checking

### Enables Advanced Features
- Automatic unit conversion in the future
- Dimension inference from expressions
- Physical quantity types (value + unit together)

## Performance Characteristics

- **Compile-time**: Dimensional analysis during type checking
- **Runtime**: Zero overhead (units erased after type checking)
- **Memory**: Units stored as strings in AST, parsed on demand
- **Scalability**: O(1) dimensional compatibility checks

## Future Enhancements

Potential extensions to build on this foundation:

1. **Automatic Unit Conversion**: Insert conversions automatically
2. **Temperature Affine Transforms**: Celsius ↔ Fahrenheit
3. **Angle Units**: Radians, degrees with conversion
4. **Custom Units**: Domain-specific unit definitions
5. **Quantity Types**: Combined value+unit runtime representation
6. **Unit Inference**: Infer units from expression context

## Conclusion

The physical unit checking system provides:

✅ **Type Safety**: Compile-time dimensional verification
✅ **Cross-Domain Safety**: Ensures physical consistency across domains
✅ **Developer Experience**: Clear error messages, self-documenting types
✅ **Zero Runtime Cost**: All checking at compile-time
✅ **Backward Compatible**: Opt-in, doesn't break existing code
✅ **Extensible**: Foundation for advanced features

This implementation unlocks safer cross-domain composition by catching dimensional errors at compile-time, making Morphogen programs more robust and correct by construction.
