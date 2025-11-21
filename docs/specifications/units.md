# Physical Units and Dimensional Analysis

## Overview

Morphogen implements a comprehensive physical unit system with dimensional analysis to enable safer cross-domain composition. The system automatically checks that operations respect physical dimensions, preventing errors like adding meters to seconds.

## Features

- **Seven SI Base Dimensions**: Mass, length, time, current, temperature, amount, luminosity
- **Dimensional Analysis**: Automatic tracking of units through arithmetic operations
- **Unit Algebra**: Multiplication, division, and exponentiation of units
- **Unit Compatibility Checking**: Verifies dimensional consistency at type-check time
- **Unit Conversion**: Convert between compatible units (e.g., meters to kilometers)
- **Cross-Domain Safety**: Ensures physical consistency when data flows between domains

## Basic Usage

### Declaring Types with Units

Units are specified in square brackets after type declarations:

```morphogen
// Scalar with units
let temperature: f32 [K] = 273.15;
let distance: f32 [m] = 10.0;
let time: f32 [s] = 2.0;

// Vector with units
let velocity: Vec2<f32> [m/s] = Vec2(5.0, 0.0);
let force: Vec2<f32> [N] = Vec2(10.0, 0.0);

// Field with units
let temp_field: Field2D<f32> [K] = field2d(128, 128, 1.0);
let pressure_field: Field2D<f32> [Pa] = field2d(128, 128, 1.0);

// Signal with units
let audio: Signal<f32> [Pa] = signal_from_mic();
```

### Unit Expressions

Units support algebraic expressions:

```morphogen
// Simple units
[m]           // meters
[kg]          // kilograms
[s]           // seconds
[K]           // kelvin
[A]           // amperes

// Derived units via multiplication
[kg*m]        // kilogram-meters

// Derived units via division
[m/s]         // velocity (meters per second)
[kg/m^3]      // density (kilograms per cubic meter)

// Complex expressions
[kg*m/s^2]    // force (Newtons)
[kg*m^2/s^2]  // energy (Joules)
[kg*m^2/s^3]  // power (Watts)

// Named derived units
[N]           // Newton (force)
[J]           // Joule (energy)
[W]           // Watt (power)
[Pa]          // Pascal (pressure)
[Hz]          // Hertz (frequency)
[V]           // Volt (voltage)
[C]           // Coulomb (charge)
[Ω]           // Ohm (resistance)
```

## Dimensional Analysis

### Arithmetic Operations

The type checker automatically infers resulting units from operations:

#### Addition/Subtraction
Units must match:
```morphogen
let d1: f32 [m] = 10.0;
let d2: f32 [m] = 5.0;
let d3 = d1 + d2;  // Result: f32 [m]

// ERROR: Cannot add incompatible units
let bad = d1 + time;  // Type error: [m] and [s] are incompatible
```

#### Multiplication
Units multiply:
```morphogen
let length: f32 [m] = 10.0;
let width: f32 [m] = 5.0;
let area = length * width;  // Result: f32 [m*m] or [m^2]

let mass: f32 [kg] = 2.0;
let accel: f32 [m/s^2] = 9.8;
let force = mass * accel;  // Result: f32 [kg*m/s^2] (Newton)
```

#### Division
Units divide:
```morphogen
let distance: f32 [m] = 100.0;
let time: f32 [s] = 10.0;
let velocity = distance / time;  // Result: f32 [m/s]

let energy: f32 [J] = 1000.0;
let power = energy / time;  // Result: f32 [J/s] or [W]
```

#### Exponentiation
Exponent must be dimensionless:
```morphogen
let radius: f32 [m] = 5.0;
let area = radius ^ 2;  // Result: f32 [m^2]

let volume = radius ^ 3;  // Result: f32 [m^3]

// ERROR: Exponent must be dimensionless
let bad = 2.0 ^ radius;  // Type error
```

## Unit Compatibility

### Compatible Units

Units with the same dimensions are compatible, even if scaled differently:

```morphogen
let d1: f32 [m] = 1000.0;
let d2: f32 [km] = 1.0;     // Compatible: both are length

let t1: f32 [s] = 1.0;
let t2: f32 [ms] = 1000.0;  // Compatible: both are time

// Derived units are compatible if dimensionally equivalent
let f1: f32 [kg*m/s^2] = 10.0;
let f2: f32 [N] = 10.0;     // Compatible: N ≡ kg·m/s²
```

### Incompatible Units

Units with different dimensions are not compatible:

```morphogen
let distance: f32 [m] = 10.0;
let time: f32 [s] = 5.0;

// ERROR: Cannot assign incompatible units
let bad: f32 [m] = time;  // Type error: [s] is not compatible with [m]
```

### Dimensionless Compatibility

`None` (no unit annotation) is compatible with any unit:

```morphogen
let temp_with_unit: f32 [K] = 273.15;
let temp_no_unit: f32 = 273.15;

// Both directions work
let a: f32 [K] = temp_no_unit;  // OK
let b: f32 = temp_with_unit;    // OK
```

## Unit Conversion

The runtime supports conversion between compatible units:

```python
from morphogen.types.units import Unit

# Create units
meter = Unit.meter()
kilometer = Unit.kilometer()
centimeter = Unit.centimeter()

# Convert values
value_km = kilometer.convert_to(meter, 1.0)  # 1 km → 1000 m
value_cm = meter.convert_to(centimeter, 1.0)  # 1 m → 100 cm

# Derived units
newton = Unit.newton()
force_manual = parse_unit("kg*m/s^2")
force_manual.is_compatible_with(newton)  # True
```

## Cross-Domain Unit Safety

Units provide critical safety guarantees when composing different computational domains:

### Field-Agent Interaction

```morphogen
// Temperature field in Kelvin
let temp_field: Field2D<f32> [K] = field2d(128, 128, 1.0);

// Agents with temperature property
struct Particle {
    pos: Vec2<f32> [m],
    temp: f32 [K],    // Must match field units
}

// Safe: Units are compatible
let particles = sample_field(agents, temp_field, "temp");

// ERROR: Would fail if units don't match
struct BadParticle {
    pos: Vec2<f32> [m],
    temp: f32 [Pa],   // Pressure, not temperature!
}
// Type checker catches the mismatch
```

### Physics-Audio Sonification

```morphogen
// Physical force in Newtons
let force: f32 [N] = calculate_collision_force();

// Convert to audio amplitude (dimensionless or Pa)
let amplitude: f32 [Pa] = scale_force_to_pressure(force);

// Create audio signal
let audio: Signal<f32> [Pa] = sonify(amplitude);
```

### Spatial-Temporal Consistency

```morphogen
// Position and velocity must be consistent
let pos: Vec2<f32> [m] = Vec2(0.0, 0.0);
let vel: Vec2<f32> [m/s] = Vec2(1.0, 0.0);
let dt: f32 [s] = 0.016;

// Units are automatically checked
let new_pos = pos + vel * dt;  // OK: [m] + [m/s]*[s] = [m] + [m]
```

## SI Base Units

The system implements all seven SI base dimensions:

| Dimension | Symbol | Unit Name | Unit Symbol | Quantity |
|-----------|--------|-----------|-------------|----------|
| M | Mass | kilogram | kg | mass |
| L | Length | meter | m | length |
| T | Time | second | s | time |
| I | Current | ampere | A | electric current |
| Θ | Temperature | kelvin | K | thermodynamic temperature |
| N | Amount | mole | mol | amount of substance |
| J | Luminosity | candela | cd | luminous intensity |

## Common Derived Units

### Mechanical

| Quantity | Unit | Symbol | Dimension Formula |
|----------|------|--------|-------------------|
| Force | newton | N | M·L·T⁻² |
| Energy | joule | J | M·L²·T⁻² |
| Power | watt | W | M·L²·T⁻³ |
| Pressure | pascal | Pa | M·L⁻¹·T⁻² |
| Frequency | hertz | Hz | T⁻¹ |

### Electromagnetic

| Quantity | Unit | Symbol | Dimension Formula |
|----------|------|--------|-------------------|
| Voltage | volt | V | M·L²·T⁻³·I⁻¹ |
| Charge | coulomb | C | I·T |
| Resistance | ohm | Ω | M·L²·T⁻³·I⁻² |

## Implementation Details

### Dimensional Formula Representation

Each unit has a dimensional formula represented as powers of base dimensions:

```python
# Force: M¹·L¹·T⁻²
Dimensions(mass=1, length=1, time=-2)

# Energy: M¹·L²·T⁻²
Dimensions(mass=1, length=2, time=-2)

# Velocity: L¹·T⁻¹
Dimensions(length=1, time=-1)
```

### Unit Algebra

Units support algebraic operations:

```python
from morphogen.types.units import Unit

meter = Unit.meter()
second = Unit.second()

# Division: m / s = m·s⁻¹
velocity = meter / second

# Multiplication: m * s = m·s
meter_second = meter * second

# Exponentiation: m² = m·m
area = meter ** 2
```

### Parsing Unit Expressions

The parser converts unit strings to Unit objects:

```python
from morphogen.types.units import parse_unit

# Simple unit
meter = parse_unit("m")

# Derived unit
velocity = parse_unit("m/s")

# Complex expression
force = parse_unit("kg*m/s^2")

# Check compatibility
assert force.is_compatible_with(Unit.newton())
```

### Type System Integration

The type system uses dimensional analysis for compatibility:

```python
from morphogen.ast.types import ScalarType, BaseType

# Create types with units
temp_k = ScalarType(BaseType.F32, "K")
temp_c = ScalarType(BaseType.F32, "K")  # Both Kelvin
length = ScalarType(BaseType.F32, "m")

# Check compatibility (uses dimensional analysis)
assert temp_k.is_compatible_with(temp_c)  # True
assert not temp_k.is_compatible_with(length)  # False
```

## Error Messages

The type checker provides helpful error messages for unit mismatches:

```
Type mismatch in assignment to 'velocity':
  cannot assign ScalarType(f32) with unit [m]
  to ScalarType(f32) with unit [m/s]

Unit mismatch in + operation:
  [m] and [s] are not compatible

Invalid unit expression 'xyz':
  Unknown unit 'xyz'

Exponent must be dimensionless, got [m]
```

## Best Practices

### 1. Always Specify Units for Physical Quantities

```morphogen
// Good
let temperature: f32 [K] = 273.15;
let pressure: f32 [Pa] = 101325.0;

// Avoid (unless truly dimensionless)
let temperature: f32 = 273.15;
```

### 2. Use Standard SI Units

```morphogen
// Preferred
let distance: f32 [m] = 1000.0;

// Also acceptable (but be consistent)
let distance: f32 [km] = 1.0;
```

### 3. Let the Type Checker Infer Derived Units

```morphogen
let velocity: f32 [m/s] = distance / time;  // Explicit
let velocity = distance / time;  // Inferred (if distance and time have units)
```

### 4. Use Named Derived Units When Clear

```morphogen
// Clear intent
let force: f32 [N] = mass * acceleration;

// Also valid but less clear
let force: f32 [kg*m/s^2] = mass * acceleration;
```

### 5. Document Unit Assumptions in Interfaces

```morphogen
/// Computes gravitational force
/// @param mass1 First mass [kg]
/// @param mass2 Second mass [kg]
/// @param distance Distance between centers [m]
/// @returns Gravitational force [N]
fn gravitational_force(
    mass1: f32 [kg],
    mass2: f32 [kg],
    distance: f32 [m]
) -> f32 [N] {
    // Implementation
}
```

## Future Extensions

### Potential Future Features

1. **Temperature Unit Conversion**: Handle Celsius/Fahrenheit conversions (affine transforms)
2. **Angle Units**: Support radians, degrees, gradians
3. **Custom Unit Definitions**: Allow users to define domain-specific units
4. **Unit Inference**: Infer units from operations without explicit annotations
5. **Quantity Types**: Combine magnitude and units in a single type
6. **Unit Simplification**: Automatically simplify complex unit expressions

## References

- [International System of Units (SI)](https://www.bipm.org/en/measurement-units)
- [Dimensional Analysis](https://en.wikipedia.org/wiki/Dimensional_analysis)
- [F# Units of Measure](https://docs.microsoft.com/en-us/dotnet/fsharp/language-reference/units-of-measure)
- Morphogen Specification (SPECIFICATION.md), Section 4.7

## Module Documentation

### Main Modules

- `morphogen/types/units.py`: Unit system implementation with dimensional analysis
- `morphogen/ast/types.py`: Type system integration with unit compatibility
- `morphogen/ast/visitors.py`: Type checker with unit validation

### API Reference

See inline documentation in:
- `/home/user/morphogen/morphogen/types/units.py:1` - Core unit system classes
- `/home/user/morphogen/morphogen/ast/types.py:7` - Type system integration
- `/home/user/morphogen/morphogen/ast/visitors.py:96` - Type checker enhancements

### Test Coverage

Comprehensive tests in:
- `/home/user/morphogen/tests/test_units.py:1` - Unit system tests covering:
  - Dimensional analysis
  - Unit algebra
  - Parsing
  - Compatibility checking
  - Conversion
  - Cross-domain scenarios
  - Edge cases
