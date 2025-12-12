# SPEC: Morphogen Graph IR

**Version:** 1.0 Draft
**Status:** RFC
**Last Updated:** 2025-11-11

---

## Overview

**Morphogen Graph IR** is the canonical intermediate representation between frontends (Morphogen.Audio, RiffStack) and the Morphogen Kernel. It's a typed, JSON-based directed acyclic graph (DAG) that captures:

- Nodes (operators with parameters)
- Edges (data dependencies)
- Type annotations (Stream types, units, domains, rates)
- Profile hints
- Event streams

**Design Goals:**
1. **Frontend-agnostic** — Both Morphogen.Audio DSL and RiffStack YAML emit the same IR
2. **Human-readable** — JSON format, inspectable, debuggable
3. **Type-safe** — Full type/unit/domain annotations for validation
4. **Deterministic** — Explicit ordering, seeding, profile constraints
5. **Extensible** — Forward-compatible versioning, custom attributes

---

## Schema (JSON)

### Top-Level Structure

```json
{
  "version": "1.0",
  "profile": "repro",
  "seed": 42,
  "sample_rate": 48000,
  "nodes": [ /* Node objects */ ],
  "edges": [ /* Edge objects */ ],
  "outputs": { /* Output mapping */ },
  "events": [ /* Event stream definitions */ ],
  "metadata": { /* Optional metadata */ }
}
```

---

### Node Object

```json
{
  "id": "osc1",
  "op": "sine",
  "params": {
    "freq": "440Hz",
    "phase": 0.0
  },
  "rate": "audio",
  "outputs": [
    {"name": "out", "type": "Sig"}
  ],
  "profile_overrides": {}
}
```

**Fields:**
- `id` (string, required): Unique node identifier
- `op` (string, required): Operator name (from registry)
- `params` (object, optional): Parameter values with units
- `rate` (string, optional): Execution rate (`audio`, `control`, `visual`, `sim`)
- `outputs` (array, required): Output port definitions
- `profile_overrides` (object, optional): Per-node profile settings

---

### Edge Object

```json
{
  "from": "osc1:out",
  "to": "lpf1:in",
  "type": "Sig"
}
```

**Fields:**
- `from` (string, required): Source node:port
- `to` (string, required): Destination node:port
- `type` (string, required): Data type annotation

**Validation:**
- Source and destination must exist
- Types must be compatible (or explicit cast required)
- No cycles allowed (DAG constraint)

---

### Output Mapping

```json
{
  "outputs": {
    "stereo": ["pan1:left", "pan1:right"],
    "spectrum": ["fft1:out"]
  }
}
```

Maps logical output names to node:port references.

---

### Event Stream Definition

```json
{
  "events": [
    {
      "id": "note_seq",
      "type": "Evt<Note>",
      "data": [
        {"time": 0.0, "value": {"pitch": "440Hz", "vel": 0.8, "dur": "0.5s"}},
        {"time": 0.5, "value": {"pitch": "554Hz", "vel": 0.6, "dur": "0.5s"}}
      ]
    }
  ]
}
```

**Fields:**
- `id` (string, required): Event stream identifier
- `type` (string, required): Event payload type
- `data` (array, required): Timestamped events (must be sorted by time)

**Validation:**
- Times must be monotonically increasing
- Payload types must match declared type
- Deterministic replay guaranteed

---

## Complete Example: Simple Synth

```json
{
  "version": "1.0",
  "profile": "repro",
  "seed": 1337,
  "sample_rate": 48000,
  "nodes": [
    {
      "id": "osc1",
      "op": "sine",
      "params": {"freq": "440Hz"},
      "rate": "audio",
      "outputs": [{"name": "out", "type": "Sig"}]
    },
    {
      "id": "lpf1",
      "op": "lpf",
      "params": {"cutoff": "2kHz", "q": 0.707},
      "rate": "audio",
      "outputs": [{"name": "out", "type": "Sig"}]
    },
    {
      "id": "env1",
      "op": "adsr",
      "params": {
        "attack": "0.01s",
        "decay": "0.1s",
        "sustain": 0.7,
        "release": "0.3s"
      },
      "rate": "control",
      "outputs": [{"name": "out", "type": "Ctl"}]
    },
    {
      "id": "mul1",
      "op": "multiply",
      "rate": "audio",
      "outputs": [{"name": "out", "type": "Sig"}]
    },
    {
      "id": "pan1",
      "op": "pan",
      "params": {"pos": 0.0},
      "rate": "audio",
      "outputs": [
        {"name": "left", "type": "Sig"},
        {"name": "right", "type": "Sig"}
      ]
    }
  ],
  "edges": [
    {"from": "osc1:out", "to": "lpf1:in", "type": "Sig"},
    {"from": "lpf1:out", "to": "mul1:in1", "type": "Sig"},
    {"from": "env1:out", "to": "mul1:in2", "type": "Ctl"},
    {"from": "mul1:out", "to": "pan1:in", "type": "Sig"}
  ],
  "outputs": {
    "stereo": ["pan1:left", "pan1:right"]
  },
  "metadata": {
    "source": "Morphogen.Audio",
    "scene_name": "SimpleSynth",
    "author": "user@example.com"
  }
}
```

---

## Type System

### Stream Types

| Type | Full Form | Domain | Rate | Description |
|------|-----------|--------|------|-------------|
| `Sig` | `Stream<f32,1D,audio>` | 1D | audio | Audio signal |
| `Ctl` | `Stream<f32,0D,control>` | 0D | control | Control signal |
| `Field2D<T>` | `Stream<T,2D,sim>` | 2D | sim | 2D field |
| `Field3D<T>` | `Stream<T,3D,sim>` | 3D | sim | 3D field |
| `Image` | `Stream<RGB,2D,visual>` | 2D | visual | Image/frame |

### Event Types

| Type | Description |
|------|-------------|
| `Evt<Note>` | Musical note events |
| `Evt<Control>` | Control change events |
| `Evt<Trigger>` | Trigger/bang events |
| `Evt<T>` | Generic timestamped events |

### Unit Annotations

Parameters must include unit annotations:

```json
{
  "freq": "440Hz",       // Frequency in Hertz
  "cutoff": "2kHz",      // Kilohertz
  "time": "0.5s",        // Seconds
  "gain": "-6dB",        // Decibels
  "phase": "0.25rad",    // Radians
  "position": "0.5"      // Normalized (unitless)
}
```

**Supported units:**
- Frequency: `Hz`, `kHz`, `MHz`, `cents`, `midi`
- Time: `s`, `ms`, `us`, `samples`, `bars`, `beats`
- Amplitude: `dB`, `linear`, `ratio`
- Angle: `rad`, `deg`, `turns`
- Spatial: `px`, `m`, `cm`, `mm`

---

## Validation Rules

### 1. Type Checking

```python
def validate_edge(edge, nodes):
    from_node, from_port = parse_ref(edge["from"])
    to_node, to_port = parse_ref(edge["to"])

    from_type = get_output_type(nodes[from_node], from_port)
    to_type = get_input_type(nodes[to_node], to_port)

    if not types_compatible(from_type, to_type):
        raise TypeError(f"Type mismatch: {from_type} → {to_type}")
```

### 2. Rate Compatibility

Cross-rate connections require explicit resampling:

```json
// ERROR: Direct audio → control connection
{"from": "osc1:out", "to": "slow_lfo:in", "type": "Sig"}

// CORRECT: Insert resample node
{
  "id": "resample1",
  "op": "resample",
  "params": {"to_rate": "control", "mode": "linear"}
}
```

### 3. DAG Constraint

```python
def validate_dag(nodes, edges):
    # Topological sort
    # Detect cycles
    if has_cycle(edges):
        raise ValueError("Graph contains cycle (not a DAG)")
```

### 4. Unit Consistency

```python
def validate_units(param_value, expected_unit):
    value, unit = parse_unit(param_value)
    if not compatible_units(unit, expected_unit):
        raise ValueError(f"Unit mismatch: {unit} != {expected_unit}")
```

---

## Profile System Integration

### Global Profile

```json
{
  "profile": "strict",
  "profile_config": {
    "determinism": "bit-exact",
    "precision": "f64",
    "fft_provider": "reference",
    "rng_seed": 42
  }
}
```

### Per-Node Overrides

```json
{
  "id": "reverb1",
  "op": "reverb",
  "params": {"wet": 0.3},
  "profile_overrides": {
    "determinism": "repro",
    "precision": "f32",
    "ir_cache": true
  }
}
```

**Precedence:** Node override > Module > Scene > Global profile

---

## Kernel Processing Pipeline

1. **Parse** JSON → IR object
2. **Validate** types, units, DAG, rates
3. **Type inference** for unspecified types/rates
4. **Rate assignment** & multirate scheduler setup
5. **Operator resolution** from registry
6. **MLIR lowering** (dialect emission)
7. **Optimization** passes
8. **Code generation** (LLVM/GPU/Audio backends)
9. **Runtime execution**

---

## Frontend Emission Examples

### From Morphogen.Audio

```morphogen
scene SimpleTone {
  let osc = sine(440Hz)
  let filtered = lpf(osc, cutoff=2kHz, q=0.8)
  out mono = filtered
}
```

**Emits Graph IR:**
```json
{
  "version": "1.0",
  "nodes": [
    {"id": "osc", "op": "sine", "params": {"freq": "440Hz"}},
    {"id": "lpf", "op": "lpf", "params": {"cutoff": "2kHz", "q": 0.8}}
  ],
  "edges": [
    {"from": "osc:out", "to": "lpf:in", "type": "Sig"}
  ],
  "outputs": {
    "mono": ["lpf:out"]
  }
}
```

---

### From RiffStack

```yaml
version: 0.3
tracks:
  - id: lead
    expr: "saw 220 0.7 lowpass 1200 0.9 play"
```

**Emits Graph IR:**
```json
{
  "version": "1.0",
  "nodes": [
    {"id": "saw1", "op": "sawtooth", "params": {"freq": "220Hz", "amp": 0.7}},
    {"id": "lpf1", "op": "lowpass", "params": {"cutoff": "1200Hz", "q": 0.9}}
  ],
  "edges": [
    {"from": "saw1:out", "to": "lpf1:in", "type": "Sig"}
  ],
  "outputs": {
    "default": ["lpf1:out"]
  }
}
```

---

## Versioning & Compatibility

### Version Field

```json
{
  "version": "1.0",
  // ...
}
```

**Semantic versioning:**
- `1.0`: Initial stable spec
- `1.1`: Backward-compatible additions (new fields, optional attributes)
- `2.0`: Breaking changes (schema restructure)

### Forward Compatibility

Unknown fields are **ignored** (not errors):

```json
{
  "version": "1.0",
  "future_field": "ignored by v1.0 kernel",
  "nodes": [...]
}
```

### Deprecation

```json
{
  "deprecated": {
    "old_param_name": "Use 'new_param_name' instead (deprecated in v1.2)"
  }
}
```

---

## Introspection & Debugging

### Graph Export

```bash
$ kairo graph export scene.morph --format json > scene_graph.json
```

### Visualization

```bash
$ kairo graph visualize scene_graph.json --output scene_graph.png
```

### Validation

```bash
$ kairo graph validate scene_graph.json
✓ Schema version: 1.0
✓ Type checking: OK (12 edges)
✓ DAG constraint: OK (no cycles)
✓ Unit consistency: OK
✓ Rate compatibility: OK
```

---

## Extensions

### Custom Operators

Add to operator registry, emit in Graph IR:

```json
{
  "id": "custom1",
  "op": "my_custom_op",
  "params": {"alpha": 0.5},
  "outputs": [{"name": "out", "type": "Sig"}],
  "metadata": {
    "registry_path": "extensions.my_custom_op"
  }
}
```

### External Resources

Reference external files (IRs, samples, models):

```json
{
  "id": "reverb1",
  "op": "convolution",
  "params": {
    "ir": "@resource:hall_reverb.wav"
  }
}
```

---

## Summary

Morphogen Graph IR provides:

✅ **Unified representation** for all frontends
✅ **Type-safe** with full annotations
✅ **Human-readable** JSON format
✅ **Deterministic** execution semantics
✅ **Extensible** schema with versioning
✅ **Inspectable** for debugging and visualization

It's the **contract** between high-level DSLs and the low-level kernel — enabling clean separation, multiple frontends, and robust validation.
