# SPEC: Snapshot ABI & Hot Reload

**Version:** 1.0 Draft
**Status:** RFC
**Last Updated:** 2025-11-13

---

## Overview

The **Snapshot ABI** defines how Morphogen captures and restores execution state, enabling:

1. **Hot reload** — Update code without restarting execution
2. **State persistence** — Save/load sessions
3. **Debugging** — Capture state for inspection
4. **Time travel** — Rewind and replay execution
5. **Distributed execution** — Migrate state across nodes

**Design Principle:** Snapshots must be **safe, deterministic, and portable**. No clicks, pops, NaNs, or undefined behavior.

---

## What is State?

In Morphogen, **state** includes:

1. **Operator state** — Internal buffers (delays, filters, integrators, RNGs)
2. **Buffer contents** — Current audio/control/visual data
3. **Scheduler state** — Current sample index, rate counters
4. **Event queue** — Pending events
5. **RNG seeds** — Current Philox counter values
6. **Profile metadata** — Active profile and overrides

**Stateless operators** (e.g., `add`, `mul`, `sin`) have no state to snapshot.

---

## Snapshot Format

### Top-Level Structure

```json
{
  "version": "1.0",
  "kairo_version": "0.4.0",
  "timestamp": "2025-11-13T12:00:00Z",
  "graph_hash": "sha256:abc123...",
  "profile": "repro",

  "scheduler": {
    "current_sample": 48000,
    "sample_rate": 48000,
    "hop_size": 128
  },

  "operators": {
    "osc1": {
      "state": { /* Operator-specific state */ },
      "buffers": { /* Double-buffered data */ }
    },
    "lpf1": {
      "state": { /* Filter coefficients, history */ },
      "buffers": { /* Current output */ }
    }
  },

  "event_queue": [
    {"time": 48100, "payload": {"type": "NoteOn", "pitch": 440}},
    {"time": 48200, "payload": {"type": "NoteOff"}}
  ],

  "rng_state": {
    "global_seed": 42,
    "operator_counters": {
      "noise1": 1234,
      "noise2": 5678
    }
  }
}
```

---

## Operator State Schema

### Stateless Operators

```json
{
  "id": "mul1",
  "op": "multiply",
  "state": null  // No state
}
```

---

### Stateful Operators: Delay

```json
{
  "id": "delay1",
  "op": "delay",
  "state": {
    "delay_line": {
      "capacity": 48000,         // Buffer size (samples)
      "write_index": 1234,       // Current write position
      "data_type": "f32",
      "data": [0.1, 0.2, ...]    // Buffer contents (base64 or array)
    }
  }
}
```

---

### Stateful Operators: IIR Filter

```json
{
  "id": "lpf1",
  "op": "lpf",
  "state": {
    "coefficients": {
      "b0": 0.067455,
      "b1": 0.134911,
      "b2": 0.067455,
      "a1": -1.142980,
      "a2": 0.412801
    },
    "history": {
      "x1": 0.05,     // Previous input samples
      "x2": 0.03,
      "y1": 0.02,     // Previous output samples
      "y2": 0.01
    }
  }
}
```

---

### Stateful Operators: RNG

```json
{
  "id": "noise1",
  "op": "white_noise",
  "state": {
    "rng": {
      "algorithm": "philox",
      "seed": 42,
      "counter": 12345,     // Current Philox counter
      "key": [1234, 5678]   // Philox key
    }
  }
}
```

---

### Stateful Operators: Field (2D/3D)

```json
{
  "id": "field1",
  "op": "velocity_field",
  "state": {
    "grid": {
      "dimensions": [128, 128],
      "spacing": 0.1,
      "centering": "node"
    },
    "data": {
      "type": "Vec2<f32>",
      "compression": "zlib",
      "data_base64": "eJzt3WEKAAAAAA..."  // Compressed binary data
    }
  }
}
```

---

## Snapshot Operations

### 1. Capture Snapshot

```python
def snapshot(graph, scheduler, operators):
    """Capture current execution state."""

    snapshot = {
        "version": "1.0",
        "kairo_version": get_kairo_version(),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "graph_hash": compute_graph_hash(graph),
        "profile": graph.profile,

        "scheduler": {
            "current_sample": scheduler.current_sample,
            "sample_rate": scheduler.sample_rate,
            "hop_size": scheduler.hop_size,
        },

        "operators": {},
        "event_queue": scheduler.event_queue.copy(),
        "rng_state": snapshot_rng_state(operators),
    }

    for op in operators:
        if op.has_state():
            snapshot["operators"][op.id] = {
                "state": op.serialize_state(),
                "buffers": serialize_buffers(op),
            }

    return snapshot
```

---

### 2. Restore Snapshot

```python
def restore(snapshot, graph, scheduler, operators):
    """Restore execution state from snapshot."""

    # Validate graph hash
    current_hash = compute_graph_hash(graph)
    if current_hash != snapshot["graph_hash"]:
        raise HotReloadError(
            f"Graph hash mismatch: {current_hash} != {snapshot['graph_hash']}"
        )

    # Restore scheduler state
    scheduler.current_sample = snapshot["scheduler"]["current_sample"]
    scheduler.sample_rate = snapshot["scheduler"]["sample_rate"]
    scheduler.hop_size = snapshot["scheduler"]["hop_size"]

    # Restore operator state
    for op in operators:
        if op.id in snapshot["operators"]:
            op_snapshot = snapshot["operators"][op.id]
            op.deserialize_state(op_snapshot["state"])
            deserialize_buffers(op, op_snapshot["buffers"])

    # Restore event queue
    scheduler.event_queue = snapshot["event_queue"].copy()

    # Restore RNG state
    restore_rng_state(operators, snapshot["rng_state"])
```

---

### 3. Save to Disk

```python
def save_snapshot(snapshot, path):
    """Save snapshot to disk."""

    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)
```

**Binary format (for large fields):**
```python
def save_snapshot_binary(snapshot, path):
    """Save snapshot in binary format (MessagePack)."""

    with open(path, "wb") as f:
        msgpack.pack(snapshot, f)
```

---

### 4. Load from Disk

```python
def load_snapshot(path):
    """Load snapshot from disk."""

    with open(path, "r") as f:
        return json.load(f)
```

---

## Hot Reload Protocol

### Safe Hot Reload Conditions

Hot reload is **safe** if:

1. ✅ **Graph topology unchanged** — Same operators, same connections
2. ✅ **Operator signatures unchanged** — Same inputs/outputs/params
3. ✅ **Rates unchanged** — Same audio/control/visual rates
4. ✅ **Profile compatible** — Same or weaker profile (strict → repro OK, repro → strict NOT OK)

Hot reload is **unsafe** if:

1. ❌ **Topology changed** — Added/removed operators or edges
2. ❌ **Signatures changed** — Input/output types changed
3. ❌ **Rates changed** — Sample rate or rate groups changed

---

### Hot Reload Workflow

```python
def hot_reload(old_graph, new_graph, scheduler, operators):
    """Perform hot reload with validation."""

    # 1. Validate compatibility
    validate_hot_reload_safe(old_graph, new_graph)

    # 2. Snapshot current state
    snapshot = snapshot(old_graph, scheduler, operators)

    # 3. Reinitialize operators with new graph
    new_operators = instantiate_operators(new_graph)

    # 4. Restore state
    try:
        restore(snapshot, new_graph, scheduler, new_operators)
    except Exception as e:
        # Rollback on failure
        raise HotReloadError(f"Failed to restore state: {e}")

    return new_operators
```

---

### Validation Rules

```python
def validate_hot_reload_safe(old_graph, new_graph):
    """Validate hot reload is safe."""

    # Check graph topology
    if old_graph.operator_ids != new_graph.operator_ids:
        raise HotReloadError("Operator set changed (added/removed operators)")

    if old_graph.edges != new_graph.edges:
        raise HotReloadError("Graph topology changed (edges modified)")

    # Check operator signatures
    for old_op, new_op in zip(old_graph.operators, new_graph.operators):
        if old_op.name != new_op.name:
            raise HotReloadError(f"Operator type changed: {old_op.name} → {new_op.name}")

        if old_op.input_types != new_op.input_types:
            raise HotReloadError(f"Input types changed for {old_op.id}")

        if old_op.output_types != new_op.output_types:
            raise HotReloadError(f"Output types changed for {old_op.id}")

    # Check rates
    if old_graph.sample_rate != new_graph.sample_rate:
        raise HotReloadError("Sample rate changed")
```

---

## Parameter Hot Reload

**Safe:** Changing operator parameters without changing topology.

```python
def hot_reload_parameter(op_id, param_name, new_value, operators):
    """Hot reload a single parameter."""

    op = operators[op_id]

    # Validate parameter exists
    if param_name not in op.params:
        raise ValueError(f"Parameter {param_name} not found in {op_id}")

    # Validate parameter type/units
    validate_param_value(op.registry_def.get_param(param_name), new_value)

    # Apply parameter change (with smoothing if needed)
    op.set_parameter(param_name, new_value, smooth=True)
```

**Smoothing:**
```python
def set_parameter(self, name, value, smooth=False):
    """Set parameter with optional smoothing."""

    if smooth:
        # Linear ramp over 10ms to avoid clicks
        ramp_samples = int(self.sample_rate * 0.01)
        old_value = self.params[name]
        self.param_ramps[name] = (old_value, value, ramp_samples)
    else:
        self.params[name] = value
```

---

## State Serialization Format

### JSON (Human-Readable)

**Pros:**
- Human-readable
- Easy to inspect and debug
- Portable across platforms

**Cons:**
- Large file size for fields
- Slower to parse

**Use case:** Debugging, small states

---

### MessagePack (Binary)

**Pros:**
- Compact binary format
- Fast serialization/deserialization
- Cross-platform

**Cons:**
- Not human-readable

**Use case:** Production, large fields

---

### NumPy `.npz` (Hybrid)

**Pros:**
- Efficient for large arrays
- Compressed (zlib)
- Python-friendly

**Cons:**
- Python-specific

**Use case:** Python-based workflows, large fields

---

## State Compression

For large fields, compress data:

```python
def serialize_field(field):
    """Serialize field with compression."""

    # Convert to bytes
    data_bytes = field.data.tobytes()

    # Compress with zlib
    compressed = zlib.compress(data_bytes, level=9)

    # Encode as base64 (for JSON compatibility)
    encoded = base64.b64encode(compressed).decode("utf-8")

    return {
        "type": str(field.dtype),
        "shape": list(field.shape),
        "compression": "zlib",
        "data_base64": encoded,
    }
```

---

## Snapshot Versioning

Snapshots include **version fields**:

```json
{
  "version": "1.0",           // Snapshot format version
  "kairo_version": "0.4.0"    // Morphogen runtime version
}
```

**Forward compatibility:**
- Older Morphogen versions reject newer snapshot formats (error)
- Newer Morphogen versions can load older snapshots (with migration)

**Migration example:**
```python
def migrate_snapshot(snapshot, from_version, to_version):
    """Migrate snapshot from old to new format."""

    if from_version == "0.9" and to_version == "1.0":
        # Example: Rename "rng_seeds" → "rng_state"
        snapshot["rng_state"] = snapshot.pop("rng_seeds")

    return snapshot
```

---

## Graph Hash

The **graph hash** ensures snapshots match the graph:

```python
def compute_graph_hash(graph):
    """Compute deterministic graph hash."""

    # Serialize graph to canonical JSON
    canonical = json.dumps(
        {
            "operators": [
                {"id": op.id, "name": op.name, "params": sorted(op.params.items())}
                for op in sorted(graph.operators, key=lambda o: o.id)
            ],
            "edges": sorted(graph.edges, key=lambda e: (e["from"], e["to"])),
        },
        sort_keys=True,
    )

    # Compute SHA256 hash
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()
```

**Validation:**
```python
if snapshot["graph_hash"] != compute_graph_hash(graph):
    raise HotReloadError("Graph mismatch: snapshot is from a different graph")
```

---

## Time Travel Debugging

### Capture Snapshots Every N Samples

```python
def run_with_snapshots(graph, scheduler, snapshot_interval=4800):
    """Run graph and capture periodic snapshots."""

    snapshots = []
    sample = 0

    while sample < duration:
        # Execute hop
        execute_hop(scheduler, sample, hop_size)

        # Capture snapshot
        if sample % snapshot_interval == 0:
            snapshots.append((sample, snapshot(graph, scheduler, operators)))

        sample += hop_size

    return snapshots
```

---

### Rewind and Replay

```python
def rewind_to(snapshots, target_sample):
    """Rewind execution to target sample."""

    # Find closest snapshot before target
    closest = max(s for s in snapshots if s[0] <= target_sample)

    # Restore snapshot
    restore(closest[1], graph, scheduler, operators)

    # Replay from snapshot to target
    replay(scheduler, closest[0], target_sample)
```

---

## Testing Strategy

### Test 1: Snapshot Roundtrip

```python
def test_snapshot_roundtrip():
    """Test snapshot capture and restore."""

    graph = create_simple_graph()
    scheduler = init_scheduler(graph, sample_rate=48000, hop_size=128)

    # Run for 1000 samples
    run(scheduler, 1000)

    # Capture snapshot
    snapshot1 = snapshot(graph, scheduler, operators)

    # Run for another 1000 samples
    run(scheduler, 1000)

    # Restore snapshot
    restore(snapshot1, graph, scheduler, operators)

    # Run for 1000 samples again
    output1 = run(scheduler, 1000)

    # Verify output is identical (deterministic)
    assert np.array_equal(output1, output_expected)
```

---

### Test 2: Hot Reload Parameter

```python
def test_hot_reload_parameter():
    """Test hot reload with parameter change."""

    graph = create_simple_synth(freq=440Hz)
    scheduler = init_scheduler(graph, sample_rate=48000, hop_size=128)

    # Run for 1000 samples
    output1 = run(scheduler, 1000)

    # Change frequency (hot reload)
    hot_reload_parameter("osc1", "freq", "880Hz", operators)

    # Run for 1000 samples
    output2 = run(scheduler, 1000)

    # Verify frequency changed
    assert dominant_frequency(output1) == 440
    assert dominant_frequency(output2) == 880

    # Verify no discontinuities (no click/pop)
    assert no_discontinuities(output2)
```

---

### Test 3: Hot Reload Unsafe

```python
def test_hot_reload_unsafe():
    """Test that unsafe hot reload is rejected."""

    old_graph = create_simple_synth(freq=440Hz)
    new_graph = create_simple_synth_with_extra_filter(freq=440Hz)

    # Attempt hot reload (should fail)
    with pytest.raises(HotReloadError, match="Operator set changed"):
        hot_reload(old_graph, new_graph, scheduler, operators)
```

---

## Summary

The Snapshot ABI provides:

✅ **State capture** — Serialize operator state, buffers, RNG seeds
✅ **Hot reload** — Update code without restarting
✅ **State persistence** — Save/load sessions
✅ **Time travel** — Rewind and replay execution
✅ **Determinism** — Snapshots are portable and bit-exact

This enables **robust development workflows** and **production reliability**.

---

## References

- `scheduler.md` — Scheduler state included in snapshots
- `operator-registry.md` — Operator state schemas
- `profiles.md` — Profile affects snapshot compatibility
- `type-system.md` — Type serialization
