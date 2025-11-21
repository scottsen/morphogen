# SPEC: Deterministic Multirate Scheduler

**Version:** 1.0 Draft
**Status:** RFC
**Last Updated:** 2025-11-13

---

## Overview

The **Morphogen Scheduler** is the heart of deterministic multirate execution. It coordinates operators running at different rates (audio, control, visual, sim) while maintaining:

1. **Sample-accurate timing** — Events fire at exact sample boundaries
2. **Deterministic ordering** — Identical runs produce identical results
3. **Efficient partitioning** — LCM-based block scheduling
4. **Cross-rate resampling** — Explicit, well-defined resampling semantics
5. **Hot reload safety** — Stable state snapshots and graph transitions

**Design Principle:** Time and rate handling must be rock-solid. This is the hardest problem in audio/physics/visual systems. Get it right once, avoid decades of hacks.

---

## Rate System

Morphogen defines **four execution rates** forming a partial order:

| Rate | Typical Frequency | Block Size | Use Case |
|------|-------------------|------------|----------|
| `audio` | 44.1kHz, 48kHz, 96kHz | 64-512 samples | Audio signals, oscillators, filters |
| `control` | 100Hz, 1kHz | 1-10 samples | Envelopes, LFOs, parameter modulation |
| `visual` | 30Hz, 60Hz, 120Hz | 1 frame | Rendering, visualization |
| `sim` | Variable | dt-dependent | Physics timesteps, PDE solvers |

**Rate Ordering:**
```
audio ≥ control ≥ visual ≥ sim
```

**Invariants:**
- Higher rates can read from lower rates (with explicit resampling)
- Lower rates cannot read from higher rates without aggregation (e.g., RMS, peak)
- Cross-rate connections must be explicit

---

## Scheduler Architecture

### 1. Rate Groups

The scheduler partitions operators into **rate groups** based on their declared rate:

```
Rate Group: audio (48kHz)
  - sine_osc
  - lpf1
  - multiply1

Rate Group: control (1kHz)
  - adsr_env
  - lfo1

Rate Group: visual (60Hz)
  - render_field
```

**Execution Model:** Each rate group executes in **blocks** at its rate.

---

### 2. LCM-Based Partitioning

The scheduler computes the **Least Common Multiple (LCM)** of all rates to determine the **master tick period**:

**Example:**
```
audio_rate = 48000 Hz
control_rate = 1000 Hz

LCM(48000, 1000) = 48000

Master tick period = 1 / 48000 = 20.83 μs
```

**Execution Plan:**
- Every **48 master ticks** → Execute audio block (48 samples @ 48kHz)
- Every **48 master ticks** → Execute control block (1 sample @ 1kHz)

**Block Sizes:**
```
audio_block_size = master_rate / audio_rate = 48000 / 48000 = 1
control_block_size = master_rate / control_rate = 48000 / 1000 = 48
```

Wait, this is backwards. Let me recalculate:

```
GCD(48000, 1000) = 1000
audio_rate / GCD = 48000 / 1000 = 48
control_rate / GCD = 1000 / 1000 = 1

Master clock runs at GCD = 1000 Hz
audio processes 48 samples per master tick
control processes 1 sample per master tick
```

Correct model:

**Master Clock:** GCD of all rates
**Block Multipliers:** rate / GCD

---

### 3. Hop Size and Partitioning

To reduce overhead, the scheduler processes **multiple master ticks** at once:

**Hop Size:** Configurable, typically 64-512 samples (for audio rate)

**Example (hop_size = 128 for audio):**
```
audio_rate = 48000 Hz
control_rate = 1000 Hz
hop_size_audio = 128 samples

hop_duration = 128 / 48000 = 2.67 ms
control_samples_per_hop = 1000 * 0.00267 = 2.67 ≈ 3 samples
```

**Execution:**
```
Hop 0:
  - Process 128 audio samples
  - Process 3 control samples
  - Resample control → audio (3 → 128)

Hop 1:
  - Process 128 audio samples
  - Process 3 control samples
  - ...
```

---

### 4. Sample-Accurate Event Fencing

Events must fire at **exact sample boundaries** within a hop.

**Event Queue:**
```python
class Event:
    time: int       # Sample index (absolute)
    payload: Any    # Event data (Note, Control, Trigger, etc.)

event_queue: List[Event]  # Sorted by time
```

**Execution with Events:**
```python
def execute_hop(start_sample, hop_size):
    end_sample = start_sample + hop_size
    events_in_hop = [e for e in event_queue if start_sample <= e.time < end_sample]

    # Partition hop into sub-blocks at event boundaries
    boundaries = [start_sample] + [e.time for e in events_in_hop] + [end_sample]

    for i in range(len(boundaries) - 1):
        sub_start = boundaries[i]
        sub_end = boundaries[i + 1]
        sub_size = sub_end - sub_start

        # Execute all operators for this sub-block
        execute_rate_group("audio", sub_start, sub_size)
        execute_rate_group("control", sub_start, sub_size)

        # Fire events at boundary
        if i < len(events_in_hop):
            fire_event(events_in_hop[i])
```

**Invariants:**
- Events split hops into sub-blocks
- Each sub-block executes deterministically
- Event ordering is stable (sorted by time, then ID)

---

### 5. Cross-Rate Resampling

When an audio-rate operator reads from a control-rate operator, **explicit resampling** occurs:

**Resampling Modes:**

| Mode | Description | Use Case |
|------|-------------|----------|
| `hold` | Zero-order hold (step) | Control signals (envelopes, switches) |
| `linear` | Linear interpolation | Smooth parameter changes |
| `cubic` | Cubic interpolation | Higher-quality smoothing |
| `sinc` | Windowed sinc | High-quality audio resampling |

**Example:**
```morphogen
scene CrossRate {
  let env: Stream<f32, time, control> = adsr(...)  // 1kHz
  let osc: Stream<f32, time, audio> = sine(440Hz)  // 48kHz

  // Implicit resampling: control → audio
  let modulated = osc * env
  // Compiler inserts: env_upsampled = resample(env, to_rate=audio, mode=linear)
}
```

**Resampling Implementation:**
```python
def resample(input_buffer, from_rate, to_rate, mode):
    """Resample from_rate → to_rate."""
    ratio = to_rate / from_rate
    output_size = int(len(input_buffer) * ratio)

    if mode == "hold":
        return zero_order_hold(input_buffer, output_size)
    elif mode == "linear":
        return linear_interpolate(input_buffer, output_size)
    elif mode == "cubic":
        return cubic_interpolate(input_buffer, output_size)
    elif mode == "sinc":
        return windowed_sinc(input_buffer, output_size)
```

---

### 6. Double Buffering

Operators with state (delays, filters, integrators) use **double buffering** to avoid read-write conflicts:

**State Buffers:**
```python
class OperatorState:
    read_buffer: Buffer   # Current input
    write_buffer: Buffer  # Next output

def execute_operator(op, state):
    # Read from read_buffer
    # Write to write_buffer
    ...

def swap_buffers(state):
    state.read_buffer, state.write_buffer = state.write_buffer, state.read_buffer
```

**Execution Order:**
```
1. Execute all operators (read from read_buffer, write to write_buffer)
2. Swap buffers (read_buffer ← write_buffer)
3. Repeat
```

---

## Scheduler Algorithm

### Initialization

```python
def init_scheduler(graph, sample_rate, hop_size):
    """Initialize scheduler for graph."""

    # 1. Extract all rates from graph
    rates = extract_rates(graph)

    # 2. Compute GCD (master clock)
    master_rate = gcd(*rates)

    # 3. Compute block multipliers
    multipliers = {rate: rate // master_rate for rate in rates}

    # 4. Build rate groups
    rate_groups = partition_by_rate(graph)

    # 5. Topological sort within each group
    for group in rate_groups:
        group.operators = topological_sort(group.operators)

    # 6. Initialize event queue
    event_queue = []

    return Scheduler(
        graph=graph,
        sample_rate=sample_rate,
        hop_size=hop_size,
        master_rate=master_rate,
        multipliers=multipliers,
        rate_groups=rate_groups,
        event_queue=event_queue,
    )
```

---

### Execution Loop

```python
def run_scheduler(scheduler, duration_samples):
    """Execute graph for duration_samples."""

    current_sample = 0

    while current_sample < duration_samples:
        # Compute hop size (may be smaller at end)
        hop = min(scheduler.hop_size, duration_samples - current_sample)

        # Execute hop with event fencing
        execute_hop_with_events(scheduler, current_sample, hop)

        # Advance time
        current_sample += hop

    return get_outputs(scheduler)
```

---

### Hop Execution with Fencing

```python
def execute_hop_with_events(scheduler, start_sample, hop_size):
    """Execute hop with sample-accurate event fencing."""

    end_sample = start_sample + hop_size

    # Get events in this hop
    events = [e for e in scheduler.event_queue
              if start_sample <= e.time < end_sample]

    # Create sub-blocks at event boundaries
    boundaries = sorted([start_sample] + [e.time for e in events] + [end_sample])

    for i in range(len(boundaries) - 1):
        sub_start = boundaries[i]
        sub_end = boundaries[i + 1]
        sub_size = sub_end - sub_start

        if sub_size == 0:
            continue  # Skip zero-length blocks

        # Execute each rate group for this sub-block
        for rate_group in scheduler.rate_groups:
            rate_samples = compute_rate_samples(
                rate_group.rate, scheduler.sample_rate, sub_size
            )
            execute_rate_group(rate_group, sub_start, rate_samples)

        # Fire events at sub-block end
        if i < len(events):
            fire_event(events[i], scheduler)

    # Swap double buffers
    for state in scheduler.operator_states:
        swap_buffers(state)
```

---

### Rate Group Execution

```python
def execute_rate_group(rate_group, start_sample, num_samples):
    """Execute all operators in rate group."""

    # Operators are already topologically sorted
    for op in rate_group.operators:
        # Get input buffers (with resampling if needed)
        inputs = get_inputs(op, num_samples)

        # Execute operator
        outputs = op.execute(inputs, num_samples)

        # Store outputs
        set_outputs(op, outputs)
```

---

## Determinism Guarantees

### 1. Stable Operator Ordering

Within a rate group, operators are **topologically sorted** and executed in deterministic order:

```python
def topological_sort(operators):
    """Deterministic topological sort (stable)."""
    # Use DFS with stable tie-breaking (by operator ID)
    visited = set()
    result = []

    def visit(op):
        if op in visited:
            return
        visited.add(op)
        for dep in sorted(op.dependencies, key=lambda o: o.id):
            visit(dep)
        result.append(op)

    for op in sorted(operators, key=lambda o: o.id):
        visit(op)

    return result
```

### 2. Stable Event Ordering

Events at the same time are ordered by **event ID**:

```python
def sort_events(events):
    """Sort events deterministically."""
    return sorted(events, key=lambda e: (e.time, e.id))
```

### 3. Deterministic Resampling

Resampling must be **bit-exact** (in strict profile):

```python
def resample_strict(input, from_rate, to_rate, mode):
    """Bit-exact resampling (strict profile)."""
    # Use reference implementation (no vendor libraries)
    # Exact coefficients (no approximations)
    ...
```

---

## Hot Reload

### State Snapshot

When hot-reloading a graph, the scheduler must **snapshot all state**:

```python
def snapshot_scheduler(scheduler):
    """Snapshot current scheduler state."""
    return {
        "current_sample": scheduler.current_sample,
        "operator_states": {
            op.id: op.state.serialize() for op in scheduler.operators
        },
        "event_queue": scheduler.event_queue.copy(),
        "buffer_states": snapshot_buffers(scheduler),
    }
```

### State Restoration

```python
def restore_scheduler(scheduler, snapshot):
    """Restore scheduler from snapshot."""
    scheduler.current_sample = snapshot["current_sample"]

    for op in scheduler.operators:
        if op.id in snapshot["operator_states"]:
            op.state.deserialize(snapshot["operator_states"][op.id])

    scheduler.event_queue = snapshot["event_queue"]
    restore_buffers(scheduler, snapshot["buffer_states"])
```

### Graph Hash Validation

```python
def validate_hot_reload(old_graph, new_graph):
    """Validate that hot reload is safe."""

    # Check that operator signatures match
    for old_op, new_op in zip(old_graph.operators, new_graph.operators):
        if old_op.signature != new_op.signature:
            raise HotReloadError(
                f"Operator {old_op.id} signature changed: "
                f"{old_op.signature} → {new_op.signature}"
            )

    # Check that rates haven't changed
    if old_graph.rates != new_graph.rates:
        raise HotReloadError("Cannot change rates during hot reload")
```

---

## Edge Cases

### Case 1: Zero-Length Sub-Blocks

Events at the same sample create zero-length sub-blocks. Skip them:

```python
if sub_size == 0:
    continue
```

### Case 2: Non-Integer Rate Ratios

If rates don't divide evenly, use fractional resampling:

```python
# Example: 44.1kHz audio, 1kHz control
# Ratio = 44.1 (not an integer)
# Use fractional resampling with interpolation
```

### Case 3: Visual Rate Sync

Visual rate (60Hz) may not divide evenly into audio rate (48kHz):

```python
# 48000 / 60 = 800 samples per frame
# Process 800 audio samples, then 1 visual frame
```

---

## Performance Optimizations

### 1. Batch Event Processing

Instead of processing events one at a time, batch them:

```python
# Group consecutive events with no operators between them
event_batches = group_consecutive_events(events)

for batch in event_batches:
    fire_events(batch)  # Fire all at once
```

### 2. Lazy Resampling

Only resample when needed:

```python
if op.input_rate == op.output_rate:
    # No resampling needed
    pass
else:
    # Resample
    input = resample(input, op.input_rate, op.output_rate)
```

### 3. SIMD Vectorization

Vectorize rate group execution:

```python
# Execute multiple operators in parallel (SIMD)
for i in range(0, len(rate_group.operators), SIMD_WIDTH):
    ops = rate_group.operators[i : i + SIMD_WIDTH]
    execute_vectorized(ops, num_samples)
```

---

## Testing Strategy

### Test 1: Multirate Correctness

```python
def test_multirate_execution():
    """Test mixed-rate graph."""

    graph = Graph()
    graph.add_operator("osc", "sine", rate="audio", params={"freq": "440Hz"})
    graph.add_operator("env", "adsr", rate="control", params={"attack": "0.1s"})
    graph.add_operator("mul", "multiply", rate="audio")
    graph.add_edge("osc", "mul", port="in1")
    graph.add_edge("env", "mul", port="in2")  # Implicit resampling

    scheduler = init_scheduler(graph, sample_rate=48000, hop_size=128)
    output = run_scheduler(scheduler, duration_samples=4800)

    # Verify output shape
    assert len(output) == 4800

    # Verify determinism
    output2 = run_scheduler(scheduler, duration_samples=4800)
    assert np.array_equal(output, output2)
```

### Test 2: Event Fencing

```python
def test_event_fencing():
    """Test sample-accurate event timing."""

    graph = Graph()
    graph.add_operator("osc", "sine", rate="audio", params={"freq": "440Hz"})

    # Add events at specific samples
    events = [
        Event(time=0, payload=NoteOn(pitch=440)),
        Event(time=100, payload=NoteOff()),
        Event(time=200, payload=NoteOn(pitch=880)),
    ]

    scheduler = init_scheduler(graph, sample_rate=48000, hop_size=128)
    scheduler.event_queue = events

    output = run_scheduler(scheduler, duration_samples=300)

    # Verify events fired at exact samples
    assert output[99] != output[100]  # Note change at sample 100
    assert output[199] != output[200]  # Note change at sample 200
```

### Test 3: Hot Reload

```python
def test_hot_reload():
    """Test hot reload with state preservation."""

    graph1 = create_simple_graph()
    scheduler = init_scheduler(graph1, sample_rate=48000, hop_size=128)

    # Run for 1000 samples
    run_scheduler(scheduler, duration_samples=1000)

    # Snapshot state
    snapshot = snapshot_scheduler(scheduler)

    # Modify graph (change parameter)
    graph2 = modify_graph_parameter(graph1, "osc", "freq", "880Hz")

    # Hot reload
    scheduler2 = init_scheduler(graph2, sample_rate=48000, hop_size=128)
    restore_scheduler(scheduler2, snapshot)

    # Continue execution
    output = run_scheduler(scheduler2, duration_samples=1000)

    # Verify no discontinuities (no clicks/pops)
    assert no_discontinuities(output)
```

---

## Summary

The Morphogen Scheduler provides:

✅ **Deterministic multirate execution** — Audio, control, visual, sim rates
✅ **Sample-accurate event fencing** — Events fire at exact boundaries
✅ **Efficient LCM partitioning** — Minimize overhead
✅ **Explicit cross-rate resampling** — Well-defined semantics
✅ **Double buffering** — Safe stateful operators
✅ **Hot reload support** — State snapshots and restoration

This is the foundation for **rock-solid time handling** in Morphogen.

---

## References

- `type-system.md` — Rate annotations on Stream types
- `profiles.md` — Profile affects block size and resampling
- `graph-ir.md` — Graph IR includes rate annotations
- `snapshot-abi.md` — State snapshot format
