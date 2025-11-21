# 🧩 Creative Computation DSL — Language Reference v0.2.2

*A typed, semantics-first DSL for expressive, deterministic simulations and generative computation.*

---

## 0) Principles

* **Pure per-step graphs, explicit cross-step state**
* **Deterministic semantics**, reproducible RNG and ordering
* **Composability + clarity:** tiny vocabulary, maximal reuse
* **MLIR-oriented lowering:** every op maps to a dialect cleanly
* **Live creativity:** tunable profiles, hot-reload runtime

---

## 1) Core Types and Units

| Type | Description |
| -------------------- | --------------------------------------- |
| FieldND<T [unit?]> | Dense grid field (2D/3D). |
| Agents<Record> | Sparse agent sets with stable id:u64. |
| Signal<T [unit?]> | Time-varying scalar/vector. |
| Visual | Opaque renderable (linear RGB). |
| BoundarySpec | reflect, periodic, noSlip, etc. |
| Vec2, Vec3 | Coordinates in world/sim space. |
| Link | Metadata dependency (no runtime cost). |

**Unit policy:** safe promotions allowed, lossy casts → error (override with @allow_unit_cast).

---

## 2) Structure and Time

```dsl
step { ... }                        # single tick
substep(n) { ... }                  # repeat n× with dt/n
module Name(params...) { ... }      # reusable subsystem
compose(A, B, ...)                  # parallel graphs
```

### Temporal/state ops

```dsl
step.state(name, init)            : T
signal.integrate(sig, dt)         : Signal<T>
signal.delay(sig, n=1)            : Signal<T>
adaptive_dt(cfl, max_dt, min_dt)  : f32
@double_buffer grid : Field2D<f32>
```

### Iteration and linking

```dsl
iterate(expr, until=cond, max_iter=1000) : T
link(A -> B, mode="oneway" | "bidirectional") : Link
```

---

## 3) Signal / Audio Domain

| Op | Signature | Purpose |
| ---------------------------------------- | -------------------- | ---------------------------- |
| signal.osc(freq, phase=0, shape="sin") | → Signal<f32> | oscillator |
| signal.noise(freq=1, seed=0) | → Signal<f32> | noise |
| signal.env(a,d,s,r) | → Signal<f32> | ADSR |
| signal.filter(sig, type, cutoff, reso) | → Signal<f32> | 1-pole / biquad |
| signal.mix([sig...]) | → Signal<f32> | float sum (@clip optional) |
| signal.map(sig, fn) | → Signal<U> | function map |
| signal.trigger(event, fn) | → Signal<U> | rising-edge trigger |
| signal.rate(hz) | → Signal<f32 [Hz]> | fixed rate |
| signal.sample_rate() | → f32 [Hz] | current rate |
| signal.block(sig, n_samples=512) | → Array<f32> | batch render |

---

## 4) Field Operations

Core PDE toolkit + new stencil and gradient ops.

```dsl
field.alloc(type, size)                    : FieldND<T>
field.advect(x, v, dt, method)             : FieldND<T>
field.diffuse(x, rate, dt, method, iter)   : FieldND<T>
field.react(a, b, Params)                  : FieldND<T>
field.laplacian(x)                         : FieldND<T>
field.gradient(x)                          : FieldND<VecN>
field.divergence(v)                        : FieldND<f32>
field.project(v, method, tol, iter)        : FieldND<VecN>
field.boundary(x, spec)                    : FieldND<T>
field.sample(x, pos, interp, out_of_bounds): T
field.combine(a, b, fn)                    : FieldND<T>
field.map(x, fn)                           : FieldND<U>
field.mask(x, mask)                        : FieldND<T>
field.threshold(x, t)                      : FieldND<bool>
field.resize(x, size, interp)              : FieldND<T>
field.random(shape, seed)                  : FieldND<f32>
field.integrate(x, rate, dt)               : FieldND<T>
field.stencil(x, fn, radius=1)             : FieldND<U>
field.sample_grad(x, pos, interp, out_of_bounds) : VecN<T>
```

out_of_bounds="boundary|zero|clamp" — uses last boundary spec if "boundary".

---

## 5) Agent Operations

```dsl
type Particle = { id:u64, pos:Vec2[m], vel:Vec2[m/s], color:i32, energy:f32 }
```

| Op | Signature | Description | |
| ------------------------------------------ | ---------------- | ---------------------- | --------------- |
| agent.map(A, fn) | → Agents | per-agent transform | |
| agent.force_sum(A, rule, method) | → Forces | pairwise forces | |
| agent.integrate(A, F, dt, method) | → Agents | update | |
| agent.sample_field(A, field, grad=false) | → AgentSignal<T | VecN<T>> | field sampling |
| agent.deposit(A, field, kernel) | → Field<T> | scatter | |
| agent.spawn(A, template | fn) | → Agents | spawn |
| agent.remove(A, pred) | → Agents | remove | |
| agent.when(pred, then=op) | → Agents | conditional | |
| agent.reduce(A, fn, init) | → Record | reduction | |
| agent.mutate(A, fn, rate, seed) | → Agents | probabilistic mutation | |
| agent.reproduce(A, template | fn, rate) | → Agents | offspring spawn |

**Determinism**
* Stable (id, creation_index) ordering
* RNG: **Philox 4×32-10**, seeded hash64(global_seed, id, tick, seed)
* Morton ordering for Barnes–Hut builds

---

## 6) Visual Domain

| Op | Signature | Purpose |
| ------------------------------------ | ---------- | -------------- |
| visual.colorize(field, palette) | → Visual | scalar→RGB |
| visual.points(agents, color) | → Visual | point sprites |
| visual.layer([vis...], blend) | → Visual | composition |
| visual.filter(vis, fn) | → Visual | post-process |
| visual.coord_warp(vis, fn) | → Visual | geometric warp |
| visual.retime(vis, t_sig) | → Visual | frame history |
| visual.text(src, fmt) | → Visual | overlay |
| visual.output(vis, target, format) | → Frame | display |
| visual.tag(key=value) | → Visual | metadata tag |

All visuals use **linear RGB**; display conversion at visual.output.

---

## 7) I/O and Streams

```dsl
io.load_field(path, format="auto")         : Field<T>
io.save_visual(vis, path)                  : Unit
io.stream<Type[unit?]>(name)               : Signal<T>
io.output(signal|visual, target="audio"|"video"|"device", format="auto") : Unit
```

Streams are deterministic only when **recorded/replayed**.

---

## 8) Solvers, Profiles, and Precision

### Per-op control

```dsl
field.diffuse(x, rate, dt, method="cg", iter=20, precond="jacobi")
```

### Profiles

```dsl
profile low:  field.project=jacobi(iter=20); precision=f16
profile high: field.project=multigrid(iter=5); precision=f64
```

### Registry

```dsl
solver my_diff = field.diffuse(method="cg", iter=30)
```

**Precedence:** per-op > solver alias > module profile > global profile

**Preconditioners:** jacobi, ilu0, multigrid_smoother

**Precision:** compute dtype; @storage() overrides per resource.

---

## 9) Diagnostics, Benchmarking & Metadata

```dsl
@benchmark(name="rd_perf")
@metadata(key="category", value="fluid")
@param wind_rate : f32 = 0.01 @range(0,0.1) @doc "Wind diffusion"
```

Lints: unit errors, unused states, NaN/Inf, zero-sized fields.

---

## 10) Execution & Determinism

```
step {
  dispatch kernels in topo order
  swap(@double_buffer)
  render(visual.output)
}
```

### Determinism tiers

| Tier | Definition | Examples |
| -------------------- | ------------------------------ | --------------------------------------- |
| **strict** | bit-identical cross-device | field.diffuse, agent.force_sum |
| **reproducible** | deterministic within precision | field.project, visual.filter |
| **nondeterministic** | external I/O or adaptive stop | io.stream(live), iterate(unbounded) |

Use @nondeterministic to annotate.

---

## 11) MLIR Lowering Map

| New / Core Op | Dialects | Notes |
| --------------------------------- | ------------------ | ------------------- |
| field.stencil | linalg, affine | fused neighborhoods |
| field.sample_grad | vector, math | analytic derivative |
| field.integrate | arith | simple add/mul |
| agent.mutate, agent.reproduce | scf.for, gpu | RNG-driven updates |
| iterate | scf.while | dynamic loop |
| io.output(audio) | async, memref | host callback |
| link | metadata only | graph viz |
| existing ops | as v0.2.1 | unchanged |

---

## 12) Example — Evolutionary Fluid Hybrid

```dsl
set profile = medium
set dt = adaptive_dt(cfl=0.5, max_dt=0.02, min_dt=0.002)

@double_buffer vel, temp : Field2D<f32>
agents = step.state(agent.alloc(Particle, count=2000))

vel = field.advect(vel, vel, dt)
vel = field.project(vel, method="cg", iter=40)

temp = field.diffuse(temp, rate=κ, dt)
temp = field.react(temp, vel, Params{k:0.3})

agents = agent.sample_field(agents, temp, grad=true)
agents = agent.mutate(agents, fn=mutate_energy, rate=0.05)
agents = agent.reproduce(agents, template=default, rate=0.02)

visual.output( visual.layer([
  visual.colorize(temp, palette="fire"),
  visual.points(agents, color="white")
]) )
```

---

## 13) Conformance Highlights

| Test | Expected Result |
| ------------------------------------ | ------------------------ |
| Unit mismatch | compile error |
| @allow_unit_cast | warning |
| Spawn/remove | identical runs |
| iterate with max_iter | deterministic loop count |
| Barnes–Hut | bit-exact forces |
| combine→mask→diffuse fused/unfused | identical |
| Recorded io.stream(audio) | deterministic playback |

---

## 14) Concept Summary of Additions

| Category | New Ops | Capability |
| ---------------- | --------------------------------------- | ------------------------- |
| **Structure** | iterate, link | dynamic loops, graph viz |
| **Field** | stencil, sample_grad, integrate | richer PDEs |
| **Agent** | mutate, reproduce | evolutionary systems |
| **Signal/Audio** | block, io.output(audio) | streaming DSP |
| **Diagnostics** | @benchmark, visual.tag, @metadata | profiling & introspection |

---

> **Creative Computation DSL v0.2.2**
> A unified language where simulations, agents, signals, and visuals interoperate seamlessly — deterministically, portably, and joyfully.
