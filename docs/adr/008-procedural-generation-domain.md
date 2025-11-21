# ADR-005: Procedural Generation Domain for Complex Structures

**Date:** 2025-11-15
**Status:** Proposed
**Authors:** Claude, User
**Related:** ADR-005 (Emergence Domain), ADR-002 (Cross-Domain Patterns), ../specifications/procedural-generation.md

---

## Context

Morphogen has successfully established itself as a multi-domain physics simulation and creative coding platform with:

- ✅ Audio DSP (oscillators, filters, effects)
- ✅ Visual/Fractal rendering
- ✅ Transform operations (FFT, STFT, wavelets)
- ✅ Fields/PDE operators
- ✅ Physics domains (FluidNetwork, ThermalODE, Combustion)
- ✅ Circuit modeling
- ✅ Geometry/CAD (TiaCAD-inspired)
- ✅ Emergence domain (CA, ABM, RD, L-systems, swarms)

However, while the EmergenceDomain includes L-systems as one component, **there is a massive opportunity for a dedicated Procedural Generation Domain** that goes far beyond basic L-systems to encompass the full spectrum of procedural modeling, specifically:

1. **Tree and Vegetation Generation** — Inspired by SpeedTree but more general
2. **Terrain and Landscape Synthesis** — Heightfields, erosion, biomes
3. **Architectural and Urban Generation** — Buildings, cities, roads
4. **Biological and Organic Forms** — Fractals, corals, organs
5. **Material and Texture Synthesis** — Procedural bark, stone, metal
6. **Instancing and Distribution** — Efficient replication across vast scenes
7. **Animation and Growth** — Wind, sway, seasonal changes

### The SpeedTree Opportunity

SpeedTree is the industry-standard tool for procedural vegetation in games and film. At its core, SpeedTree is:

- **A graph-driven procedural system** that outputs high-quality geometry
- **A hierarchical generative grammar** (L-systems + custom rules)
- **A noise/random field system** (for variation)
- **A growth simulation** (approximate, not biological)
- **A parametric graph compiler** → mesh generator
- **An animation metadata generator** (wind, sway)

**This is exactly the type of system Morphogen can express cleanly and exceed in generality.**

SpeedTree demonstrates a critical truth: **procedural generation is a domain-specific version of what Morphogen wants to generalize**.

### Why Morphogen Can Do Better

SpeedTree is specialized for vegetation. Morphogen can provide:

1. **True physics-based growth** — Combine AO, light, wind, resource competition
2. **Unified procedural + simulation** — Generate branches → simulate sway → compute occlusion → feed back into growth
3. **GPU-accelerated real-time generation** — 100,000+ trees per frame
4. **Cross-domain integration** — Trees + fluids, trees + acoustics, trees + terrain + ecology
5. **General-purpose procedural framework** — Not just trees, but cities, terrains, fractals, organs, materials

---

## Decision

**We will add a ProceduralDomain to Morphogen as a core domain (v0.11-v1.0).**

The ProceduralDomain will provide a unified framework for synthesizing complex structures through:

### 1. Generative Grammars (Extended L-Systems)
- **Enhanced L-systems** — Beyond EmergenceDomain's basic L-systems
- **Parametric rules** — Context-sensitive, stochastic
- **Conditional growth** — Environment-aware branching
- **Grammar → Geometry** — Direct compilation to meshes

### 2. Branching and Tree Structures
- **Branch algorithms** — Space colonization, tropism, collision avoidance
- **Spline-based branches** — Smooth curves, controllable thickness
- **Branch meshing** — Cylinder sweeps, bark texture mapping
- **Pruning and optimization** — Depth limits, aesthetic control

### 3. Noise and Stochastic Fields
- **Noise operators** — Perlin, Simplex, FBM, Worley, Curl
- **Turbulence and variation** — Bark displacement, branch curvature
- **Random distribution** — Leaf scatter, forest placement
- **Seeded randomness** — Deterministic variation

### 4. Foliage and Instancing
- **Leaf scatter** — Density-based distribution on branches
- **Instance rendering** — Efficient forests (100k+ instances)
- **Billboard optimization** — LOD system
- **Procedural leaves** — Parametric shapes, variation

### 5. Material and Palette Systems
- **Procedural materials** — Bark, stone, metal generators
- **Color palettes** — Height-based, curvature-based gradients
- **Texture synthesis** — Noise-driven material properties
- **PBR integration** — Roughness, metallic, normal maps

### 6. Terrain and Landscape
- **Heightfield generation** — Multi-octave noise, fractal terrains
- **Erosion simulation** — Hydraulic, thermal, wind erosion
- **Biome distribution** — Temperature, moisture, altitude mapping
- **Terrain → Vegetation** — Auto-placement based on slope, moisture

### 7. Wind and Animation
- **Wind simulation** — Simple sway, turbulent flow
- **Branch dynamics** — Weight-based bending, resonance modes
- **Leaf flutter** — Parametric oscillation
- **Seasonal changes** — Color transitions, leaf fall

### 8. Architectural and Urban
- **Building generation** — Parametric facades, floor layouts
- **City networks** — Road graphs, lot subdivision
- **Modular assembly** — Reusable components
- **Procedural interiors** — Room layouts, furniture placement

---

## Why ProceduralDomain Fits Morphogen

### 1. Natural Operator Composition

Procedural generation is fundamentally about **composing simple rules into complex outputs** — exactly Morphogen's strength:

```morphogen
// Natural composition
procedural:
  - id: birch_tree
    steps:
      - lsystem.expand: {axiom: "A", rules: {A: "AB", B: "A[+A][-A]"}, iterations: 7}
      - branches.prune: {depth: 12}
      - branches.randomize_angles: {noise: simplex3d(freq=1.0)}
      - branches.to_mesh: {radius_fn: "exp(-0.1 * depth)"}
      - foliage.scatter_on_branches: {density: 0.3}
      - wind.simple_sway: {strength: 0.5}
      - palette.map_height: {gradient: "greens"}
```

This outputs a fully meshed, textured, animated tree — in ~15 lines of YAML.

---

### 2. Cross-Domain Workflows Unlock Innovation

**Example 1: Physics-Based Tree Growth**
```morphogen
// True biomechanical simulation
let tree = lsystem.expand(axiom="F", iterations=8)
let branches = branches.to_splines(tree)

// Simulate phototropism (light seeking)
let light_field = field.directional_light(direction=vec3(0, 1, 0))
branches = growth.tropism(branches, field=light_field, weight=0.3)

// Apply gravity bending
branches = growth.gravity(branches, weight=0.1)

// Check structural integrity
let stress = physics.stress_test(branches, wind_load=100N)
branches = branches.reinforce(stress_threshold=10MPa)

// Generate final mesh
let mesh = branches.to_mesh(radius_fn="structural")
```

**This is impossible in SpeedTree.**

---

**Example 2: Terrain + Ecology + Audio**
```morphogen
scene ProceduralWorld {
    // Generate terrain
    let terrain = terrain.fractal(size=1000m, octaves=8)
    terrain = terrain.erode(iterations=100, type="hydraulic")

    // Distribute biomes
    let moisture = noise.perlin2d(terrain, scale=100m)
    let temp = noise.gradient(terrain, direction="altitude")
    let biomes = terrain.biome_map(moisture, temp)

    // Populate vegetation
    let forest = vegetation.distribute(
        terrain,
        density_map = biomes.mask("forest"),
        species = [oak_tree, pine_tree, birch_tree],
        count = 10000
    )

    // Wind through trees
    let wind_field = fluid.wind(velocity=vec3(5, 0, 0))
    forest = vegetation.apply_wind(forest, wind_field)

    // Acoustic propagation through forest
    let sound_source = acoustic.point_source(position=vec3(0, 2, 0))
    let scattered = acoustic.forest_scattering(
        source = sound_source,
        trees = forest,
        frequency_range = (100Hz, 8000Hz)
    )

    out audio = acoustic.listener(scattered, position=vec3(100, 2, 0))
    out visual = visual.render_terrain(terrain, vegetation=forest)
}
```

**No existing tool can do this workflow.**

---

**Example 3: City Generation + Traffic Simulation**
```morphogen
// Generate road network
let major_roads = urban.highways(city_bounds, density=0.1)
let minor_roads = urban.subdivide_blocks(major_roads, block_size=100m)

// Generate buildings
let lots = urban.lot_subdivision(minor_roads)
let buildings = lots.map(|lot| {
    urban.procedural_building(
        lot,
        height = noise.simplex2d(lot.center) * 50m,
        style = lot.zone_type
    )
})

// Traffic simulation
let traffic = urban.traffic_sim(
    roads = minor_roads,
    vehicles = 1000,
    destinations = buildings.filter("commercial")
)

// Acoustic city simulation
let noise_map = acoustic.city_noise(traffic, buildings)
out audio = acoustic.binaural_city(noise_map, listener_pos=vec3(50, 1.7, 50))
```

---

### 3. GPU Acceleration Is Natural

All procedural operators are massively parallel:

**L-Systems:**
- String evolution → parallel character replacement
- Branch generation → independent branch processing

**Noise:**
- Each sample is independent
- Perfectly parallelizable on GPU

**Meshing:**
- Cylinder sweeps → parallel per-branch
- Instancing → GPU instancing APIs

**Distribution:**
- 10M+ instances with GPU instancing
- Real-time LOD switching

---

### 4. Determinism Is Achievable

**Strict Determinism:**
- L-system expansion (deterministic grammar)
- Seeded noise (deterministic random fields)
- Branch algorithms (stable ordering + deterministic RNG)

**Reproducible Determinism:**
- Erosion simulation (iterative with fixed tolerance)
- Growth algorithms (deterministic RNG, stable force ordering)

**This matches Morphogen's determinism profiles perfectly.**

---

### 5. Unique Differentiator

**Existing tools:**
- **SpeedTree:** Vegetation only, offline tool, no physics integration
- **Houdini:** Powerful but non-deterministic, steep learning curve, expensive
- **Blender Geometry Nodes:** Great for modeling, not physics-integrated
- **Unity/Unreal ProBuilder:** Game-specific, not general-purpose
- **Substance Designer:** Materials only, not full 3D generation

**Morphogen + ProceduralDomain:**
- ✅ Trees + terrains + cities + materials in one system
- ✅ Cross-domain integration (Geometry, Physics, Audio, Optimization)
- ✅ GPU acceleration
- ✅ Deterministic execution
- ✅ MLIR-based compilation
- ✅ Type + unit safety
- ✅ Real-time generation (100k+ instances)
- ✅ YAML-based declarative syntax

**No competitor offers this.**

---

## Consequences

### Positive

1. **New User Communities**
   - Game developers (procedural content)
   - Film/VFX artists (environment generation)
   - Architects (procedural design)
   - Landscape architects
   - Generative artists
   - 3D printing enthusiasts

2. **Novel Research Applications**
   - Physics-based procedural growth
   - Biomechanical tree simulation
   - Acoustic landscape modeling
   - Climate-driven vegetation distribution
   - Urban planning optimization

3. **Portfolio Projects**
   - "Morphogen Trees" showcase (rival SpeedTree)
   - Procedural city generator
   - Acoustic forest simulation
   - Terrain + ecology + weather system
   - Procedural architectural design tool

4. **Commercial Potential**
   - Game studios need procedural tools
   - Film industry needs large-scale environments
   - Architecture firms need parametric design
   - SpeedTree is expensive — Morphogen could be open alternative

5. **Academic Impact**
   - "Morphogen: A Unified Platform for Procedural Generation and Multi-Physics Simulation"
   - First deterministic, GPU-accelerated procedural platform
   - Cross-domain workflows unprecedented in literature

---

### Negative / Risks

1. **Implementation Complexity**
   - 8 sub-domains (grammars, branching, noise, foliage, materials, terrain, wind, urban)
   - ~100+ new operators
   - Complex meshing algorithms (cylinder sweeps, LOD generation)
   - Extensive testing (determinism, visual quality, performance)

   **Mitigation:** Phased rollout (v0.11: Grammars + Branching + Noise, v0.12: Terrain + Materials, v1.0: Urban + Animation)

2. **Performance Expectations**
   - Large scenes (100k+ trees) → memory bandwidth
   - Real-time LOD → complex GPU management
   - Meshing operations → CPU/GPU transfer overhead

   **Mitigation:** GPU instancing, aggressive LOD, streaming geometry

3. **Documentation Burden**
   - Each sub-domain needs specification, examples, tutorials
   - Cross-domain examples require deep domain knowledge
   - Visual results need gallery/showcase

   **Mitigation:** Incremental documentation, example-driven approach, screenshot gallery

4. **Scope Creep**
   - Users may request every possible procedural feature
   - Quality bar is high (must rival SpeedTree)
   - Animation and wind physics can get very complex

   **Mitigation:** Start with core algorithms, extensible operator registry, focus on quality over quantity

5. **Competition with Established Tools**
   - SpeedTree has 20+ years of development
   - Houdini has massive procedural ecosystem
   - Blender Geometry Nodes is rapidly improving

   **Mitigation:** Focus on unique value prop (cross-domain integration, determinism, real-time GPU, YAML simplicity)

---

## Alternatives Considered

### Alternative 1: Don't Add ProceduralDomain

**Pros:**
- Less implementation work
- Narrower scope (stay focused on physics/audio)

**Cons:**
- ❌ Misses huge creative coding community
- ❌ Limits Morphogen to "serious engineering" (excludes game devs, artists, architects)
- ❌ No compelling "wow factor" demo (trees are universally impressive)
- ❌ Competitors (Houdini, Blender) dominate procedural space
- ❌ EmergenceDomain L-systems are too basic for production use

**Rejected:** ProceduralDomain is too valuable for creative + commercial applications.

---

### Alternative 2: Integrate SpeedTree as Library

**Pros:**
- Leverage existing, proven tree generation
- Faster time-to-market
- Industry-standard quality

**Cons:**
- ❌ SpeedTree is proprietary and expensive
- ❌ No cross-domain integration (can't feed to Morphogen physics)
- ❌ No determinism guarantees
- ❌ Can't extend to terrains, cities, materials
- ❌ Dependency hell (licensing, versioning)

**Rejected:** Breaks Morphogen's unified compilation model and open philosophy.

---

### Alternative 3: ProceduralDomain as Plugin

**Pros:**
- Optional for users who don't need it
- Doesn't bloat core

**Cons:**
- ❌ Fragments ecosystem (core vs. plugin)
- ❌ Harder to maintain cross-domain integration
- ❌ Less polished than built-in domain
- ❌ Noise and palette systems needed by other domains

**Rejected:** Procedural generation is too fundamental (affects Geometry, Physics, Visualization, Materials).

---

### Alternative 4: Rely on EmergenceDomain L-Systems

**Pros:**
- L-systems already implemented in EmergenceDomain
- No additional work needed

**Cons:**
- ❌ EmergenceDomain L-systems are basic (string rewriting only)
- ❌ No branch meshing, no foliage scatter, no wind
- ❌ No terrain generation
- ❌ No materials or textures
- ❌ Can't rival SpeedTree in quality

**Rejected:** EmergenceDomain L-systems are a foundation, but ProceduralDomain needs 100x more capability.

---

## Implementation Plan

### Phase 1: Core Infrastructure (v0.11)
- [ ] Spline types (Bezier, Catmull-Rom, B-spline)
- [ ] Branch hierarchy representation
- [ ] Mesh generation utilities (cylinder sweeps, caps)
- [ ] Noise operator foundation (Perlin, Simplex, FBM)

### Phase 2: Grammar and Branching (v0.11)
- [ ] `lsystem.parametric` (context-sensitive rules)
- [ ] `lsystem.stochastic` (probabilistic rules)
- [ ] `branches.from_lsystem`
- [ ] `branches.to_splines`
- [ ] `branches.to_mesh`
- [ ] `branches.randomize_angles`
- [ ] `branches.prune`
- [ ] Tests: determinism, grammar correctness
- [ ] Example: basic tree generation

### Phase 3: Growth Algorithms (v0.11)
- [ ] `growth.space_colonization` (attraction points)
- [ ] `growth.tropism` (directional bias)
- [ ] `growth.gravity` (bending under weight)
- [ ] `growth.collision_avoidance`
- [ ] Tests: convergence, determinism
- [ ] Example: realistic tree with phototropism

### Phase 4: Noise and Variation (v0.11)
- [ ] `noise.perlin2d/3d`
- [ ] `noise.simplex2d/3d`
- [ ] `noise.fbm` (fractal Brownian motion)
- [ ] `noise.worley` (cellular/Voronoi)
- [ ] `noise.curl3d` (divergence-free fields)
- [ ] `random.seed`, `random.uniform`, `random.normal`
- [ ] Tests: determinism, statistical properties
- [ ] Example: varied forest (each tree unique)

### Phase 5: Foliage and Instancing (v0.12)
- [ ] `foliage.scatter_on_branches`
- [ ] `foliage.align_to_normal`
- [ ] `foliage.instance_mesh`
- [ ] `instancing.create` (GPU instancing)
- [ ] `instancing.lod` (level-of-detail switching)
- [ ] Tests: performance (100k+ instances), determinism
- [ ] Example: dense forest rendering

### Phase 6: Materials and Palettes (v0.12)
- [ ] `palette.gradient` (color gradients)
- [ ] `palette.map_height` (altitude-based coloring)
- [ ] `palette.map_curvature`
- [ ] `material.bark` (procedural bark texture)
- [ ] `material.stone`, `material.wood`, `material.metal`
- [ ] `material.pbr` (roughness, metallic, normal maps)
- [ ] Tests: visual regression tests
- [ ] Example: realistic tree with bark and leaf materials

### Phase 7: Terrain Generation (v0.12)
- [ ] `terrain.fractal` (multi-octave noise)
- [ ] `terrain.erode` (hydraulic, thermal erosion)
- [ ] `terrain.biome_map` (moisture + temperature → biomes)
- [ ] `terrain.to_mesh` (heightfield → geometry)
- [ ] Tests: determinism, erosion convergence
- [ ] Example: realistic terrain with rivers and mountains

### Phase 8: Wind and Animation (v1.0)
- [ ] `wind.simple_sway` (sinusoidal oscillation)
- [ ] `wind.turbulent` (noise-based wind field)
- [ ] `wind.branch_weighting` (mass-based bending)
- [ ] `wind.apply_to_mesh` (vertex animation)
- [ ] Integration with FluidDomain (true wind simulation)
- [ ] Tests: animation smoothness, determinism
- [ ] Example: forest swaying in wind

### Phase 9: Architectural and Urban (v1.0)
- [ ] `urban.road_network` (graph generation)
- [ ] `urban.lot_subdivision`
- [ ] `urban.procedural_building` (parametric facades)
- [ ] `arch.modular_assembly` (component-based)
- [ ] Tests: connectivity, determinism
- [ ] Example: procedural city block

### Phase 10: Cross-Domain Examples (v1.0)
- [ ] Procedural → Geometry (tree → mesh, terrain → surface)
- [ ] Procedural → Physics (wind simulation → branch bending)
- [ ] Procedural → Acoustics (forest → acoustic scattering)
- [ ] Procedural → Optimization (tree growth → structural optimization)
- [ ] Portfolio project: "Morphogen Trees" — rival SpeedTree showcase

---

## Success Metrics

**Technical:**
- [ ] 100+ procedural operators implemented
- [ ] 100% determinism for strict-tier operators
- [ ] GPU acceleration (10x+ speedup vs. CPU)
- [ ] Real-time generation (100k+ tree instances at 60 FPS)
- [ ] Visual quality comparable to SpeedTree

**Creative:**
- [ ] 20+ cross-domain examples
- [ ] Gallery of procedural outputs (trees, terrains, cities)
- [ ] 3D-printable procedural models
- [ ] Video showcase: forest → terrain → city

**Community:**
- [ ] "Morphogen Trees" becomes signature demo
- [ ] Game developers adopt Morphogen for procedural content
- [ ] Film studios evaluate for environment generation
- [ ] Academic papers cite ProceduralDomain

---

## Related Decisions

- **ADR-004:** Emergence Domain (Cross-Domain Patterns)
  - ProceduralDomain extends EmergenceDomain L-systems with full production-ready features
- **ADR-002:** Cross-Domain Architectural Patterns
  - ProceduralDomain follows same patterns for integration with Geometry, Physics, Audio
- **../specifications/geometry.md:** Geometry domain
  - Procedural → Geometry is critical integration point (splines → meshes)
- **../specifications/physics-domains.md:** Physics domains
  - Procedural → Physics enables wind simulation, structural optimization
- **../specifications/emergence.md:** Emergence domain
  - ProceduralDomain builds on L-systems foundation from EmergenceDomain

---

## References

### Academic

1. **Lindenmayer, A.** "Mathematical models for cellular interactions in development" (1968) — L-systems
2. **Prusinkiewicz, P. & Lindenmayer, A.** "The Algorithmic Beauty of Plants" (1990) — L-systems for vegetation
3. **Runions, A. et al.** "Modeling Trees with a Space Colonization Algorithm" (2007) — Space colonization
4. **Perlin, K.** "An Image Synthesizer" (1985) — Perlin noise
5. **Worley, S.** "A Cellular Texture Basis Function" (1996) — Worley/cellular noise
6. **Musgrave, F.K. et al.** "The Synthesis and Rendering of Eroded Fractal Terrains" (1989) — Terrain generation
7. **Parish, Y. & Müller, P.** "Procedural Modeling of Cities" (2001) — Urban generation

### Tools

- **SpeedTree** (industry-standard vegetation tool)
- **Houdini** (procedural modeling platform)
- **Blender Geometry Nodes** (node-based procedural modeling)
- **Substance Designer** (procedural material authoring)
- **Gaea** (terrain generation)

### Morphogen Specs

- **../specifications/procedural-generation.md** — Complete domain specification
- **../reference/procedural-operators.md** — Operator catalog

---

## Decision Outcome

**✅ APPROVED for v0.11-v1.0 implementation**

**Rationale:**
1. Natural fit for Morphogen's operator model (composition of simple rules)
2. Unique cross-domain integration (no competitor offers this)
3. Unlocks major user communities (game devs, film artists, architects)
4. GPU-accelerable, deterministic, composable
5. Phased rollout mitigates implementation risk
6. "Morphogen Trees" can rival SpeedTree in quality while offering far more generality
7. Opens path to commercial adoption (game studios, film industry)

**Next Steps:**
1. ✅ Create ADR-008 (this document)
2. ✅ Create ../specifications/procedural-generation.md (completed)
3. ✅ Create ../reference/procedural-operators.md (completed)
4. ⏳ Update ../architecture/domain-architecture.md (add ProceduralDomain to Next-Wave tier)
5. ⏳ Create example: L-system tree → mesh → visualization
6. ⏳ Implement Phase 1 (v0.11): Core infrastructure + Grammars + Branching
7. ⏳ Implement Phases 2-10 (v0.11-v1.0)

---

**Authors:** Claude (AI assistant), User (domain expert)
**Review Status:** Pending community review
**Implementation Target:** v0.11 (Grammars + Branching + Noise), v0.12 (Terrain + Materials), v1.0 (Urban + Animation)

---

**End of ADR-005**
