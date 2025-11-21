# Procedural Generation Operators Catalog

**Version:** 1.0
**Last Updated:** 2025-11-15
**Related:** ADR-008, ../specifications/procedural-generation.md

---

## Overview

This document catalogs all operators in the **ProceduralDomain**, organized by subdomain. The ProceduralDomain enables synthesis of complex structures (trees, terrains, cities, materials) through rule-based, noise-driven, and grammar-based generation.

**Implementation Target:** v0.11-v1.0

---

## Operator Summary by Subdomain

| Subdomain | Operator Count | Status | Target Version |
|-----------|----------------|--------|----------------|
| **Generative Grammars** | 10 | Proposed | v0.11 |
| **Branching Algorithms** | 12 | Proposed | v0.11 |
| **Noise and Stochastic** | 11 | Proposed | v0.11 |
| **Foliage and Instancing** | 6 | Proposed | v0.12 |
| **Materials and Palettes** | 10 | Proposed | v0.12 |
| **Terrain Generation** | 8 | Proposed | v0.12 |
| **Wind and Animation** | 7 | Proposed | v1.0 |
| **Architectural/Urban** | 8 | Proposed | v1.0 |
| **Total** | **72** | Proposed | v0.11-v1.0 |

---

## 1. Generative Grammars (10 operators)

### 1.1 lsystem.define
**Define L-system grammar**
- **Inputs:** None
- **Params:** name, axiom, rules
- **Outputs:** LSystem
- **Determinism:** Strict
- **Layer:** 4 (Higher-level composition)

### 1.2 lsystem.expand
**Expand L-system string n iterations**
- **Inputs:** LSystem
- **Params:** iterations, seed
- **Outputs:** LSystemState
- **Determinism:** Strict
- **Use Case:** Tree generation, fractal curves

### 1.3 lsystem.parametric
**Parametric L-system with context-sensitive rules**
- **Inputs:** None
- **Params:** axiom, rules (with parameters), iterations
- **Outputs:** LSystemState
- **Determinism:** Strict
- **Use Case:** Realistic trees with parameter evolution (branch thickness decay)

### 1.4 lsystem.stochastic
**Stochastic L-system with probabilistic rules**
- **Inputs:** None
- **Params:** axiom, rules (with probabilities), iterations, seed
- **Outputs:** LSystemState
- **Determinism:** Strict
- **Use Case:** Natural variation in tree structures

### 1.5 lsystem.context_sensitive
**Context-sensitive L-system (rules depend on neighbors)**
- **Inputs:** None
- **Params:** axiom, rules (with context), iterations
- **Outputs:** LSystemState
- **Determinism:** Strict
- **Use Case:** Advanced biological modeling

### 1.6 lsystem.preset
**Load common L-system patterns**
- **Inputs:** None
- **Params:** name (koch_curve, hilbert_curve, fractal_tree, etc.)
- **Outputs:** LSystem
- **Determinism:** Strict
- **Presets:** koch_curve, hilbert_curve, sierpinski_triangle, fractal_tree, realistic_tree, bush, coral

### 1.7 lsystem.to_string
**Get current string representation**
- **Inputs:** LSystemState
- **Outputs:** String
- **Determinism:** Strict

### 1.8 lsystem.visualize
**Visualize L-system evolution (debug)**
- **Inputs:** LSystemState
- **Outputs:** Visualization
- **Use Case:** Debugging grammar evolution

### 1.9 lsystem.analyze
**Analyze grammar properties (branching factor, depth)**
- **Inputs:** LSystem
- **Outputs:** Statistics
- **Use Case:** Understanding complexity

### 1.10 lsystem.optimize
**Optimize grammar for specific properties**
- **Inputs:** LSystem, objective function
- **Outputs:** Optimized LSystem
- **Use Case:** Find grammar that produces desired shape

---

## 2. Branching Algorithms (12 operators)

### 2.1 branches.from_lsystem
**Convert L-system string to branch hierarchy**
- **Inputs:** LSystemState
- **Params:** angle, step_size, radius_decay
- **Outputs:** BranchTree
- **Determinism:** Strict

### 2.2 branches.to_splines
**Convert branches to smooth splines**
- **Inputs:** BranchTree
- **Params:** spline_type, smoothness
- **Outputs:** Array\<Spline\>
- **Determinism:** Strict
- **Use Case:** Smooth organic curves

### 2.3 branches.to_mesh
**Generate cylindrical mesh from branch tree**
- **Inputs:** BranchTree
- **Params:** radial_segments, radius_fn, bark_texture
- **Outputs:** Mesh
- **Determinism:** Strict
- **GPU:** Parallelize per-branch

### 2.4 branches.randomize_angles
**Add stochastic variation to branch angles**
- **Inputs:** BranchTree
- **Params:** noise_fn, amplitude, frequency, seed
- **Outputs:** BranchTree
- **Determinism:** Strict
- **Use Case:** Natural variation

### 2.5 branches.prune
**Remove branches by depth/length/criteria**
- **Inputs:** BranchTree
- **Params:** max_depth, min_length, criterion
- **Outputs:** BranchTree
- **Determinism:** Strict
- **Use Case:** Aesthetic control

### 2.6 branches.thickness
**Set branch thickness function**
- **Inputs:** BranchTree
- **Params:** radius_fn (depth_decay, structural, parametric)
- **Outputs:** BranchTree
- **Determinism:** Strict

### 2.7 growth.space_colonization
**Space colonization algorithm for realistic growth**
- **Inputs:** None
- **Params:** root, attractors, influence_radius, kill_radius, segment_length, iterations
- **Outputs:** BranchTree
- **Determinism:** Strict
- **Use Case:** Realistic tree crown shapes

### 2.8 growth.tropism
**Add directional growth bias (phototropism, gravitropism)**
- **Inputs:** BranchTree
- **Params:** direction, weight
- **Outputs:** BranchTree
- **Determinism:** Strict
- **Use Case:** Trees bending toward light or gravity

### 2.9 growth.gravity
**Apply gravity bending to branches**
- **Inputs:** BranchTree
- **Params:** g, density, stiffness
- **Outputs:** BranchTree
- **Determinism:** Strict
- **Use Case:** Realistic drooping branches

### 2.10 growth.collision_avoidance
**Prevent self-intersection and obstacle collision**
- **Inputs:** BranchTree
- **Params:** obstacles, self_collision, margin
- **Outputs:** BranchTree
- **Determinism:** Strict
- **Use Case:** Dense forests, urban trees

### 2.11 branches.reinforce
**Reinforce weak branches based on stress analysis**
- **Inputs:** BranchTree, stress_field
- **Params:** stress_threshold
- **Outputs:** BranchTree
- **Determinism:** Strict
- **Cross-domain:** Uses PhysicsDomain stress analysis

### 2.12 branches.animate_growth
**Animate tree growth over time**
- **Inputs:** LSystem
- **Params:** time_range, fps
- **Outputs:** AnimatedBranchTree
- **Use Case:** Growth timelapse

---

## 3. Noise and Stochastic Fields (11 operators)

### 3.1 noise.perlin2d
**2D Perlin noise**
- **Params:** position (Vec2), frequency, seed
- **Outputs:** f32 in [-1, 1]
- **Determinism:** Strict
- **GPU:** Inline, parallelize

### 3.2 noise.perlin3d
**3D Perlin noise**
- **Params:** position (Vec3), frequency, seed
- **Outputs:** f32 in [-1, 1]
- **Determinism:** Strict
- **Use Case:** Volumetric noise, 3D textures

### 3.3 noise.simplex2d
**2D Simplex noise (faster, better isotropy)**
- **Params:** position (Vec2), frequency, seed
- **Outputs:** f32 in [-1, 1]
- **Determinism:** Strict

### 3.4 noise.simplex3d
**3D Simplex noise**
- **Params:** position (Vec3), frequency, seed
- **Outputs:** f32 in [-1, 1]
- **Determinism:** Strict

### 3.5 noise.fbm
**Fractal Brownian Motion (multi-octave noise)**
- **Params:** position (Vec3), octaves, lacunarity, persistence, base_frequency, noise_type, seed
- **Outputs:** f32
- **Determinism:** Strict
- **Use Case:** Terrains, clouds, organic textures

### 3.6 noise.worley
**Worley/cellular noise (Voronoi)**
- **Params:** position (Vec3), frequency, distance_fn, seed
- **Outputs:** (f1, f2) — distances to 1st and 2nd nearest feature points
- **Determinism:** Strict
- **Use Case:** Stone textures, cellular patterns

### 3.7 noise.curl3d
**Curl noise (divergence-free vector field)**
- **Params:** position (Vec3), frequency, seed
- **Outputs:** Vec3 (curl vector)
- **Determinism:** Strict
- **Use Case:** Wood grain, marble, turbulent flow

### 3.8 noise.turbulence
**Turbulence (absolute value of noise)**
- **Params:** position (Vec3), octaves, seed
- **Outputs:** f32
- **Use Case:** Marble, chaotic patterns

### 3.9 noise.ridged
**Ridged multi-fractal (inverted, sharpened noise)**
- **Params:** position (Vec3), octaves, seed
- **Outputs:** f32
- **Use Case:** Mountain ridges, sharp features

### 3.10 random.distribute
**Randomly distribute points in volume**
- **Params:** bounds (BoundingBox/Sphere/Mesh), count, distribution (uniform/poisson_disk/blue_noise), seed
- **Outputs:** Array\<Vec3\>
- **Determinism:** Strict
- **Use Case:** Forest placement, particle scatter

### 3.11 random.seed
**Create deterministic RNG**
- **Params:** seed
- **Outputs:** RNG state
- **Determinism:** Strict

---

## 4. Foliage and Instancing (6 operators)

### 4.1 foliage.scatter_on_branches
**Scatter leaves on branch endpoints**
- **Inputs:** BranchTree
- **Params:** density, min_depth, leaf_mesh, size_range, seed
- **Outputs:** Array\<Instance\>
- **Determinism:** Strict
- **Use Case:** Add leaves to tree

### 4.2 foliage.align_to_normal
**Align foliage to branch normals**
- **Inputs:** Array\<Instance\>
- **Params:** alignment (branch_normal, up, random)
- **Outputs:** Array\<Instance\>
- **Determinism:** Strict

### 4.3 foliage.instance_mesh
**Instance single mesh at positions**
- **Inputs:** Mesh, Array\<Transform\>
- **Outputs:** InstanceCollection
- **GPU:** GPU instancing

### 4.4 instancing.create
**Create GPU instance collection**
- **Params:** base_mesh, transforms, lod_distances
- **Outputs:** InstanceCollection
- **Determinism:** Strict
- **Use Case:** Render 100k+ trees efficiently

### 4.5 instancing.lod
**Level-of-detail management**
- **Inputs:** InstanceCollection, camera, lod_levels
- **Outputs:** InstanceCollection
- **GPU:** Automatic LOD switching
- **Use Case:** Performance optimization

### 4.6 instancing.cull
**Frustum and occlusion culling**
- **Inputs:** InstanceCollection, camera
- **Outputs:** InstanceCollection
- **GPU:** GPU-based culling
- **Use Case:** Only render visible instances

---

## 5. Materials and Palettes (10 operators)

### 5.1 palette.gradient
**Define color gradient**
- **Params:** stops (Array\<(f32, Color)\>)
- **Outputs:** Gradient
- **Determinism:** Strict

### 5.2 palette.map_height
**Map height to color gradient**
- **Inputs:** Mesh, Gradient
- **Params:** height_range
- **Outputs:** Mesh (with vertex colors)
- **Determinism:** Strict
- **Use Case:** Autumn tree colors by height

### 5.3 palette.map_curvature
**Map curvature to color**
- **Inputs:** Mesh, Gradient
- **Outputs:** Mesh
- **Use Case:** Highlight bark crevices

### 5.4 palette.map_slope
**Map slope to color**
- **Inputs:** Terrain, Gradient
- **Outputs:** Terrain
- **Use Case:** Color steep slopes differently

### 5.5 material.bark
**Procedural bark texture**
- **Params:** scale, roughness, color_base, color_variation, seed
- **Outputs:** Material
- **Determinism:** Strict
- **Use Case:** Realistic tree bark

### 5.6 material.stone
**Procedural stone texture**
- **Params:** type (granite, marble, sandstone), scale, seed
- **Outputs:** Material
- **Determinism:** Strict

### 5.7 material.wood
**Procedural wood grain**
- **Params:** grain_direction, ring_frequency, turbulence, seed
- **Outputs:** Material
- **Use Case:** Wood textures, cut logs

### 5.8 material.metal
**Procedural metal texture**
- **Params:** type (brushed, scratched, rusted), seed
- **Outputs:** Material

### 5.9 material.pbr
**PBR material from maps**
- **Params:** base_color, roughness, metallic, normal
- **Outputs:** Material
- **Use Case:** Physically-based rendering

### 5.10 material.blend
**Blend materials based on mask**
- **Inputs:** Material A, Material B, mask
- **Outputs:** Material
- **Use Case:** Weathering, transitions

---

## 6. Terrain Generation (8 operators)

### 6.1 terrain.fractal
**Fractal terrain generation**
- **Params:** size, resolution, height_scale, octaves, lacunarity, persistence, seed
- **Outputs:** Terrain
- **Determinism:** Strict
- **Use Case:** Base terrain generation

### 6.2 terrain.erode
**Hydraulic/thermal erosion simulation**
- **Inputs:** Terrain
- **Params:** type (hydraulic, thermal, wind), iterations, erosion_strength, sediment_capacity
- **Outputs:** Terrain
- **Determinism:** Repro
- **Use Case:** Realistic valleys, rivers

### 6.3 terrain.biome_map
**Generate biome distribution**
- **Inputs:** Terrain
- **Params:** moisture_scale, temp_gradient, seed
- **Outputs:** BiomeMap
- **Determinism:** Strict
- **Use Case:** Forest/desert/tundra placement

### 6.4 terrain.to_mesh
**Convert heightfield to mesh**
- **Inputs:** Terrain
- **Outputs:** Mesh
- **Determinism:** Strict

### 6.5 terrain.add_features
**Add rivers, lakes, craters**
- **Inputs:** Terrain
- **Params:** features (Array of feature specs)
- **Outputs:** Terrain
- **Use Case:** Artistic control

### 6.6 terrain.smooth
**Smooth terrain (Gaussian blur)**
- **Inputs:** Terrain
- **Params:** kernel_size
- **Outputs:** Terrain
- **Use Case:** Reduce sharp noise

### 6.7 vegetation.distribute
**Distribute vegetation based on biome**
- **Inputs:** Terrain, BiomeMap
- **Params:** species (Array\<VegetationSpec\>), density_map, seed
- **Outputs:** Array\<Instance\>
- **Determinism:** Strict
- **Use Case:** Populate terrain with trees, grass

### 6.8 vegetation.apply_wind
**Animate vegetation with wind field**
- **Inputs:** Array\<Instance\>, wind_field
- **Outputs:** Array\<Instance\>
- **Cross-domain:** Uses FluidDomain wind

---

## 7. Wind and Animation (7 operators)

### 7.1 wind.simple_sway
**Simple sinusoidal sway**
- **Inputs:** BranchTree
- **Params:** strength, frequency, time
- **Outputs:** BranchTree
- **Determinism:** Strict
- **Use Case:** Basic tree animation

### 7.2 wind.turbulent
**Noise-based turbulent wind field**
- **Inputs:** BranchTree
- **Params:** base_velocity, turbulence, time, seed
- **Outputs:** BranchTree
- **Determinism:** Strict
- **Use Case:** Realistic wind variation

### 7.3 wind.branch_weighting
**Weight-based bending (heavier branches bend more)**
- **Inputs:** BranchTree
- **Params:** wind_force, depth_falloff
- **Outputs:** BranchTree
- **Determinism:** Strict

### 7.4 wind.apply_to_mesh
**Bake wind animation into vertex animation**
- **Inputs:** Mesh
- **Params:** wind_fn, time_range, fps
- **Outputs:** AnimatedMesh
- **Use Case:** Export animated mesh to game engine

### 7.5 wind.from_fluid
**Use FluidDomain wind simulation**
- **Inputs:** BranchTree, FluidField
- **Outputs:** BranchTree
- **Cross-domain:** FluidDomain → ProceduralDomain
- **Use Case:** Physics-accurate wind

### 7.6 seasonal.color_transition
**Animate seasonal color changes**
- **Inputs:** BranchTree
- **Params:** season (0=spring, 0.5=autumn, etc.), palettes
- **Outputs:** BranchTree
- **Use Case:** Seasonal animation

### 7.7 seasonal.leaf_fall
**Simulate leaf falling in autumn**
- **Inputs:** BranchTree
- **Params:** season, fall_rate
- **Outputs:** BranchTree, Array\<Particle\> (falling leaves)
- **Use Case:** Autumn animation

---

## 8. Architectural and Urban (8 operators)

### 8.1 urban.road_network
**Generate road network graph**
- **Params:** bounds, density, method (lsystem, tensor, grid), seed
- **Outputs:** Graph
- **Determinism:** Strict
- **Use Case:** City road layout

### 8.2 urban.lot_subdivision
**Subdivide city blocks into lots**
- **Inputs:** Graph (roads)
- **Params:** min_lot_size, seed
- **Outputs:** Array\<Polygon\>
- **Determinism:** Strict

### 8.3 urban.procedural_building
**Generate building on lot**
- **Params:** lot, height, style (residential, commercial, industrial), seed
- **Outputs:** Mesh
- **Determinism:** Strict
- **Use Case:** Populate city

### 8.4 arch.facade
**Generate parametric building facade**
- **Params:** wall, window_pattern, floor_height, seed
- **Outputs:** Mesh
- **Use Case:** Detailed building exteriors

### 8.5 arch.modular_assembly
**Assemble building from modular components**
- **Params:** components, layout
- **Outputs:** Mesh
- **Use Case:** Parametric architecture

### 8.6 urban.traffic_sim
**Simulate traffic flow**
- **Inputs:** Graph (roads)
- **Params:** vehicles, destinations
- **Outputs:** Array\<Vehicle\> (positions, velocities)
- **Use Case:** Acoustic city simulation

### 8.7 urban.pedestrian_paths
**Generate pedestrian walkways**
- **Inputs:** Array\<Building\>, Array\<Destination\>
- **Outputs:** Graph
- **Use Case:** Campus planning

### 8.8 urban.zoning
**Assign zones to city blocks**
- **Inputs:** Array\<Polygon\>
- **Params:** zone_rules
- **Outputs:** Array\<Zone\>
- **Use Case:** City planning

---

## Cross-Domain Integration Operators

### 9.1 procedural.to_geometry
**Convert procedural structures to GeometryDomain**
- **Inputs:** BranchTree | Terrain | Mesh
- **Outputs:** GeometryDomain.Solid
- **Cross-domain:** ProceduralDomain → GeometryDomain

### 9.2 procedural.from_field
**Generate procedural structures from field**
- **Inputs:** Field2D/3D
- **Outputs:** BranchTree | Terrain
- **Cross-domain:** FieldDomain → ProceduralDomain

### 9.3 procedural.stress_guided_growth
**Use stress analysis to guide growth**
- **Inputs:** BranchTree
- **Outputs:** Optimized BranchTree
- **Cross-domain:** PhysicsDomain (stress) → ProceduralDomain (growth)

### 9.4 procedural.acoustic_placement
**Place vegetation to optimize acoustics**
- **Inputs:** Terrain, acoustic objectives
- **Outputs:** Array\<Instance\>
- **Cross-domain:** AcousticsDomain → ProceduralDomain

---

## Operator Design Principles

### 1. Determinism
- **Strict tier:** All grammar, noise, branching operators must be bit-exact reproducible
- **Repro tier:** Erosion, optimization-based operators (stable but may vary with solver)
- **All operators require explicit `seed` parameter** for stochastic processes

### 2. GPU Acceleration
- Noise operators: inline, parallelize
- Meshing: parallelize per-branch
- Instancing: use GPU instancing APIs
- Target: 100k+ instances at 60 FPS

### 3. Composability
- Operators chain naturally: `lsystem.expand → branches.from_lsystem → branches.to_mesh`
- Cross-domain: `terrain → vegetation → wind → acoustic`

### 4. Type Safety
- Units: `f32<m>`, `f32<deg>`, `Vec3<m/s>`
- Type checking prevents errors (angle in radians vs. degrees)

### 5. Visual Quality
- Must rival SpeedTree in tree quality
- Realistic terrains (erosion essential)
- Production-ready materials (PBR support)

---

## Implementation Priorities

### Phase 1: Foundation (v0.11)
**Goal:** Basic tree generation
- L-systems (define, expand, parametric, stochastic)
- Branching (from_lsystem, to_mesh, randomize, prune)
- Noise (perlin, simplex, fbm, worley)
- Space colonization algorithm

**Deliverable:** "Morphogen Trees v1" demo (basic but functional)

---

### Phase 2: Production Quality (v0.12)
**Goal:** SpeedTree-quality trees + terrain
- Foliage (scatter, instancing, LOD)
- Materials (bark, stone, wood, PBR)
- Terrain (fractal, erode, biome_map, vegetation.distribute)
- Growth algorithms (tropism, gravity, collision_avoidance)

**Deliverable:** "Morphogen Trees v2" demo (production-ready) + "Morphogen Terrain" demo

---

### Phase 3: Animation and Urban (v1.0)
**Goal:** Complete procedural platform
- Wind (simple_sway, turbulent, from_fluid)
- Seasonal (color_transition, leaf_fall)
- Urban (roads, buildings, traffic)
- Architectural (facades, modular assembly)

**Deliverable:** "Morphogen Procedural World" demo (trees + terrain + city + acoustics)

---

## Testing Requirements

### Per-Operator Tests

**Determinism:**
```morphogen
// Bit-exact reproducibility
let result1 = operator(params, seed=42)
let result2 = operator(params, seed=42)
assert_eq!(result1, result2)
```

**Visual Regression:**
```morphogen
// Compare rendered output to reference image
let tree = procedural.birch_tree(seed=42)
let rendered = visual.render(tree)
assert_image_similar(rendered, "ref_birch.png", threshold=0.95)
```

**Performance:**
```morphogen
// Ensure real-time performance
let forest = instancing.create(tree_mesh, count=100000)
let fps = visual.benchmark(forest)
assert!(fps >= 60)
```

---

## Success Metrics

**Technical:**
- [ ] 72+ operators implemented
- [ ] 100% determinism for strict-tier operators
- [ ] GPU acceleration (10x+ speedup)
- [ ] Real-time rendering (100k+ instances at 60 FPS)

**Visual Quality:**
- [ ] Trees rival SpeedTree
- [ ] Terrains look realistic (erosion essential)
- [ ] Materials are production-ready (PBR)

**Community:**
- [ ] "Morphogen Trees" becomes signature demo
- [ ] Game developers adopt for procedural content
- [ ] Film studios evaluate for environment generation
- [ ] Academic papers cite ProceduralDomain

---

## References

### Academic
1. Prusinkiewicz & Lindenmayer — "The Algorithmic Beauty of Plants" (1990)
2. Runions et al. — "Modeling Trees with a Space Colonization Algorithm" (2007)
3. Perlin — "An Image Synthesizer" (1985)
4. Musgrave et al. — "Synthesis and Rendering of Eroded Fractal Terrains" (1989)
5. Parish & Müller — "Procedural Modeling of Cities" (2001)

### Tools
- SpeedTree
- Houdini
- Blender Geometry Nodes
- Substance Designer
- Gaea (terrain)

---

**End of Catalog**
