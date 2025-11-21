# ADR-011: Project Renaming - Morphogen & Philbrick

**Status:** Accepted
**Date:** 2025-11-16
**Authors:** Strategic Planning
**Supersedes:** Portions of ADR-010 (Ecosystem Branding)

---

## Context

As of November 2025, we have two interconnected projects:

1. **Morphogen** - Digital temporal programming language (23 domains, 80K+ lines)
2. **Analog Platform** - Modular hardware computation vision (documentation only)

### The Strategic Question

While developing ADR-010 (Ecosystem Branding & Naming Strategy), we realized these projects:
- Share the same philosophical foundation (4 primitives: sum, integrate, nonlinearity, events)
- Implement the same computational vision across different substrates (digital vs. analog)
- Mirror each other architecturally (software/hardware duals)
- But had inconsistent naming that didn't honor this relationship

### The Triggering Insight

The analog platform documentation (PR #85) revealed a profound connection:
> "You haven't built two separate projects. You've built two reflections of the same deep architecture in different media."

This raised the question: **Should our naming reflect this unified vision?**

### Key Constraints

1. **Morphogen has momentum** - 88 PRs, established codebase, documentation
2. **Education market is primary** - GTM requires credibility and clarity
3. **Hardware doesn't exist yet** - Analog platform is pure vision (1,810 lines of docs)
4. **Need clear separation** - Different markets, different timelines
5. **Historical grounding matters** - Educational value in honoring computational pioneers

---

## Decision

We are renaming both projects to honor the computational pioneers whose work created them:

### **Digital Platform: Morphogen** (was Morphogen)

Named after **Alan Turing's morphogenesis** - his pioneering work on pattern formation through reaction-diffusion equations.

**Rationale:**
- Turing's morphogenesis directly aligns with our vision: simple continuous-time rules → emergent complexity
- Emphasizes emergence and pattern formation (our unique value proposition)
- Honors Turing's lesser-known but profound work on continuous dynamics
- Avoids collision with Turing Award, Alan Turing Institute, other "Turing" products
- Creates unique positioning: "emergence-focused continuous-time computation"

### **Analog Platform: Philbrick**

Named after **George A. Philbrick** (1913-1974) - inventor of modular analog computing blocks (1952).

**Rationale:**
- Philbrick literally invented what we're rebuilding: plug-in modular analog computation
- Direct historical lineage and legitimacy
- Unique name (zero collision)
- Educational value: teaches forgotten computing history
- "Philbricks" (modules) creates product category like "LEGO bricks"

### **The Modules: Philbricks**

Individual function blocks in the Philbrick platform are called **Philbricks**.

**Rationale:**
- Natural shortening: "Philbrick modules" → "Philbricks"
- Creates product category (like LEGO bricks)
- Memorable and unique
- Works grammatically: "I need three Philbricks for this patch"

---

## Alternatives Considered

### Alternative 1: Keep "Morphogen" (Status Quo)

**Pros:**
- No disruption to existing momentum
- Established brand (88 PRs, documentation)
- Avoid rename complexity

**Cons:**
- "Morphogen" is an invented word with no meaning
- Misses opportunity for historical grounding
- Doesn't reflect unified vision with analog platform
- No educational narrative

**Verdict:** Rejected - Misses opportunity for meaningful branding

---

### Alternative 2: "Turing" for Digital Platform

**Pros:**
- Direct honor to Turing
- Immediate name recognition
- Prestigious associations (Turing Award, Alan Turing Institute)

**Cons:**
- Heavy collision: Turing Award, Alan Turing Institute, Turing Test, Turing Pharmaceuticals, Turing Phone, multiple AI startups
- Trademark concerns
- Less unique positioning
- Fights in crowded "Turing" namespace

**Verdict:** Rejected - Too much collision

---

### Alternative 3: "Mathison" (Turing's Middle Name)

**Pros:**
- Still honors Turing directly (Alan Mathison Turing)
- Zero collision
- Beautiful name (sounds mathematical + sonic)
- Clear Turing connection when explained
- Professional tone

**Cons:**
- Emphasizes precision/rigor over emergence
- Less immediately recognizable
- Requires explanation ("It's Turing's middle name")
- Doesn't capture the emergence philosophy

**Verdict:** Strong alternative, but "Morphogen" better captures our unique value proposition

---

### Alternative 4: "Bletchley" (Location Homage)

**Pros:**
- Strong Turing association (Bletchley Park)
- Historical significance
- Unique name

**Cons:**
- Associated with cryptography/secrecy (not our domain)
- Already a museum/tourist site
- Less direct Turing connection

**Verdict:** Rejected - Wrong domain association

---

### Alternative 5: Keep Separate Brands (No Unified Naming)

**Pros:**
- Maximum independence
- Each project can succeed/fail separately
- Simpler to explain

**Cons:**
- Loses "mirror image" narrative
- Misses educational opportunity
- No unified vision story

**Verdict:** Rejected - Unified Pantheon naming is more powerful

---

## Rationale

### Why "Morphogen" Wins

**1. Conceptually Perfect Alignment**

Turing's morphogenesis work (1952) showed how:
- Simple continuous-time differential equations
- Create complex emergent patterns
- Through local interactions and diffusion
- With no central control

**This is exactly what our platform does:**
- Four simple primitives (sum, integrate, nonlinearity, events)
- Compose into complex cross-domain behaviors
- Through continuous-time dynamics
- Via modular composition

The name **describes the system architecture**.

**2. Unique Market Positioning**

Nobody else positions as "emergence-focused continuous-time computation."

**Traditional tools:**
- MATLAB → "engineering computation"
- Mathematica → "symbolic mathematics"
- Julia → "high-performance numerical computing"

**Morphogen:**
- "Emergent continuous-time computation"
- Owns the emergence narrative
- Category-creating, not category-competing

**3. Educational Narrative**

For Education & Academia GTM:
- Teaches Turing's morphogenesis (often overlooked in CS curricula)
- Connects biology, mathematics, and computation
- Shows continuous-time dynamics produce patterns
- Reaction-diffusion as gateway to complex systems

Students learn computational history while learning the tool.

**4. Technical Accuracy**

Key Morphogen features map directly to morphogenesis:
- Reaction-diffusion patterns → `field` domain
- Emergent behavior → cross-domain composition
- Continuous-time dynamics → temporal execution model
- Pattern formation → agent swarms, cellular automata

The name isn't marketing fluff - it's **descriptively accurate**.

**5. Emotional Resonance**

"Morphogen" evokes:
- Transformation and growth
- Natural emergence
- Beauty in mathematics
- Discovery and exploration

This attracts the right users: scientists, artists, educators who love emergence.

---

### Why "Philbrick" Wins

**1. Historical Accuracy**

George A. Philbrick:
- Invented commercial modular analog computing (K2-W op-amp module, 1952)
- Created standardized form factors for analog computation
- Built plug-in analog computing systems **decades before Moog**
- Believed electronics should be compositional blocks, not fixed circuits

**We are literally reviving his vision** but for modern makers, musicians, and researchers.

**2. Educational Legacy**

Philbrick is a forgotten pioneer. By naming our platform after him:
- We teach computing history
- We honor a deserving but overlooked inventor
- We establish historical legitimacy
- We connect to analog computing's roots

**3. "Philbricks" Product Category**

Just as LEGO created "bricks," we create "Philbricks":
- Memorable product name
- Self-explanatory category
- Natural language fit: "I need three Philbricks"
- Brand becomes the category

**4. Zero Collision**

"Philbrick" is:
- Unique in tech namespace
- Trademark-friendly
- Distinctive and memorable
- Not overused

---

## The Pantheon Layer Naming

Both platforms implement the same architecture across different substrates. We honor this by naming internal layers after the inventors who created those concepts:

### Morphogen (Digital Platform)

```
User Surfaces:
  - Morphogen.Audio (compositional audio DSL)
  - RiffStack (live performance environment)

Domain Libraries:
  - field, agent, audio, rigid, linalg...
  - (follows ADR-010 single-word lowercase convention)

Kernel (morphogen.internal):
  - Wiener Scheduler (Norbert Wiener - cybernetics, control theory)
  - Shannon Protocol (Claude Shannon - information theory)
  - Turing Core (Alan Turing - universal computation)
```

### Philbrick (Analog Platform)

```
User Facing:
  - Philbrick Rack (the complete system)
  - Moog Surface (Robert Moog - voltage-controlled modularity)

Modules:
  - Philbricks (individual function blocks)
  - Sum, Integrate, Nonlinearity, Events

Substrate:
  - DeForest Layer (Lee de Forest - amplification)
  - Black Layer (Harold Black - negative feedback)
  - Shannon Bus (Claude Shannon - digital protocol)

Compute:
  - Mead Processors (Carver Mead - neuromorphic computing)
```

**Rationale:**
- Makes architectural layers self-documenting
- Educational value (each layer teaches history)
- Honors the inventors whose work created each concept
- Creates "computational mythology" rooted in real history

---

## Consequences

### Positive

✅ **Unique Positioning** - "Emergence-focused continuous-time computation" (Morphogen) is unclaimed market space

✅ **Historical Grounding** - Both names honor actual pioneers, creating legitimacy

✅ **Educational Value** - Names teach computational history (supports Education & Academia GTM)

✅ **Clear Separation** - Morphogen (software) and Philbrick (hardware) are obviously different projects

✅ **Unified Vision** - Pantheon layer naming shows they're architectural mirrors

✅ **Product Category Creation** - "Philbricks" creates self-contained product category

✅ **Emotional Resonance** - "Morphogenesis" and "modular analog computing" are compelling narratives

✅ **Trademark Friendly** - Both names appear available and distinctive

---

### Negative

⚠️ **Kills Morphogen Brand** - Loses 88 PRs worth of brand recognition (mitigated: pre-v1.0, no paying customers)

⚠️ **Rename Complexity** - All code, docs, repos, imports must change (significant effort)

⚠️ **External Link Breakage** - Any external references to "Morphogen" break (mitigated: GitHub auto-redirects)

⚠️ **Learning Curve** - Multiple historical names to learn (mitigated: educational context makes this a feature)

⚠️ **Requires Explanation** - Must explain morphogenesis connection (mitigated: great conversation starter)

⚠️ **Risk of Pretentiousness** - Could feel academic/overwrought if not executed well (mitigated: focus on substance over style)

---

### Neutral

◼️ **Pre-v1.0 Timing** - Best time to rename is before production users (now)

◼️ **Philbrick Hardware Doesn't Exist** - Naming vision before implementation (establishes direction)

◼️ **Two Brands to Manage** - More complex than one, but enables clear separation

---

## Implementation Plan

### Phase 1: Create Philbrick Repository (Week 1)

**Tasks:**
1. Create new GitHub repo: `scottsen/philbrick`
2. Move `docs/analog-platform/` → `philbrick/docs/`
3. Create Philbrick README.md with vision and cross-reference to Morphogen
4. Update all internal docs: "analog platform" → "Philbrick", "modules" → "Philbricks"
5. Rename `06-KAIRO-BRIDGE.md` → `MORPHOGEN-BRIDGE.md`

**Deliverables:**
- [ ] Philbrick repository created
- [ ] Documentation moved and updated
- [ ] Cross-references established

---

### Phase 2: Prepare Morphogen Rename (Week 2)

**Tasks:**
1. Create comprehensive migration guide
2. Create find/replace checklist for all files
3. Update ADR-010 to reflect Morphogen/Philbrick decision
4. Draft new README.md with Morphogen vision
5. Update SPECIFICATION.md with Morphogen language spec
6. Plan import path changes (`kairo` → `morphogen`)

**Deliverables:**
- [ ] Migration guide complete
- [ ] Documentation drafts ready
- [ ] Checklist verified

---

### Phase 3: Execute Morphogen Rename (Week 3)

**Tasks:**
1. Rename GitHub repo: `kairo` → `morphogen`
2. Update all Python package names: `kairo` → `morphogen`
3. Update all imports across codebase
4. Update all documentation (README, STATUS, CHANGELOG, specs, guides)
5. Update examples and tests
6. Implement Pantheon layer names (Wiener Scheduler, Shannon Protocol)
7. Create git tag: `v0.11.0-morphogen-rename`

**Deliverables:**
- [ ] Repository renamed
- [ ] All code updated
- [ ] All docs updated
- [ ] Tests passing
- [ ] v0.11.0 tagged

---

### Phase 4: Update Branding (Week 4)

**Tasks:**
1. Create Morphogen tagline and positioning
2. Create Philbrick tagline and positioning
3. Update cross-reference documentation
4. Create MORPHOGEN-PHILBRICK-BRIDGE.md
5. Update CONTRIBUTING.md with new names
6. Announce rename with rationale

**Deliverables:**
- [ ] Brand positioning documented
- [ ] All cross-references updated
- [ ] Announcement prepared

---

## Success Metrics

### Immediate (3 months)

- [ ] Morphogen repository renamed successfully
- [ ] All tests passing post-rename
- [ ] Documentation comprehensively updated
- [ ] Philbrick repository established
- [ ] Zero confusion about which project is which

### Short-term (6 months)

- [ ] Morphogen v1.0 shipped with new branding
- [ ] Educational materials reference morphogenesis concepts
- [ ] Users understand and appreciate historical naming
- [ ] Philbrick hardware prototyping begins (optional)

### Long-term (1-2 years)

- [ ] Morphogen recognized for emergence-focused positioning
- [ ] "Philbricks" becomes recognized product category
- [ ] Educational institutions adopt due to historical grounding
- [ ] Cross-substrate compilation (Morphogen → Philbrick routing) possible

---

## Rollback Plan

If the rename creates insurmountable problems:

**Reversal is possible before v1.0:**
1. Revert GitHub repo name: `morphogen` → `kairo`
2. Revert package names: `morphogen` → `kairo`
3. Keep Philbrick (it's already separate)
4. Maintain Pantheon layer names internally (still valuable)

**Cost of rollback:** High but manageable pre-v1.0. After v1.0 and paying customers, reversal becomes impractical.

**Decision point:** If we're going to rename, do it **now** (pre-v1.0).

---

## References

### Historical Sources

- Turing, A. M. (1952). "The Chemical Basis of Morphogenesis". *Philosophical Transactions of the Royal Society of London B*, 237(641): 37-72
- Philbrick, G. A. (1952). "K2-W Computing Amplifier" (first modular analog computing block)
- Philbrick Researches, Inc. product catalogs (1950s-1960s)

### Internal Documents

- ADR-010: Ecosystem Branding & Naming Strategy
- PR #85: Analog Platform Vision Documentation (1,810 lines)
- PR #88: Ecosystem Branding Strategy (Pantheon layer naming)
- `docs/analog-platform/01-PANTHEON.md` (historical lineage)
- `docs/analog-platform/06-KAIRO-BRIDGE.md` (software/hardware duality)

### Market Analysis

- PR #64: Strategic Domain Value Analysis (Education & Academia = #1 GTM)
- Session siromodu-1115: Strategic direction (infrastructure vs. expansion)

---

## Appendix: The Unified Tagline

**Morphogen** and **Philbrick**: Computational substrates for the continuous world.

**Morphogen:**
> Where emergence meets engineering. Named after Turing's morphogenesis, Morphogen is a deterministic continuous-time language where simple primitives compose into complex patterns.

**Philbrick:**
> Modular analog computation. Named after George A. Philbrick, who invented modular analog computing in 1952. Philbricks are composable function blocks for continuous-time signal processing.

**Together:**
> Two projects, one vision. Simple primitives (sum, integrate, nonlinearity, events) compose into emergent complexity - in software and hardware.

---

**Decision Date:** 2025-11-16
**Approved By:** Project Leadership
**Implementation Start:** 2025-11-17
**Target Completion:** 2025-12-15 (4 weeks)
