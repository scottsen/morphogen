# Morphogen Rename Preparation

**Date**: 2025-11-16
**Decision Reference**: ADR-011 (Project Renaming - Morphogen & Philbrick)
**Status**: Preparation Phase
**Target**: Version 0.11.0 "The Morphogen Rename"

---

## Executive Summary

**Kairo** will be renamed to **Morphogen** to better reflect its essence: emergence-focused continuous-time computation, named after Alan Turing's morphogenesis work (1952).

This document outlines all changes required for the rename and establishes linkage with the sister hardware project **Philbrick**.

---

## Why "Morphogen"?

### The Decision
From ADR-011 (wuluje-1116 session):

- **Kairo** was an invented word with no inherent meaning
- **Morphogen** references Turing's morphogenesis: simple continuous-time differential equations creating complex emergent patterns through local interactions
- **Perfect conceptual fit**: Our 4 primitives compose into emergent complexity, just like Turing's morphogens

### Strategic Value
- **Unique positioning**: "Emergence-focused continuous-time computation" - unclaimed market space
- **Educational value**: Teaches morphogenesis while teaching the platform
- **Historical grounding**: Turing's 1952 work legitimizes our approach
- **Market differentiation**: Category-creating, not category-competing

---

## The Two-Project Strategy

### Morphogen (Software) + Philbrick (Hardware)

| Aspect | Morphogen (This Project) | Philbrick (Sister Project) |
|--------|--------------------------|----------------------------|
| **Repository** | `scottsen/morphogen` | `scottsen/philbrick` |
| **Purpose** | Digital simulation of continuous phenomena | Physical embodiment of continuous dynamics |
| **Substrate** | CPU/GPU computation | Analog circuits, DSP chips, neural accelerators |
| **Primitives** | Streams, fields, transforms | Sum, integrate, nonlinearity, events |
| **Type Safety** | Domain/rate/units checking | Pin/voltage/impedance contracts |
| **Execution** | Multirate deterministic scheduler | Latency-aware routing fabric |
| **GTM** | Education & Academia (#1), Creative Tools | Musicians, Makers, Researchers, Educators |

**Shared Philosophy**: Computation = composition. Emergence from simple primitives.

---

## Rename Scope

### What Changes

#### Repository & Package Names
- **GitHub**: `scottsen/kairo` → `scottsen/morphogen`
- **Python Package**: `kairo` → `morphogen`
- **PyPI**: `kairo` (if published) → `morphogen`

#### Code & Imports
```python
# OLD
from kairo.stdlib import audio, field
import kairo.core

# NEW
from morphogen.stdlib import audio, field
import morphogen.core
```

#### User-Facing Surfaces
- **Kairo.Audio** → **Morphogen.Audio**
- RiffStack (unchanged - separate brand, but Morphogen-powered)

#### Internal Namespaces
- `kairo.internal` → `morphogen.internal`
- Pantheon layer names remain: Wiener Scheduler, Shannon Protocol, Turing Core

#### Documentation
- All docs referencing "Kairo" → "Morphogen"
- Add Philbrick cross-references
- Update vision/philosophy docs

### What Stays the Same

#### Architecture
- Three-layer model (surfaces, domains, kernel)
- 23+ implemented domains
- Cross-domain composition
- MLIR compilation
- Deterministic execution model

#### Philosophy & Design
- Four primitives (sum, integrate, nonlinearity, events)
- Emergence from composition
- Type-safe cross-domain integration
- Transform-first thinking

#### Quality & Stability
- 580+ tests (all must pass)
- Production-ready core
- Comprehensive documentation
- Performance characteristics

---

## Rename Checklist

### Phase 1: Documentation Updates

#### Root Level Files
- [ ] README.md
  - [x] Add Philbrick sister project section
  - [ ] Update title from "Kairo" to "Morphogen"
  - [ ] Update installation instructions
  - [ ] Update examples
  - [ ] Update badges/links

- [ ] ARCHITECTURE.md
  - [ ] Update project name throughout
  - [ ] Add Philbrick references where relevant

- [ ] ROADMAP.md
  - [ ] Update project name
  - [ ] Add Philbrick integration milestones

- [ ] STATUS.md
  - [ ] Update project name
  - [ ] Update version to 0.11.0

- [ ] SPECIFICATION.md
  - [ ] Update language name references

- [ ] CHANGELOG.md
  - [ ] Add v0.11.0 entry documenting the rename

#### Documentation Directory
- [ ] docs/
  - [ ] Create docs/philbrick-bridge/ with integration docs
  - [ ] Update all markdown files referencing "Kairo"
  - [ ] Update use-case examples
  - [ ] Update tutorial content

- [ ] docs/adr/
  - [ ] Verify ADR-010 exists (Ecosystem Branding)
  - [ ] Verify ADR-011 exists (Project Renaming)
  - [ ] Create if missing
  - [ ] Update ADR README

### Phase 2: Code Changes

#### Python Package Structure
- [ ] Rename `morphogen/` directory → `morphogen/`
- [ ] Update `setup.py`
  - [ ] name="morphogen"
  - [ ] packages=find_packages(), ensuring "morphogen" package
- [ ] Update `pyproject.toml`
  - [ ] [tool.poetry] name = "morphogen"
  - [ ] Update description

#### Source Code
- [ ] Update all imports
  ```bash
  find . -name "*.py" -exec sed -i 's/from morphogen/from morphogen/g' {} +
  find . -name "*.py" -exec sed -i 's/import morphogen/import morphogen/g' {} +
  ```
- [ ] Update internal references
- [ ] Update docstrings
- [ ] Update comments

#### Tests
- [ ] Update test imports
- [ ] Update test assertions referencing package name
- [ ] Verify all 580+ tests pass

#### Examples
- [ ] examples/ directory
  - [ ] Update all example imports
  - [ ] Update example documentation
  - [ ] Verify examples run

### Phase 3: Build & Configuration

#### Build Files
- [ ] setup.py - name, packages
- [ ] pyproject.toml - project name, dependencies
- [ ] requirements.txt - if self-referencing
- [ ] .gitignore - update if any kairo-specific entries

#### CI/CD
- [ ] .github/workflows/ - update any hardcoded references
- [ ] GitHub Actions - update package names in test commands

### Phase 4: Repository & External

#### GitHub Repository
- [ ] Create new repository: `scottsen/morphogen`
- [ ] Migrate all code
- [ ] Update repository description
- [ ] Update topics/tags
- [ ] Set up README badges

#### Or: Rename Existing
- [ ] GitHub Settings → Repository Name → "morphogen"
- [ ] GitHub will auto-redirect kairo → morphogen

#### External References
- [ ] Update any external documentation
- [ ] Update any published papers/references
- [ ] Notify any users/contributors

---

## Migration Strategy

### For Users

**Breaking Change**: Version 0.11.0 will require import changes.

#### Migration Guide for Users
```python
# Before (v0.10.x and earlier)
from kairo.stdlib import audio, field
import kairo

# After (v0.11.0+)
from morphogen.stdlib import audio, field
import morphogen
```

#### Transition Support
1. **Deprecation Warning** (Optional v0.10.x patch):
   - Add warnings when `kairo` imports are used
   - Guide users to `morphogen` imports

2. **Compatibility Shim** (Optional, short-term):
   ```python
   # morphogen/__init__.py
   import sys
   sys.modules['kairo'] = sys.modules['morphogen']
   ```
   - Allows old `import kairo` to work temporarily
   - Print deprecation warning

3. **Clear Documentation**:
   - Migration guide
  - Changelog entry explaining rename
   - Updated README with new instructions

### Timeline

#### Preparation Phase (Current)
- [ ] This document created
- [x] Philbrick sister project established
- [x] README updated with Philbrick reference
- [ ] All documentation reviewed

#### Execution Phase (Week 1-2)
- [ ] Complete checklist above
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Migration guide written

#### Release Phase (Week 2)
- [ ] Tag v0.11.0
- [ ] GitHub rename or new repository
- [ ] Announce rename
- [ ] Update all external links

---

## Philbrick Integration Plan

### Documentation Linkage

#### In Morphogen Repository
- [x] README.md - Sister Project section
- [ ] docs/philbrick-bridge/ directory
  - [ ] 01-OVERVIEW.md - How the platforms connect
  - [ ] 02-SHARED-PRIMITIVES.md - The four operations
  - [ ] 03-WORKFLOWS.md - Design→Build→Validate
  - [ ] 04-COMPILATION.md - Morphogen→Philbrick firmware

#### In Philbrick Repository
- [x] README.md - Morphogen connection section
- [x] docs/vision/03-MORPHOGEN-BRIDGE.md - Complete architectural mirror

### Technical Integration

#### Phase 1: Shared Vocabulary (Months 1-3)
- [ ] Define unified descriptor protocol
- [ ] Map Morphogen operators → Philbrick primitives
- [ ] Create shared type system

#### Phase 2: Bidirectional Workflow (Months 3-6)
- [ ] Morphogen simulates Philbrick modules
- [ ] Philbrick validates Morphogen designs
- [ ] Test vector generation

#### Phase 3: Compilation Pipeline (Months 6-12)
- [ ] Morphogen → firmware compiler prototype
- [ ] MLIR lowering to Cortex-M
- [ ] Philbrick as Morphogen accelerator

---

## Risks & Mitigation

### Technical Risks

**Risk**: Import changes break existing user code
**Mitigation**:
- Clear migration guide
- Compatibility shim (short-term)
- Semver major bump signals breaking change

**Risk**: Tests fail after rename
**Mitigation**:
- Systematic sed replacement
- Run full test suite after each step
- Manual review of critical paths

**Risk**: Dependencies break
**Mitigation**:
- Update all setup files
- Test fresh install in clean environment
- Verify PyPI upload (if applicable)

### Community Risks

**Risk**: Users confused by rename
**Mitigation**:
- Announce well in advance
- Clear rationale in announcement
- Migration guide with examples

**Risk**: Loss of recognition/SEO
**Mitigation**:
- GitHub auto-redirects kairo → morphogen
- "Formerly Kairo" in initial descriptions
- Morphogen name has better SEO potential (unique term)

---

## Success Criteria

### Must-Have (v0.11.0 Release)
- ✅ All code imports updated
- ✅ All documentation updated
- ✅ All 580+ tests passing
- ✅ Clean GitHub repository
- ✅ Migration guide published

### Should-Have
- ✅ Philbrick documentation linkage complete
- ✅ Updated website/external links
- ✅ Announcement blog post

### Nice-to-Have
- ⏳ Compatibility shim for gradual migration
- ⏳ Video explaining the rename
- ⏳ Updated conference presentations

---

## Timeline

| Milestone | Target | Status |
|-----------|--------|--------|
| Preparation doc created | Nov 16 | ✅ Done |
| README updated | Nov 16 | ✅ Done |
| Philbrick repository created | Nov 16 | ✅ Done |
| Complete documentation updates | Week 1 | ⏳ Pending |
| Code rename executed | Week 2 | ⏳ Pending |
| Tests passing | Week 2 | ⏳ Pending |
| v0.11.0 tagged | Week 2 | ⏳ Pending |
| GitHub renamed | Week 2 | ⏳ Pending |
| Announcement published | Week 2 | ⏳ Pending |

---

## Related Documents

- **ADR-010**: Ecosystem Branding & Naming Strategy (to be verified/created)
- **ADR-011**: Project Renaming - Morphogen & Philbrick (to be verified/created)
- **Philbrick Repository**: https://github.com/scottsen/philbrick
- **wuluje-1116 Session**: Decision session for rename

---

## Notes

### From ADR-011 Decision (wuluje-1116)

**The Decision**:
- **Kairo → Morphogen** (digital platform) - Named after Turing's morphogenesis
- **Analog Platform → Philbrick** (hardware platform) - Named after George A. Philbrick
- **Modules → Philbricks** (composable function blocks)

**Why This Matters**:
- Creates unique market positioning: "emergence-focused continuous-time computation"
- Establishes historical legitimacy and educational value
- Supports Education & Academia GTM (primary market)
- Separates software and hardware into distinct projects with unified vision
- Creates "computational mythology" where every layer honors its inventor

---

## Next Actions

**Immediate**:
1. ✅ Create this document
2. ✅ Update README with Philbrick section
3. ⏳ Create docs/philbrick-bridge/ directory
4. ⏳ Verify/create ADR-010 and ADR-011

**Short-term**:
5. ⏳ Complete documentation updates
6. ⏳ Plan code rename execution
7. ⏳ Write migration guide

**Medium-term**:
8. ⏳ Execute rename
9. ⏳ Test thoroughly
10. ⏳ Release v0.11.0

---

**Status**: Preparation Phase Complete
**Next Milestone**: Documentation Updates
**Target Release**: v0.11.0 "The Morphogen Rename"

---

*"The universe computes in analog. We model it in Morphogen. We embody it in Philbrick. This is the full circle."*
