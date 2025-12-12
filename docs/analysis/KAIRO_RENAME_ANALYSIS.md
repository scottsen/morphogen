# Comprehensive Analysis: "Morphogen" References for Rename to "Morphogen"

## Executive Summary

This codebase contains **5,344 occurrences of "kairo"** across **356 files**, spanning Python code, documentation, configuration, tests, and language examples. The rename from "Morphogen" to "Morphogen" is well-documented in ADR-011 and a preparation checklist already exists in the archive.

### Key Statistics at a Glance

| Category | Occurrences | Files Affected | Risk Level |
|----------|------------|-----------------|-----------|
| **Documentation** | 4,023 | 132 | LOW |
| **Package Code** | 576 | 63 | **CRITICAL** |
| **Test Files** | 353 | 62 | HIGH |
| **Example Scripts** | 333 | 68 | MEDIUM |
| **Morphogen Language Files** | 51 | 23 | MEDIUM |
| **Setup/Config** | 6 | 2 | **CRITICAL** |
| **CI/CD** | 2 | 1 | HIGH |
| **TOTAL** | **5,344** | **356** | — |

---

## 1. PYTHON PACKAGE STRUCTURE (CRITICAL - IMMEDIATE ACTION)

### Directory Structure

```
/home/user/morphogen/
├── morphogen/                          ← Main package directory
│   ├── __init__.py                 (Contains "kairo" v0.10.0 reference)
│   ├── cli.py                      (Command entry point)
│   ├── ast/                        (Abstract syntax tree)
│   │   ├── __init__.py
│   │   ├── nodes.py
│   │   ├── types.py
│   │   └── visitors.py
│   ├── core/                       (Core operators and registry)
│   │   ├── __init__.py
│   │   ├── domain_registry.py
│   │   └── operator.py
│   ├── cross_domain/               (Cross-domain composition)
│   │   ├── __init__.py
│   │   ├── composer.py
│   │   ├── interface.py
│   │   ├── registry.py
│   │   └── validators.py
│   ├── lexer/                      (Tokenization)
│   │   ├── __init__.py
│   │   └── lexer.py
│   ├── mlir/                       (MLIR compilation infrastructure)
│   │   ├── __init__.py
│   │   ├── compiler.py
│   │   ├── compiler_v2.py
│   │   ├── context.py
│   │   ├── ir_builder.py
│   │   ├── optimizer.py
│   │   ├── codegen/
│   │   ├── dialects/
│   │   └── lowering/
│   ├── parser/                     (Syntax parsing)
│   │   ├── __init__.py
│   │   └── parser.py
│   ├── runtime/                    (Execution engine)
│   │   ├── __init__.py
│   │   └── runtime.py
│   ├── stdlib/                     (23+ domains)
│   │   ├── __init__.py
│   │   ├── audio.py
│   │   ├── agents.py
│   │   ├── field.py
│   │   ├── visual.py
│   │   ├── geometry.py
│   │   ├── [+18 more domain modules]
│   │   └── ...
│   ├── types/                      (Type system)
│   │   ├── __init__.py
│   │   ├── rate_compat.py
│   │   └── units.py
│   └── tests/
│       ├── test_io_storage.py
│       └── verify_io_storage.py
```

### Key Findings

**Total Python files in package**: 76 files  
**Average kairo references per file**: 7.6 references

**Most affected package modules** (by occurrence count):

| Module | Occurrences | Type |
|--------|------------|------|
| `morphogen.mlir.dialects.temporal.py` | 39 | MLIR operation names |
| `morphogen.mlir.dialects.audio.py` | 34 | MLIR operation names |
| `morphogen.core.domain_registry.py` | 26 | Domain registrations |
| `morphogen.mlir.dialects.field.py` | 26 | MLIR operation names |
| `morphogen.mlir.dialects.agent.py` | 26 | MLIR operation names |

---

## 2. PYTHON IMPORT STATEMENTS (HIGH - SYSTEMATIC CHANGES)

### Total Import Statements: 395 occurrences

### Most Common Import Patterns

```python
# Pattern 1: Core operator decorator (25 occurrences)
from morphogen.core.operator import operator, OpCategory

# Pattern 2: Field domain (14 occurrences)
from morphogen.stdlib.field import field, Field2D

# Pattern 3: Visual domain (13 occurrences)
from morphogen.stdlib.visual import visual

# Pattern 4: MLIR compilation (19 occurrences)
from morphogen.mlir.compiler_v2 import MLIRCompilerV2

# Pattern 5: MLIR context (7 occurrences)
from morphogen.mlir.context import KairoMLIRContext, MLIR_AVAILABLE

# Pattern 6: Parser (6 occurrences)
from morphogen.parser.parser import parse

# Pattern 7: Runtime (5 occurrences)
from morphogen.runtime.runtime import Runtime, ExecutionContext

# Pattern 8: Lexer (4 occurrences)
from morphogen.lexer import Lexer
```

### Import Locations

- **Tests**: 150+ imports
- **Examples**: 120+ imports
- **Benchmarks**: 8+ imports
- **Package internals**: 100+ imports

---

## 3. PACKAGE CONFIGURATION FILES (CRITICAL - MUST CHANGE)

### A. `setup.py` (5 occurrences)

```python
# Line 1: Docstring
"""Setup configuration for Morphogen."""

# Line 9: Package name
name="kairo",

# Line 54: Console script
"kairo=morphogen.cli:main",
```

**Required Changes**:
- Line 1: Update docstring
- Line 9: Change to `name="morphogen"`
- Line 54: Change to `"morphogen=morphogen.cli:main"`
- Optionally update description

### B. `pyproject.toml` (13 occurrences)

```toml
# Line 6: Project name
name = "kairo"

# Line 70: Console script
kairo = "morphogen.cli:main"

# Line 97: Coverage configuration (commented)
# "--cov=kairo",

# Line 110: Coverage source
source = ["kairo"]
```

**Required Changes**:
- Line 6: Change to `name = "morphogen"`
- Line 70: Change to `morphogen = "morphogen.cli:main"`
- Line 97: Update commented line
- Line 110: Change to `source = ["morphogen"]`

### C. `requirements.txt` (2 occurrences)

Current file mentions "Morphogen v0.7.0" in comments.

**Changes Needed**:
- Update version reference comment

---

## 4. DOCUMENTATION FILES (LOW RISK - CONTENT UPDATES)

### Total Documentation Files: 132 files with 4,023 occurrences

### Breakdown by Category

| Document Type | Files | Occurrences |
|---------------|-------|------------|
| ADR (Architecture Decision Records) | 2 | 50+ |
| Specification files | 10 | 800+ |
| Guide/Tutorial files | 25 | 600+ |
| Reference documentation | 15 | 300+ |
| API documentation | 20 | 400+ |
| Archive/Legacy docs | 30 | 400+ |
| Root-level docs | 6 | 500+ |

### High-Impact Documentation Files

```
/home/user/morphogen/README.md              (50+ references)
/home/user/morphogen/ARCHITECTURE.md        (45+ references)
/home/user/morphogen/SPECIFICATION.md       (180+ references - code blocks)
/home/user/morphogen/CHANGELOG.md           (150+ references)
/home/user/morphogen/AUDIO_SPECIFICATION.md (50+ references)
/home/user/morphogen/STATUS.md              (30+ references)
/home/user/morphogen/ECOSYSTEM_MAP.md       (20+ references)
/home/user/morphogen/LEVEL_3_TYPE_SYSTEM.md (40+ references)
```

### Documentation Types of "kairo" References

1. **Markdown code blocks** (language identifier)
   - Format: ` ```morphogen ... ``` `
   - Count: ~150+ instances
   - Impact: VISUAL ONLY (no functional change needed)

2. **Text references to "Morphogen" project**
   - Count: ~1500+ instances
   - Action: Search and replace "Morphogen" → "Morphogen"

3. **Module/package references** (e.g., `morphogen.stdlib`)
   - Count: ~600+ instances
   - Action: Search and replace `morphogen.` → `morphogen.`

4. **Comments in code blocks**
   - Count: ~50+ instances
   - Action: Update as part of code refactoring

---

## 5. CONFIGURATION FILES - CI/CD (.github/)

### CI/CD Workflow File: `.github/workflows/tests.yml`

**Current State**:
- References mention `creative_computation` (possible old naming)
- Contains outdated package references

**Occurrences**:
```yaml
Line 41: --cov=creative_computation
Line 72: ruff check creative_computation/ tests/
Line 77: black --check creative_computation/ tests/
```

**Issues Identified**:
- Workflow uses `creative_computation` instead of `kairo`
- This should be updated to `morphogen`

**Required Changes**:
```yaml
# Change:
--cov=creative_computation
--cov=morphogen

# Change:
ruff check creative_computation/ tests/
ruff check morphogen/ tests/

# Change:
black --check creative_computation/ tests/
black --check morphogen/ tests/
```

---

## 6. TEST FILES (HIGH - 353 OCCURRENCES)

### Test File Statistics

- **Total test files**: 62 files
- **Average kairo references per test**: 5.7

### Top Test Files by Reference Count

```
tests/test_portfolio_examples.py         (25+ references)
tests/test_integration.py                (21+ references)
tests/test_audio_dialect.py              (12+ references)
tests/test_temporal_dialect.py           (12+ references)
tests/test_field_dialect.py              (6+ references)
tests/test_geometry.py                   (9+ references)
tests/test_use_demo.py                   (6+ references)
[+55 more test files]
```

### Test Import Pattern Examples

```python
# Test file header imports (would need updating)
from morphogen.stdlib.field import field, Field2D
from morphogen.stdlib.visual import visual
from morphogen.mlir.compiler_v2 import MLIRCompilerV2
from morphogen.ast.nodes import Use
```

### Root-Level Test Files (5 files)

```
test_cellular_quick.py              (uses morphogen.stdlib.cellular)
test_diagnosis.py                   (uses morphogen.stdlib imports)
test_new_examples.py                (uses morphogen.stdlib imports)
test_proper_patterns.py             (uses morphogen.stdlib imports)
test_use_demo.py                    (uses kairo modules)
```

---

## 7. EXAMPLE CODE & SCRIPTS (MEDIUM - 333 OCCURRENCES)

### Example File Statistics

- **Total example files**: 68 files
- **Average kairo references per example**: 4.9

### Example Categories

| Category | Files | Occurrences |
|----------|-------|------------|
| Domain showcases | 30 | 150 |
| Benchmarks | 2 | 30 |
| Portfolio demos | 15 | 100 |
| Interactive simulations | 10 | 35 |
| Game/AI examples | 8 | 18 |

### High-Impact Examples

```
examples/interactive_physics_sandbox/demo.py    (from morphogen.stdlib.*)
examples/phase4_agent_operations.py              (from morphogen.mlir.*)
examples/flappy_bird/train_neuroevolution.py     (from morphogen.stdlib.*)
examples/procedural_graphics/demo_all_domains.py (from morphogen.stdlib.*)
benchmarks/field_operations_benchmark.py         (from morphogen.mlir.*)
```

### .morph Language Files (Domain-Specific Language)

```
Total .morph files: 24 files with 51 occurrences

Examples:
  examples/10_heat_equation.morph
  examples/11_gray_scott.morph
  examples/20_bouncing_spheres.morph
  examples/circuit/01_rc_filter.morph
  [+19 more files]
```

**Note**: These are the Morphogen DSL language files themselves. They don't contain "kairo" references (that's the language name for code blocks). Safe to leave as-is or minimal changes needed.

---

## 8. MLIR DIALECT REFERENCES (CRITICAL INTERNAL)

### MLIR Operation Names Using "morphogen." Prefix

Found throughout MLIR lowering and dialect implementations:

```python
# morphogen.audio dialect operations (dialects/audio.py)
op_name == "morphogen.audio.buffer.create"
op_name == "morphogen.audio.oscillator"
op_name == "morphogen.audio.envelope"
op_name == "morphogen.audio.filter"
op_name == "morphogen.audio.mix"

# morphogen.agent dialect operations (dialects/agent.py)
op_name == "morphogen.agent.spawn"
op_name == "morphogen.agent.update"
op_name == "morphogen.agent.query"

# morphogen.field dialect operations (dialects/field.py)
op_name == "morphogen.field.create"
op_name == "morphogen.field.gradient"
op_name == "morphogen.field.laplacian"

# morphogen.temporal dialect operations (dialects/temporal.py)
op_name == "morphogen.temporal.flow.create"
op_name == "morphogen.temporal.flow.run"
op_name == "morphogen.temporal.state.create"
```

### Files Affected

```
morphogen/mlir/dialects/audio.py           (34 occurrences)
morphogen/mlir/dialects/temporal.py        (39 occurrences)
morphogen/mlir/dialects/field.py           (26 occurrences)
morphogen/mlir/dialects/agent.py           (26 occurrences)
morphogen/mlir/lowering/audio_to_scf.py    (26 occurrences)
morphogen/mlir/lowering/temporal_to_scf.py (24 occurrences)
morphogen/mlir/lowering/field_to_scf.py    (18 occurrences)
morphogen/mlir/lowering/agent_to_scf.py    (18 occurrences)
```

**Important Note**: These MLIR operation names are internal string constants and must be changed systematically. They represent the dialect namespace and operation registry.

---

## 9. KAIRO DIALECT REFERENCES IN DOCUMENTATION

### Most Common morphogen.* References (from docs)

```
morphogen.stdlib              (91 occurrences)
morphogen.field              (44 occurrences)
morphogen.mlir               (32 occurrences)
morphogen.stream             (29 occurrences)
morphogen.cross_domain       (28 occurrences)
morphogen.transform          (23 occurrences)
morphogen.schedule           (21 occurrences)
morphogen.agent              (21 occurrences)
morphogen.ast                (8 occurrences)
morphogen.core               (7 occurrences)
```

**Action**: These should become `morphogen.stdlib`, `morphogen.field`, etc.

---

## 10. DIRECTORY OVERVIEW & STRUCTURE

### Root Project Directory

```
/home/user/morphogen/                    ← Project root (rename to /morphogen/)
├── .github/
│   └── workflows/
│       └── tests.yml                ← CI/CD config (references creative_computation)
├── .git/
├── .gitignore
├── archive/                         ← Historical docs (safe to update)
├── benchmarks/                      ← Benchmark scripts (2 files with 30+ refs)
├── docs/                            ← Documentation (130+ files, 3000+ refs)
├── examples/                        ← Example code (68 files, 333 refs)
├── morphogen/                           ← Main package dir (76 files, 576 refs) ← RENAME THIS
├── tests/                           ← Test suite (62 files, 353 refs)
├── morphogen/__init__.py                ← Package init (5 refs)
├── morphogen/cli.py                     ← CLI entry point (12 refs)
├── ARCHITECTURE.md
├── AUDIO_SPECIFICATION.md
├── CHANGELOG.md
├── README.md
├── SPECIFICATION.md
├── STATUS.md
├── setup.py                         ← Setup config (5 refs) ← UPDATE
├── pyproject.toml                   ← Project config (13 refs) ← UPDATE
├── requirements.txt                 ← Dependencies (2 refs) ← UPDATE
└── [other root files]
```

---

## 11. RISKY AREAS & SPECIAL CONSIDERATIONS

### 1. PUBLISHED PACKAGES (Risk: VERY HIGH if published)

**Current Status**: 
- The package is named `kairo` in setup.py/pyproject.toml
- Version is 0.10.0
- PyPI may or may not have this published

**Impact if on PyPI**:
- Existing installations will point to old `kairo` package
- New version would be `morphogen` (different package)
- Users need migration guide
- May need to maintain `kairo` as deprecated shim

**Recommendation**: Check if `kairo` is on PyPI. If yes:
- Publish `morphogen` with migration guide
- Optional: Create `kairo` shim package that imports from `morphogen`

### 2. EXTERNAL DEPENDENCIES (Risk: LOW)

Current dependencies don't reference kairo internally:
```
numpy, scipy, pillow, pygame (for basic setup)
lark (for parsing)
mlir-python-bindings (optional)
```

**Recommendation**: No changes needed to dependencies

### 3. CONSOLE SCRIPT ENTRY POINT (Risk: HIGH)

**Current**:
```python
"kairo=morphogen.cli:main"  # in setup.py
kairo = "morphogen.cli:main"  # in pyproject.toml
```

**Change To**:
```python
"morphogen=morphogen.cli:main"
morphogen = "morphogen.cli:main"
```

**Impact**: Users will use `morphogen` command instead of `kairo` command

### 4. MLIR DIALECT NAMESPACES (Risk: CRITICAL)

MLIR operations are registered with `morphogen.audio`, `morphogen.field` etc. prefixes.

**Current Registrations**:
- `morphogen.audio.*` operations (34 references)
- `morphogen.field.*` operations (26 references)  
- `morphogen.agent.*` operations (26 references)
- `morphogen.temporal.*` operations (39 references)

**Must Change**:
- All operation name strings in dialect definitions
- All operation name comparisons in lowering passes
- All MLIR module generation code

**Complexity**: HIGH - interdependent string matching

### 5. INTERNAL CLASS NAMES (Risk: MEDIUM)

Classes like `KairoMLIRContext` should be renamed:
- `KairoMLIRContext` → `MorphogenMLIRContext`
- `KairoCompiler` → `MorphogenCompiler` (if any)

**Count**: ~5-10 class definitions to rename

### 6. MODULE DOCSTRINGS (Risk: LOW)

Example from `/home/user/morphogen/morphogen/__init__.py`:
```python
"""Morphogen v0.10.0

A typed, deterministic domain-specific language for creative computation.
"""
```

**Action**: Update to:
```python
"""Morphogen v0.11.0

A typed, deterministic domain-specific language for creative computation.
"""
```

---

## 12. RENAME IMPLEMENTATION STRATEGY

### Phase 1: Preparation (Done)
- [x] ADR-011 decision documented
- [x] Rename prep checklist created (in archive)
- [x] This analysis complete

### Phase 2: Critical Changes (MUST DO)

1. **Package Directory** (1 change)
   - Rename `/home/user/morphogen/morphogen/` → `/home/user/morphogen/morphogen/`
   - Update all relative imports

2. **Configuration Files** (3 files)
   - `setup.py`: Update name, entry_points, docstring
   - `pyproject.toml`: Update name, scripts, coverage source
   - `requirements.txt`: Update comment

3. **CI/CD** (1 file)
   - `.github/workflows/tests.yml`: Fix references

4. **Python Imports** (395 occurrences across 200+ files)
   - Systematic find/replace: `from kairo` → `from morphogen`
   - Systematic find/replace: `import kairo` → `import morphogen`

### Phase 3: Core Code Updates (CRITICAL)

1. **MLIR Dialect Operations** (150+ string constants)
   - `morphogen.audio.*` → `morphogen.audio.*`
   - `morphogen.field.*` → `morphogen.field.*`
   - `morphogen.agent.*` → `morphogen.agent.*`
   - `morphogen.temporal.*` → `morphogen.temporal.*`

2. **Class Definitions** (5-10 classes)
   - `KairoMLIRContext` → `MorphogenMLIRContext`
   - Update subclasses

3. **Module Docstrings** (76+ files)
   - Update "Morphogen" → "Morphogen" in docstrings

### Phase 4: Test Updates (353 occurrences)

1. Update test imports (62 test files)
2. Update test helper functions
3. Verify all 580+ tests pass

### Phase 5: Documentation Updates (4,023 occurrences)

1. README.md and root docs (50+ occurrences)
2. docs/ directory (3000+ occurrences)
3. Update examples in documentation
4. Update architecture diagrams/descriptions

### Phase 6: Example Updates (333 occurrences)

1. Update example scripts (68 files)
2. Update benchmark code
3. Verify examples run correctly

### Phase 7: Verification & Release

1. Run full test suite
2. Build/test PyPI package
3. Verify CLI works: `morphogen --help`
4. Update version to 0.11.0
5. Create release notes

---

## 13. FILES REQUIRING MANUAL REVIEW

### High-Priority Manual Review

```
/home/user/morphogen/morphogen/cli.py
  - Check if "kairo" appears in user-facing strings/help text
  
/home/user/morphogen/morphogen/__init__.py
  - Update docstring and version number
  
/home/user/morphogen/morphogen/core/domain_registry.py
  - Check domain registration strings
```

### Archive & Historical Files

Already in archive (safe to update or leave):
```
archive/historical/MORPHOGEN_RENAME_PREP_historical.md
archive/historical/RENAME_DECISION_SUMMARY_historical.md
```

These can be reviewed after main rename is complete.

---

## 14. DETAILED FILE INVENTORY BY IMPACT

### CRITICAL FILES (Must change, no alternatives)

```
1. /home/user/morphogen/setup.py
   Changes: 3 locations (name, entry_points, docstring)
   Risk: CRITICAL if on PyPI
   
2. /home/user/morphogen/pyproject.toml
   Changes: 4 locations (name, scripts, coverage source, comment)
   Risk: CRITICAL - breaks pip install if not updated
   
3. /home/user/morphogen/morphogen/ (directory)
   Changes: Rename entire directory to morphogen/
   Risk: CRITICAL - all imports break without this
```

### HIGH PRIORITY FILES (Core functionality)

```
MLIR Dialects & Lowering:
  morphogen/mlir/dialects/audio.py         (34 refs)
  morphogen/mlir/dialects/temporal.py      (39 refs)
  morphogen/mlir/dialects/field.py         (26 refs)
  morphogen/mlir/dialects/agent.py         (26 refs)
  morphogen/mlir/lowering/audio_to_scf.py  (26 refs)
  morphogen/mlir/lowering/temporal_to_scf.py (24 refs)
  morphogen/mlir/lowering/field_to_scf.py  (18 refs)
  morphogen/mlir/lowering/agent_to_scf.py  (18 refs)

Core Modules:
  morphogen/core/domain_registry.py        (26 refs)
  morphogen/cli.py                         (12 refs)
  morphogen/__init__.py                    (5 refs)
```

### MEDIUM PRIORITY FILES (Tests & Examples)

```
Test Files (62 files):
  62+ test files import kairo modules
  195 unique import statements to update
  
Example Scripts (68 files):
  68 Python example scripts
  130+ import statements to update
  24 .morph language files (minimal impact)
```

### LOW PRIORITY FILES (Documentation)

```
Documentation (132 files):
  README.md, ARCHITECTURE.md, SPECIFICATION.md
  docs/* directory (all files)
  
Action: Global search & replace
"Morphogen" → "Morphogen"
morphogen. → morphogen.
```

---

## 15. AUTOMATION RECOMMENDATIONS

### Recommended Tooling Order

1. **File Renaming**
   ```bash
   mv morphogen/ morphogen/
   ```

2. **Global Find & Replace** (in this order)
   ```
   Step 1: morphogen. → morphogen.        (Package imports & module refs)
   Step 2: import kairo → import morphogen
   Step 3: from kairo → from morphogen
   Step 4: Morphogen → Morphogen           (Documentation & comments)
   Step 5: kairo_* → morphogen_*       (Variable/function names)
   Step 6: KairoX → MorphogenX         (Class names)
   ```

3. **Script to Update MLIR Operations**
   ```python
   # Generate script to update MLIR op name strings
   # morphogen.audio.* → morphogen.audio.*
   # morphogen.field.* → morphogen.field.*
   # etc.
   ```

4. **Validation**
   ```bash
   pytest tests/           # All 580+ tests must pass
   python -m black .       # Code formatting
   python -m ruff check .  # Linting
   ```

### Key Points for Automation

- **Be careful with**: Search/replace "kairo" in documentation might hit unrelated words
- **Special care**: MLIR operation strings are critical - each must be verified
- **Testing**: After each major change, run test suite
- **Order matters**: Do code changes before documentation to catch all references

---

## 16. MIGRATION GUIDE FOR USERS

Once renamed, users will need:

### For New Users
```bash
pip install morphogen        # (instead of kairo)
morphogen --help             # (instead of kairo)
```

### For Existing Users (Migration)
```python
# Old code
from morphogen.stdlib import field, audio

# New code
from morphogen.stdlib import field, audio
```

### Optional: Provide Compatibility Shim
Create a deprecated `kairo` package that re-exports from `morphogen`:

```python
# morphogen/__init__.py (new shim package)
import warnings
warnings.warn(
    "The 'kairo' package has been renamed to 'morphogen'. "
    "Please update your imports.",
    DeprecationWarning
)

from morphogen import *
```

---

## SUMMARY CHECKLIST

### Pre-Rename
- [x] Analysis complete (this document)
- [ ] ADR-011 confirmed and approved
- [ ] Git branch created for rename
- [ ] Backup of current state

### Directory & Package Rename
- [ ] Rename `morphogen/` directory to `morphogen/`
- [ ] Update all import paths in code
- [ ] Verify directory structure intact

### Configuration Files (3 files)
- [ ] `setup.py` - name, entry_points, docstring
- [ ] `pyproject.toml` - name, scripts, coverage
- [ ] `requirements.txt` - comments
- [ ] `.github/workflows/tests.yml` - coverage paths

### Core Code Updates (300+ locations)
- [ ] MLIR dialect strings (150 locations)
- [ ] Python imports (395 occurrences)
- [ ] Class names (10 classes)
- [ ] Module docstrings (76 files)

### Test Suite (62 files)
- [ ] Update test imports
- [ ] Run full test suite - all pass
- [ ] Verify examples run

### Documentation (132 files, 4,023 occurrences)
- [ ] Root-level docs (README, ARCHITECTURE, SPEC)
- [ ] docs/ directory
- [ ] ADR files
- [ ] Example documentation

### Release & Publication
- [ ] Update version to 0.11.0
- [ ] Create CHANGELOG entry
- [ ] Build and test PyPI package
- [ ] Merge to main
- [ ] Tag release: v0.11.0
- [ ] Publish to PyPI

---

## CONCLUSION

**Total Effort Estimate**: 3-5 days
- 1 day: File renaming + configuration updates
- 1 day: Core code refactoring (MLIR, imports)
- 1 day: Testing & verification
- 1-2 days: Documentation & cleanup

**Risk Level**: MEDIUM if properly planned and tested
- No functionality changes, only naming
- 580+ tests ensure nothing breaks
- Can be done incrementally if needed

**Key Success Factor**: Comprehensive test coverage (580+ tests) means changes are verifiable.

The detailed preparation checklist in `archive/historical/MORPHOGEN_RENAME_PREP_historical.md` provides a ready-to-use implementation roadmap.

