# Archived Documentation

This directory contains documentation that is no longer current but preserved for historical reference.

## Archive Organization

### phase-summaries/
Completion summaries and implementation plans from various development phases (v0.1-v0.7).

**Archived:** 2025-11-15

**Files:**
- `MLIR_PHASE5_COMPLETION.md` - Phase 5 completion summary
- `PHASE4_COMPLETION_SUMMARY.md` - Phase 4 completion summary
- `PHASE5_COMPLETION_SUMMARY.md` - Phase 5 completion summary
- `PHASE2_IMPLEMENTATION_PLAN.md` - Phase 2 implementation plan
- `PHASE2_COMPLETION_SUMMARY.md` - Phase 2 completion summary
- `PHASE3_COMPLETION_SUMMARY.md` - Phase 3 completion summary
- `MVP_COMPLETION_SUMMARY.md` - MVP completion summary
- `MLIR_PIPELINE_STATUS.md` - MLIR pipeline status snapshot
- `KAIRO_MLIR_PHASE3_PROMPT.md` - Phase 3 prompt document

**Why archived:** These phase completion summaries documented incremental development progress. The final status is now captured in current specs (SPEC-MLIR-DIALECTS.md, STATUS.md, CHANGELOG.md). Keeping them in the archive preserves the development history without cluttering current documentation.

### version-specific/
Documentation specific to older versions (v0.3.1, v0.7.0) that has been superseded by current specs.

**Archived:** 2025-11-15

**Files:**
- `KAIRO_v0.3.1_SUMMARY.md` - v0.3.1 summary
- `PARSER_v0.3.1_CHANGES.md` - Parser changes in v0.3.1
- `TESTING_IMPROVEMENTS_SUMMARY.md` - Testing improvements summary
- `TESTING_COMPLETE_SUMMARY.md` - Testing completion summary
- `TESTING_QUICKSTART.md` - Testing quickstart guide
- `v0.7.0_DESIGN.md` - v0.7.0 design document
- `RUNTIME_V0_3_1.md` - Runtime v0.3.1 documentation

**Why archived:** Version-specific documentation from older releases. Current version status is tracked in STATUS.md and CHANGELOG.md. Testing documentation is now in TESTING_STRATEGY.md and test suite comments.

### old-reviews/
Code reviews, validation reports, and status snapshots that are no longer actionable.

**Archived:** 2025-11-15

**Files:**
- `CODE_REVIEW_FINDINGS.md` - Code review findings
- `DOCS_VALIDATION_REPORT.md` - Documentation validation report
- `PORTFOLIO_EXAMPLES_STATUS.md` - Examples portfolio status
- `QUICK_ACTION_PLAN.md` - Quick action plan
- `ARCHITECTURE_ANALYSIS.md` - Architecture analysis (Nov 2025)

**Why archived:** These were point-in-time reviews and action items that have either been addressed or superseded by current development. Issues tracked in GitHub; current status in STATUS.md and PROJECT_REVIEW_AND_NEXT_STEPS.md.

## Current Documentation Structure

For current documentation, see:

- **Specifications:** `docs/SPEC-*.md` - Domain specifications, type system, scheduler, etc.
- **Architecture:** `ARCHITECTURE.md`, `docs/architecture.md`, `docs/DOMAIN_ARCHITECTURE.md`
- **Guides:** `docs/GUIDES/` - Implementation guides
- **Examples:** `docs/EXAMPLES/` - Example use cases
- **ADRs:** `docs/ADR/` - Architectural decision records
- **Status:** `STATUS.md`, `CHANGELOG.md`, `PROJECT_REVIEW_AND_NEXT_STEPS.md`

## Retention Policy

Archived documents are kept indefinitely for historical reference. They may be removed if:
1. The information is fully integrated into current docs
2. The content is no longer relevant to understanding Morphogen's evolution
3. Disk space becomes a concern (unlikely)

---

**Last Updated:** 2025-11-15
