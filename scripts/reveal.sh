#!/usr/bin/env bash
# Progressive Reveal CLI Wrapper for Morphogen Project
#
# This script provides easy access to the reveal tool for exploring
# Morphogen codebase files at different levels of detail.
#
# Usage:
#   ./scripts/reveal.sh 0 src/morphogen/domains/audio.py        # Metadata
#   ./scripts/reveal.sh 1 morphogen/domains/audio.py            # Structure
#   ./scripts/reveal.sh 2 docs/specifications/chemistry.md      # Preview
#   ./scripts/reveal.sh 3 SPECIFICATION.md                      # Full content

set -euo pipefail

LEVEL=${1:-1}
shift || true

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOCAL_REVEAL="$SCRIPT_DIR/reveal.py"

# Check if external reveal is installed, otherwise use local reveal.py
if command -v reveal &> /dev/null; then
    # Use external reveal tool
    exec reveal --level "$LEVEL" "$@"
elif [ -f "$LOCAL_REVEAL" ]; then
    # Use local reveal.py
    exec python3 "$LOCAL_REVEAL" --level "$LEVEL" "$@"
else
    echo "Error: No reveal tool found"
    echo ""
    echo "Options:"
    echo "  1. Install from gist:"
    echo "     pip install git+https://gist.github.com/scottsen/ee3fff354a79032f1c6d9d46991c8400"
    echo ""
    echo "  2. Use local version (should exist at $LOCAL_REVEAL)"
    exit 1
fi
