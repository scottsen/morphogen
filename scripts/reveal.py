#!/usr/bin/env python3
"""
Reveal - Progressive File Explorer for Morphogen Project

A simple tool to explore files at different levels of detail:
- Level 0: Metadata (size, lines, type)
- Level 1: Structure (headings, functions, classes)
- Level 2: Preview (sample content)
- Level 3: Full content

Usage:
    python scripts/reveal.py --level 0 README.md
    python scripts/reveal.py --level 1 src/morphogen/domains/audio.py
    python scripts/reveal.py --level 2 docs/specifications/chemistry.md
"""

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path


def get_file_metadata(filepath):
    """Level 0: Basic metadata"""
    path = Path(filepath)
    if not path.exists():
        return {"error": f"File not found: {filepath}"}

    stat = path.stat()
    size = stat.st_size

    # Count lines
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = sum(1 for _ in f)
    except:
        lines = 0

    # Hash
    try:
        with open(filepath, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:8]
    except:
        file_hash = "unknown"

    return {
        "path": str(path),
        "size": size,
        "lines": lines,
        "hash": file_hash,
        "type": path.suffix or "no extension"
    }


def get_python_structure(filepath):
    """Extract Python file structure"""
    structure = {"imports": [], "classes": [], "functions": []}

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.rstrip()

                # Imports
                if line.startswith('import ') or line.startswith('from '):
                    structure["imports"].append(f"{i:4d}: {line}")

                # Classes
                if line.startswith('class '):
                    match = re.match(r'class\s+(\w+)', line)
                    if match:
                        structure["classes"].append(f"{i:4d}: {match.group(1)}")

                # Functions
                if line.startswith('def '):
                    match = re.match(r'def\s+(\w+)\s*\(([^)]*)\)', line)
                    if match:
                        name = match.group(1)
                        args = match.group(2)
                        structure["functions"].append(f"{i:4d}: {name}({args})")
    except Exception as e:
        structure["error"] = str(e)

    return structure


def get_markdown_structure(filepath):
    """Extract Markdown file structure"""
    structure = {"headings": [], "code_blocks": 0, "lists": 0}
    in_code_block = False

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.rstrip()

                # Code blocks
                if line.startswith('```'):
                    in_code_block = not in_code_block
                    if in_code_block:
                        structure["code_blocks"] += 1
                    continue

                if in_code_block:
                    continue

                # Headings
                if line.startswith('#'):
                    level = len(line) - len(line.lstrip('#'))
                    title = line.lstrip('#').strip()
                    indent = '  ' * (level - 1)
                    structure["headings"].append(f"{i:4d}: {indent}{title}")

                # Lists
                if re.match(r'^\s*[-*+]\s', line) or re.match(r'^\s*\d+\.\s', line):
                    structure["lists"] += 1
    except Exception as e:
        structure["error"] = str(e)

    return structure


def get_file_preview(filepath, num_lines=20):
    """Level 2: Preview with sample lines"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        total = len(lines)
        if total <= num_lines * 2:
            # Show all if file is small
            return {
                "total_lines": total,
                "preview": ''.join(f"{i+1:4d}: {line.rstrip()}\n" for i, line in enumerate(lines))
            }
        else:
            # Show beginning, middle, and end
            head = lines[:num_lines//2]
            middle_idx = total // 2
            middle = lines[middle_idx:middle_idx + num_lines//4]
            tail = lines[-num_lines//4:]

            preview = ""
            preview += ''.join(f"{i+1:4d}: {line.rstrip()}\n" for i, line in enumerate(head))
            preview += f"\n... ({total - num_lines} more lines) ...\n\n"
            preview += ''.join(f"{middle_idx+i+1:4d}: {line.rstrip()}\n" for i, line in enumerate(middle))
            preview += f"\n... ...\n\n"
            preview += ''.join(f"{total-len(tail)+i+1:4d}: {line.rstrip()}\n" for i, line in enumerate(tail))

            return {
                "total_lines": total,
                "preview": preview
            }
    except Exception as e:
        return {"error": str(e)}


def get_full_content(filepath):
    """Level 3: Full file content with line numbers"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        content = ''.join(f"{i+1:4d}: {line.rstrip()}\n" for i, line in enumerate(lines))
        return {
            "total_lines": len(lines),
            "content": content
        }
    except Exception as e:
        return {"error": str(e)}


def format_output(filepath, level):
    """Format and display output based on level"""
    print(f"\n{'='*80}")
    print(f"File: {filepath}")
    print(f"Level: {level}")
    print(f"{'='*80}\n")

    if level == 0:
        # Metadata
        meta = get_file_metadata(filepath)
        if "error" in meta:
            print(f"ERROR: {meta['error']}")
            return

        print(f"Path:  {meta['path']}")
        print(f"Size:  {meta['size']:,} bytes")
        print(f"Lines: {meta['lines']:,}")
        print(f"Type:  {meta['type']}")
        print(f"Hash:  {meta['hash']}")

    elif level == 1:
        # Structure
        meta = get_file_metadata(filepath)
        if "error" in meta:
            print(f"ERROR: {meta['error']}")
            return

        print(f"Lines: {meta['lines']:,} | Size: {meta['size']:,} bytes\n")

        ext = Path(filepath).suffix.lower()

        if ext == '.py':
            struct = get_python_structure(filepath)

            if struct["imports"]:
                print(f"IMPORTS ({len(struct['imports'])}):")
                for imp in struct["imports"][:10]:  # Limit to first 10
                    print(f"  {imp}")
                if len(struct["imports"]) > 10:
                    print(f"  ... and {len(struct['imports']) - 10} more")
                print()

            if struct["classes"]:
                print(f"CLASSES ({len(struct['classes'])}):")
                for cls in struct["classes"]:
                    print(f"  {cls}")
                print()

            if struct["functions"]:
                print(f"FUNCTIONS ({len(struct['functions'])}):")
                for func in struct["functions"][:20]:  # Limit to first 20
                    print(f"  {func}")
                if len(struct["functions"]) > 20:
                    print(f"  ... and {len(struct['functions']) - 20} more")

        elif ext == '.md':
            struct = get_markdown_structure(filepath)

            if struct["headings"]:
                print(f"STRUCTURE ({len(struct['headings'])} headings):")
                for heading in struct["headings"]:
                    print(f"  {heading}")
                print()

            print(f"Code blocks: {struct['code_blocks']}")
            print(f"List items:  ~{struct['lists']}")

        else:
            print(f"[Level 1 structure not available for {ext} files]")
            print("Use --level 2 for preview or --level 3 for full content")

    elif level == 2:
        # Preview
        preview = get_file_preview(filepath, num_lines=40)
        if "error" in preview:
            print(f"ERROR: {preview['error']}")
            return

        print(f"Total lines: {preview['total_lines']:,}\n")
        print(preview['preview'])

    elif level == 3:
        # Full content
        full = get_full_content(filepath)
        if "error" in full:
            print(f"ERROR: {full['error']}")
            return

        print(f"Total lines: {full['total_lines']:,}\n")
        print(full['content'])

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Progressive file explorer for Morphogen project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/reveal.py --level 0 README.md
  python scripts/reveal.py --level 1 morphogen/domains/audio.py
  python scripts/reveal.py --level 2 docs/specifications/chemistry.md
  python scripts/reveal.py --level 3 SPECIFICATION.md
        """
    )

    parser.add_argument('file', help='File to reveal')
    parser.add_argument('--level', '-l', type=int, default=1, choices=[0, 1, 2, 3],
                        help='Detail level: 0=metadata, 1=structure, 2=preview, 3=full')

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"ERROR: File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    format_output(args.file, args.level)


if __name__ == '__main__':
    main()
