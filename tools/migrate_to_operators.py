#!/usr/bin/env python3
"""
Migrate legacy domain files to use @operator decorators.

This script:
1. Parses Python source files
2. Finds function definitions
3. Adds @operator decorators
4. Preserves existing code structure

Usage:
    python tools/migrate_to_operators.py morphogen/stdlib/molecular.py
    python tools/migrate_to_operators.py morphogen/stdlib/thermal_ode.py --dry-run
"""

import argparse
import ast
import re
from pathlib import Path
from typing import List, Tuple


class OperatorMigrator:
    """Add @operator decorators to domain functions."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.modified = []

    def migrate_file(self, filepath: Path) -> Tuple[bool, int]:
        """
        Migrate a single file.

        Returns:
            (success, num_functions_decorated)
        """
        print(f"\n{'[DRY RUN] ' if self.dry_run else ''}Processing: {filepath}")

        if not filepath.exists():
            print(f"  ERROR: File not found")
            return False, 0

        # Read source
        source = filepath.read_text()

        # Parse to find functions
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            print(f"  ERROR: Failed to parse: {e}")
            return False, 0

        # Find functions that need decorators
        functions_to_decorate = self._find_functions(tree, source)

        if not functions_to_decorate:
            print(f"  No functions to decorate")
            return True, 0

        print(f"  Found {len(functions_to_decorate)} functions to decorate")

        # Add decorators
        modified_source = self._add_decorators(source, functions_to_decorate)

        # Add import if needed
        if '@operator' in modified_source and 'from morphogen.core.operator import operator' not in modified_source:
            modified_source = self._add_import(modified_source)

        # Write back (unless dry run)
        if not self.dry_run:
            # Backup original
            backup = filepath.with_suffix('.py.backup')
            filepath.rename(backup)

            # Write modified
            filepath.write_text(modified_source)
            print(f"  ✅ Modified (backup: {backup.name})")
        else:
            print(f"  Would modify {len(functions_to_decorate)} functions")

        return True, len(functions_to_decorate)

    def _find_functions(self, tree: ast.AST, source: str) -> List[Tuple[str, int]]:
        """
        Find function definitions that should be decorated.

        Returns:
            List of (function_name, line_number) tuples
        """
        functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip private functions
                if node.name.startswith('_'):
                    continue

                # Skip methods inside classes (we only decorate module-level functions)
                # Check if parent is a ClassDef
                is_method = False
                for parent_node in ast.walk(tree):
                    if isinstance(parent_node, ast.ClassDef):
                        for class_node in ast.walk(parent_node):
                            if class_node == node:
                                is_method = True
                                break

                if is_method:
                    continue

                # Skip if already decorated with @operator
                has_operator = any(
                    isinstance(d, ast.Name) and d.id == 'operator' or
                    isinstance(d, ast.Attribute) and d.attr == 'operator'
                    for d in node.decorator_list
                )

                if has_operator:
                    continue

                # This function needs decoration
                functions.append((node.name, node.lineno))

        return sorted(functions, key=lambda x: x[1])

    def _add_decorators(self, source: str, functions: List[Tuple[str, int]]) -> str:
        """Add @operator decorators to functions."""
        lines = source.split('\n')

        # Process in reverse order to maintain line numbers
        for func_name, lineno in reversed(functions):
            # Find the actual def line (lineno is 1-indexed)
            idx = lineno - 1

            # Find indentation
            def_line = lines[idx]
            indent = len(def_line) - len(def_line.lstrip())
            indent_str = ' ' * indent

            # Insert decorator before function
            decorator = f'{indent_str}@operator("{func_name}")'
            lines.insert(idx, decorator)

        return '\n'.join(lines)

    def _add_import(self, source: str) -> str:
        """Add operator import at appropriate location."""
        lines = source.split('\n')

        # Find where to insert (after docstring, before first code)
        insert_idx = 0
        in_docstring = False

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Handle module docstring
            if i == 0 and (stripped.startswith('"""') or stripped.startswith("'''")):
                in_docstring = True
                quote = '"""' if stripped.startswith('"""') else "'''"
                if stripped.count(quote) >= 2:
                    in_docstring = False
                continue

            if in_docstring:
                if '"""' in line or "'''" in line:
                    in_docstring = False
                continue

            # Skip existing imports
            if stripped.startswith('import ') or stripped.startswith('from '):
                insert_idx = i + 1
                continue

            # Found first non-import, non-docstring line
            if stripped and not stripped.startswith('#'):
                break

        # Insert import
        import_line = 'from morphogen.core.operator import operator'

        # Add blank line if needed
        if insert_idx > 0 and lines[insert_idx - 1].strip():
            lines.insert(insert_idx, '')
            insert_idx += 1

        lines.insert(insert_idx, import_line)

        return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Migrate domain files to use @operator decorators')
    parser.add_argument('files', nargs='+', help='Files to migrate')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done without modifying files')
    args = parser.parse_args()

    migrator = OperatorMigrator(dry_run=args.dry_run)

    total_files = 0
    total_functions = 0

    for filepath in args.files:
        path = Path(filepath)
        success, num_funcs = migrator.migrate_file(path)
        if success:
            total_files += 1
            total_functions += num_funcs

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Summary:")
    print(f"  Files processed: {total_files}")
    print(f"  Functions decorated: {total_functions}")

    if args.dry_run:
        print(f"\nRun without --dry-run to apply changes")
    else:
        print(f"\n✅ Migration complete!")
        print(f"   Original files backed up with .py.backup extension")


if __name__ == '__main__':
    main()
