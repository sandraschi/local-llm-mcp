"""Syntax-check all gateway module files."""
import ast
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent / "src" / "llm_mcp" / "gateway"
files = list(root.rglob("*.py"))
errors = []
for f in files:
    try:
        ast.parse(f.read_bytes(), filename=str(f))
        print(f"  OK: {f.relative_to(root)}")
    except SyntaxError as e:
        errors.append((f, e))
        print(f"  FAIL: {f.relative_to(root)}: {e}")

if errors:
    print(f"\n{len(errors)} files with syntax errors")
    sys.exit(1)
else:
    print(f"\nAll {len(files)} files OK")
