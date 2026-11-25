"""The layering contract: ragtools stands alone, rag_etl stands on it."""

import ast
from pathlib import Path

RAGTOOLS = Path(__file__).parent.parent / "src" / "ragtools"


def imports_of(path: Path) -> set[str]:
    """The absolute modules a file imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    return modules


def test_ragtools_imports_nothing_from_rag_etl():
    for path in RAGTOOLS.rglob("*.py"):
        imported = imports_of(path)
        offenders = {name for name in imported if name == "rag_etl" or name.startswith("rag_etl.")}
        assert not offenders, f"{path.relative_to(RAGTOOLS)} imports {offenders}"


def test_ragtools_never_imports_the_extractors():
    for path in RAGTOOLS.rglob("*.py"):
        imported = imports_of(path)
        offenders = {name for name in imported if "extractor" in name}
        assert not offenders, f"{path.relative_to(RAGTOOLS)} imports {offenders}"
