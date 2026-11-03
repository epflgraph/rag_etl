import importlib
import pkgutil

from rag_etl.projects.spec import ProjectSpec

__all__ = ["ProjectSpec", "from_code"]


def from_code(code: str) -> ProjectSpec:
    """Resolve a project code to its spec, failing loud when unknown.

    Specs live in the semester folders (e.g. ``2627_autumn``), one file per
    project: named after the project in lowercase, defining the spec under
    the project code in uppercase. A file that exists is a project.
    """

    for info in pkgutil.iter_modules(__path__):
        if not info.ispkg or not info.name.startswith("semester_"):
            continue

        try:
            module = importlib.import_module(f"{__name__}.{info.name}.{code.lower()}")
        except ModuleNotFoundError:
            continue

        spec = getattr(module, code.upper(), None)
        if isinstance(spec, ProjectSpec):
            return spec

    raise ValueError(f"Unknown project code: {code}")
