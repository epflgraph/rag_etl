import importlib
import pkgutil

from rag_etl.projects.common.base import BaseProject
from rag_etl.projects.spec import ProjectSpec

from rag_etl.projects.cs470 import CS470Project
from rag_etl.projects.phys108 import PHYS108Project
from rag_etl.projects.micro315 import MICRO315Project
from rag_etl.projects.cs112g import CS112gProject
from rag_etl.projects.mgt645 import MGT645Project
from rag_etl.projects.math115a import MATH115aProject
from rag_etl.projects.math106e import MATH106eProject
from rag_etl.projects.me421 import ME421Project
from rag_etl.projects.me524 import ME524Project
from rag_etl.projects.math240 import MATH240Project
from rag_etl.projects.cs202 import CS202Project
from rag_etl.projects.com202 import COM202Project
from rag_etl.projects.math105a import MATH105aProject
from rag_etl.projects.math261 import MATH261Project
from rag_etl.projects.env342 import ENV342Project
from rag_etl.projects.phys106b import PHYS106bProject
from rag_etl.projects.bioeng310 import BIOENG310Project
from rag_etl.projects.me331 import ME331Project
from rag_etl.projects.mse236 import MSE236Project
from rag_etl.projects.cs233 import CS233Project
from rag_etl.projects.swissunidemo import SWISSUNIDEMOProject
from rag_etl.projects.fin405 import FIN405Project
from rag_etl.projects.math535 import MATH535Project
from rag_etl.projects.plasma import PLASMAProject
from rag_etl.projects.cmi import CMIProject
from rag_etl.projects.cmirestricted import CMIRESTRICTEDProject
from rag_etl.projects.unil134 import UNIL134Project
from rag_etl.projects.math310 import MATH310Project
from rag_etl.projects.cs119d import CS119dProject
from rag_etl.projects.cs550 import CS550Project
from rag_etl.projects.math111a import MATH111aProject
from rag_etl.projects.micro303 import MICRO303Project
from rag_etl.projects.micro452 import MICRO452Project
from rag_etl.projects.micro452_case_studies import MICRO452CaseStudiesProject
from rag_etl.projects.phys201a import PHYS201aProject
from rag_etl.projects.me326 import ME326Project
from rag_etl.projects.math101e import MATH101eProject
from rag_etl.projects.bio695 import BIO695Project
from rag_etl.projects.mse234 import MSE234Project
from rag_etl.projects.cs200 import CS200Project
from rag_etl.projects.ee310 import EE310Project
from rag_etl.projects.math111c import MATH111cProject

__all__ = [
    "BaseProject",
    "ProjectSpec",
    "CS470Project",
    "PHYS108Project",
    "MICRO315Project",
    "CS112gProject",
    "MGT645Project",
    "MATH115aProject",
    "MATH106eProject",
    "ME421Project",
    "ME524Project",
    "MATH240Project",
    "CS202Project",
    "COM202Project",
    "MATH105aProject",
    "MATH261Project",
    "ENV342Project",
    "PHYS106bProject",
    "BIOENG310Project",
    "ME331Project",
    "MSE236Project",
    "CS233Project",
    "SWISSUNIDEMOProject",
    "FIN405Project",
    "MATH535Project",
    "PLASMAProject",
    "CMIProject",
    "CMIRESTRICTEDProject",
    "UNIL134Project",
    "MATH310Project",
    "CS119dProject",
    "CS550Project",
    "MATH111aProject",
    "MICRO303Project",
    "MICRO452Project",
    "MICRO452CaseStudiesProject",
    "PHYS201aProject",
    "ME326Project",
    "MATH101eProject",
    "BIO695Project",
    "MSE234Project",
    "CS200Project",
    "EE310Project",
    "MATH111cProject",
]


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
