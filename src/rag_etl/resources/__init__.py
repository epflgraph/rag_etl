from rag_etl.resources.base_resource import BaseResource

from rag_etl.resources.moodle_resource import MoodleResource
from rag_etl.resources.mooc_resource import MOOCResource
from rag_etl.resources.ed_discussion_resource import EdDiscussionResource
from rag_etl.resources.local_resource import LocalResource

__all__ = [
    "BaseResource",
    "MoodleResource",
    "MOOCResource",
    "EdDiscussionResource",
    "LocalResource",
]
