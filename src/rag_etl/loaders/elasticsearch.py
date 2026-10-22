"""The Elasticsearch loader: publishes one run into a fresh index.

The index is named after the project and the date, so a day's first run
is clean blue/green: recreate, bulk-load, refresh, then one atomic alias
update points consumers at it. The alias consumers query is named after
the project only, and a failed run never touches it.
"""

from __future__ import annotations

import json
import logging

from datetime import datetime
from pathlib import Path
from time import time_ns
from typing import Any

from elasticsearch import Elasticsearch, NotFoundError
from elasticsearch.helpers import bulk

from rag_etl.config import config_value
from rag_etl.core import Resource

logger = logging.getLogger(__name__)


def alias_name(project: str) -> str:
    """The stable name consumers query: rag_{project}."""
    return f"rag_{project}"


def index_name(project: str, timestamp: datetime) -> str:
    """The fresh index one run publishes into: rag_{project}_{YYYY_MM_DD}."""
    return f"{alias_name(project)}_{timestamp:%Y_%m_%d}"


def index_settings(language: str | None) -> dict[str, Any]:
    """The create-index body, with the content analyzer set for the language."""
    settings: dict[str, Any] = json.loads(
        (Path(__file__).parent / "index_settings.json").read_text(encoding="utf-8")
    )
    settings["mappings"]["properties"]["content"]["analyzer"] = f"base_{language or 'en'}"
    return settings


def _client() -> Elasticsearch:
    """The shared instance, configured from the environment."""
    return Elasticsearch(
        hosts=[f"https://{config_value('ES_HOST')}:{config_value('ES_PORT')}"],
        basic_auth=(config_value("ES_USERNAME"), config_value("ES_PASSWORD")),
        ca_certs=config_value("ES_CAFILE"),
        request_timeout=3600,
    )


class ElasticsearchLoader:
    """Publishes one run's chunks into a fresh index, then shifts the alias."""

    def __init__(self, project: str, timestamp: datetime, language: str | None, client: Elasticsearch | None = None):
        self.alias = alias_name(project)
        self.index = index_name(project, timestamp)
        self.language = language
        self.client = client if client is not None else _client()

    def load(self, roots: list[Resource]) -> None:
        """Recreate the index, bulk-load the chunks, shift the alias."""
        self._recreate_index()
        bulk(self.client, self._actions(roots))
        self.client.indices.refresh(index=self.index)
        self._shift_alias()

    def _recreate_index(self) -> None:
        if self.client.indices.exists(index=self.index):
            self.client.indices.delete(index=self.index)
        body = index_settings(self.language)
        self.client.indices.create(index=self.index, settings=body["settings"], mappings=body["mappings"])

    def _actions(self, roots: list[Resource]) -> list[dict[str, Any]]:
        """One index action per chunk, with a strictly increasing per-document timestamp."""
        epoch = 0
        actions: list[dict[str, Any]] = []

        for root in roots:
            for node in root.walk():
                if node.text is None:
                    continue
                epoch = max(time_ns() // 1_000_000, epoch + 1)
                actions.append(
                    {
                        "_op_type": "index",
                        "_index": self.index,
                        "_id": f"{epoch}-{node.content_hash}",
                        "_source": self._source(node),
                    }
                )

        return actions

    def _source(self, node: Resource) -> dict[str, Any]:
        """One document: content, vector, breadcrumb lineage, deep link."""
        embedding = node.embedding
        if embedding is None:
            raise ValueError(f"chunk {node.title!r} is not embedded")

        url = next((ancestor.url for ancestor in reversed(node.lineage()) if ancestor.url is not None), "<redacted>")
        document: dict[str, Any] = {
            "content": node.text,
            "embedding": embedding,
            "lineage": [ancestor.title or ancestor.url or "unknown" for ancestor in node.lineage()],
            "url": url,
            "type": node.type,
            "subtype": node.subtype,
            "number": node.number,
            "subnumber": node.subnumber,
            "week": node.week,
            "is_solution": node.is_solution,
            "from": node.from_,
            "until": node.until,
        }
        return {key: value for key, value in document.items() if value is not None}

    def _shift_alias(self) -> None:
        """Point the alias at the fresh index in one atomic update."""
        actions: list[dict[str, Any]] = []
        try:
            current = self.client.indices.get_alias(name=self.alias)
            actions.extend({"remove": {"index": name, "alias": self.alias}} for name in current)
        except NotFoundError:
            pass

        actions.append({"add": {"index": self.index, "alias": self.alias}})
        self.client.indices.update_aliases(actions=actions)
