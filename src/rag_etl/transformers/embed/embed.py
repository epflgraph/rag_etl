"""Embed: one vector per chunk, handed to the loader, cached by content.

Vectors live in the cache as JSON files keyed by the chunk text's hash,
so a chunk that changed is embedded again and a chunk that did not is
not; there is no eviction, a model change is handled manually with
``--refresh embed``.
"""

from __future__ import annotations

import json
import logging

from collections.abc import Sequence

from ragtools.llm import RCPClient

from rag_etl.cache import Cache
from rag_etl.core import Resource

logger = logging.getLogger(__name__)

# The vector of one chunk, cached by the chunk text's content hash
EMBEDDING_FILE = "embedding.json"

# Embedding calls are batched: one request per this many chunks
BATCH_SIZE = 32


def chunks_of(roots: list[Resource]) -> list[Resource]:
    """Every chunk in the trees: a node with text."""
    return [node for root in roots for node in root.walk() if node.text is not None]


def embed(chunks: Sequence[Resource], llm: RCPClient, model: str, cache: Cache) -> list[tuple[Resource, list[float]]]:
    """Embed every chunk, one vector per chunk, cached by content.

    The pairs go to the loader as they are.
    """
    pairs: list[tuple[Resource, list[float]]] = []
    missing: list[Resource] = []

    for chunk in chunks:
        cached = cache.read(chunk.content_hash, EMBEDDING_FILE)
        if cached is not None:
            pairs.append((chunk, json.loads(cached)))
        else:
            missing.append(chunk)

    logger.info(f"{len(chunks)} chunks, {len(missing)} to embed")

    for position in range(0, len(missing), BATCH_SIZE):
        batch = missing[position : position + BATCH_SIZE]
        vectors = llm.embeddings([chunk.text or "" for chunk in batch], model=model)

        for chunk, vector in zip(batch, vectors):
            cache.write(chunk.content_hash, EMBEDDING_FILE, json.dumps(vector))
            pairs.append((chunk, vector))

    return pairs
