"""Embed: the last transform, one vector per chunk, cached by content.

Vectors live in the cache as JSON files keyed by the chunk text's hash,
so a chunk that changed is embedded again and a chunk that did not is
not; there is no eviction, a model change is handled manually with
``--refresh embed``.
"""

from __future__ import annotations

import json
import logging

from ragtools.llm import get_model

from rag_etl.cache import Cache
from rag_etl.core import Resource
from rag_etl.transformers.base import Transformer

logger = logging.getLogger(__name__)

# The vector of one chunk, cached by the chunk text's content hash
EMBEDDING_FILE = "embedding.json"

# Embedding calls are batched: one request per this many chunks
BATCH_SIZE = 32


def chunks_of(roots: list[Resource]) -> list[Resource]:
    """Every chunk in the trees: a node with text."""
    return [node for root in roots for node in root.walk() if node.text is not None]


class EmbedTransformer(Transformer):
    """Annotates every chunk with its vector, in place."""

    def __init__(self, cache: Cache):
        self.cache = cache

    def transform(self, roots: list[Resource]) -> list[Resource]:
        chunks = chunks_of(roots)
        missing: list[Resource] = []

        for chunk in chunks:
            cached = self.cache.read(chunk.content_hash, EMBEDDING_FILE)
            if cached is None:
                missing.append(chunk)
            else:
                chunk.embedding = json.loads(cached)

        logger.info(f"{len(chunks)} chunks, {len(missing)} to embed")

        for position in range(0, len(missing), BATCH_SIZE):
            batch = missing[position : position + BATCH_SIZE]
            vectors = get_model("embedding").embeddings([chunk.text or "" for chunk in batch])

            for chunk, vector in zip(batch, vectors):
                self.cache.write(chunk.content_hash, EMBEDDING_FILE, json.dumps(vector))
                chunk.embedding = vector

        return roots
