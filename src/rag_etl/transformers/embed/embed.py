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

# Embedding calls are batched: at most BATCH_SIZE chunks per request, and at
# most REQUEST_CHAR_BUDGET chars of text, so one request cannot overflow the
# model's context (40,960 tokens; 96,000 chars leaves headroom even at a
# pessimistic 3 chars per token)
BATCH_SIZE = 32
REQUEST_CHAR_BUDGET = 96_000


def chunks_of(roots: list[Resource]) -> list[Resource]:
    """Every chunk in the trees: a node with text."""
    return [node for root in roots for node in root.walk() if node.text is not None]


class EmbedTransformer(Transformer):
    """Annotates every chunk with its vector, in place."""

    def __init__(self, cache: Cache):
        self.cache = cache

    def _embed_batch(self, batch: list[Resource]) -> None:
        """One request for the batch; each vector cached by its chunk's content hash."""
        vectors = get_model("embedding").embeddings([chunk.text or "" for chunk in batch])

        for chunk, vector in zip(batch, vectors):
            self.cache.write(chunk.content_hash, EMBEDDING_FILE, json.dumps(vector))
            chunk.embedding = vector

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

        # Pack requests greedily: a batch closes at the count cap or when the
        # next chunk would push it over the char budget
        batch: list[Resource] = []
        chars = 0
        for chunk in missing:
            length = len(chunk.text or "")
            if length > REQUEST_CHAR_BUDGET:
                raise ValueError(
                    f"{chunk.title}: {length:,} chars over the {REQUEST_CHAR_BUDGET:,}-char embedding budget; fix its cut"
                )
            if batch and (len(batch) == BATCH_SIZE or chars + length > REQUEST_CHAR_BUDGET):
                self._embed_batch(batch)
                batch, chars = [], 0
            batch.append(chunk)
            chars += length

        if batch:
            self._embed_batch(batch)

        return roots
