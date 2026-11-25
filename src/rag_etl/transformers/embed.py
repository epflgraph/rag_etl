"""Embed: the last transform, one vector per chunk, cached by content.

Vectors live in the cache as JSON files keyed by the chunk text's hash,
so a chunk that changed is embedded again and a chunk that did not is
not; there is no eviction. A model change bumps the cache version.
"""

import json
import logging

from dataclasses import dataclass

from ragtools.llm import get_model

from rag_etl.core import Resource, iter_chunks
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

# A chunk must fit in one request on its own: the cut drops anything bigger,
# so this is the one size limit every chunk answers to
MAX_CHUNK_CHARS = REQUEST_CHAR_BUDGET


@dataclass(frozen=True)
class EmbedTransformer(Transformer):
    """Annotates every chunk with its vector, in place."""

    name = "embed"
    # Vectors of two models are not comparable, and a refresh that stops
    # halfway would leave both in one folder, so a new model bumps this
    cache_version = "embedding-1"

    def _embed_batch(self, batch: list[Resource]) -> None:
        """One request for the batch; each vector cached by its chunk's content hash."""
        vectors = get_model("embedding").embeddings([chunk.text or "" for chunk in batch])

        cache = self.cache
        for chunk, vector in zip(batch, vectors):
            cache.write(chunk.content_hash, EMBEDDING_FILE, json.dumps(vector))
            chunk.embedding = vector

    def transform(self, roots: list[Resource]) -> list[Resource]:
        chunks = list(iter_chunks(roots))
        missing: list[Resource] = []

        cache = self.cache
        for chunk in chunks:
            cached = cache.read(chunk.content_hash, EMBEDDING_FILE)
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
            # The cut drops oversized chunks, so this only fires on a cut bug
            if length > MAX_CHUNK_CHARS:
                raise ValueError(
                    f"{chunk.title}: {length:,} chars over the {MAX_CHUNK_CHARS:,}-char chunk limit; fix its cut"
                )
            if batch and (len(batch) == BATCH_SIZE or chars + length > REQUEST_CHAR_BUDGET):
                self._embed_batch(batch)
                batch, chars = [], 0
            batch.append(chunk)
            chars += length

        if batch:
            self._embed_batch(batch)

        return roots
