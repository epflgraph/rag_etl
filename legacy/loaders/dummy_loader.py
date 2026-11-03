from __future__ import annotations

from typing import Sequence

import logging

from rag_etl.loaders import BaseLoader
from rag_etl.resources import BaseResource


class DummyLoader(BaseLoader):
    """
    A no-op loader that only logs resources instead of persisting them.

    Useful for testing pipelines or verifying extractor/transformer output
    without touching a database or external system.
    """

    def load(self, resources: Sequence[BaseResource]) -> None:
        """Log the resources instead of persisting them."""
        logging.debug(f"DummyLoader received {len(resources)} resources")

        for resource in resources:
            logging.debug(resource)

        logging.debug("DummyLoader finished (no persistence performed).")
