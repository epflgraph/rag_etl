"""Model invocation layer: the unified RCP client and the catalog of its settled models."""

from ragtools.llm.catalog import MODELS, get_model
from ragtools.llm.rcp import AnswerTooLong, RCPClient

__all__ = ["AnswerTooLong", "MODELS", "RCPClient", "get_model"]
