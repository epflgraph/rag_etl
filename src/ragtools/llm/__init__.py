"""Model invocation layer: the unified RCP client, its settled models and defaults."""

from ragtools.llm.defaults import MODELS, get_model
from ragtools.llm.rcp import AnswerTooLong, RCPClient

__all__ = ["AnswerTooLong", "MODELS", "RCPClient", "get_model"]
