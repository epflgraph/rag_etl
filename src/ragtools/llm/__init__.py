"""Model invocation layer: the unified RCP client, its settled models and defaults."""

from ragtools.llm.defaults import DEFAULT_LLM_PARAMS, MODELS, get_model
from ragtools.llm.rcp import RCPClient

__all__ = ["DEFAULT_LLM_PARAMS", "MODELS", "RCPClient", "get_model"]
