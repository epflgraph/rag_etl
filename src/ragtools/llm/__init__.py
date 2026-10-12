"""Model invocation layer: the unified RCP client and its defaults."""

from ragtools.llm.defaults import DEFAULT_LLM_PARAMS
from ragtools.llm.rcp import RCPClient

__all__ = ["DEFAULT_LLM_PARAMS", "RCPClient"]
