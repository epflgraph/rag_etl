"""Model invocation layer: the unified RCP client and its defaults."""

from rag_etl.llm.defaults import DEFAULT_LLM_PARAMS
from rag_etl.llm.rcp import RCPClient

__all__ = ["DEFAULT_LLM_PARAMS", "RCPClient"]
