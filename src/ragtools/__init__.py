"""ragtools: file preparation for RAG pipelines, usable standalone.

Archives, notebooks, PDFs, images and videos are turned into readable
text here. The library takes plain paths and hands files and text back;
it never imports the pipeline (`rag_etl`) — capabilities build their own
model clients from the config, so callers just call them.
"""

from ragtools.llm import DEFAULT_LLM_PARAMS, RCPClient, get_model

__all__ = ["DEFAULT_LLM_PARAMS", "RCPClient", "get_model"]
