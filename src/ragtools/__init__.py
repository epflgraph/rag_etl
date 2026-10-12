"""ragtools: pure file preparation for RAG pipelines.

Archives, notebooks, PDFs, images and videos are turned into readable
text here. The library takes plain paths and hands files and text back;
it never imports the pipeline (`rag_etl`) — model clients are constructed
by the caller and handed over.
"""

from ragtools.llm import DEFAULT_LLM_PARAMS, RCPClient

__all__ = ["DEFAULT_LLM_PARAMS", "RCPClient"]
