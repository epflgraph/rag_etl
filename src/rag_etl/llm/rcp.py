"""The unified RCP client: one entry point for every model invocation.

Vision, text, structured output and embeddings all go through here; the
model travels with each call, handed over by the project spec, and the
parameters are the recommended ones for that model unless the caller
overrides them. Failures are retried a few times and then raised: either
the run completes or it stops loud.
"""

from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from langfuse import get_client
from openai import OpenAI
from pydantic import BaseModel

import rag_etl.utils.mime_types as mt

from rag_etl.config import CONFIG
from rag_etl.llm.defaults import DEFAULT_LLM_PARAMS

ModelT = TypeVar("ModelT", bound=BaseModel)

# Closers vLLM may still leave inline in the content of thinking models
THINKING_CLOSER = "</think>"


@dataclass(frozen=True)
class RCPClient:
    """Client for the EPFL RCP inference platform (OpenAI-compatible).

    A failed request is retried a few times by the underlying client and
    then raised: a page that cannot be read stops the run rather than
    leaving a silent gap in the index.
    """

    base_url: str | None = None
    api_key: str | None = None
    max_retries: int = 3
    timeout: int = 600

    def _client(self) -> OpenAI:
        return OpenAI(
            base_url=self.base_url or CONFIG["RCP_BASE_URL"],
            api_key=self.api_key or CONFIG["RCP_API_KEY"],
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def _send(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        response_format: type[BaseModel] | None = None,
        name: str,
        enable_thinking: bool = True,
        params: dict[str, Any] | None = None,
    ) -> str:
        mode = "thinking" if enable_thinking else "instruct"
        try:
            call_params = dict(DEFAULT_LLM_PARAMS[(model, mode)])
        except KeyError as e:
            raise ValueError(
                f"No recommended parameters for model {model} in mode {mode}: add them to rag_etl/llm/defaults.py"
            ) from e
        if params:
            call_params.update(params)

        if response_format:
            response_format_schema = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema": response_format.model_json_schema(),
                    "strict": True,
                },
            }
        else:
            response_format_schema = None

        client = self._client()
        with get_client().start_as_current_observation(as_type="generation", name=name, model=model, input=messages) as generation:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format=response_format_schema,
                    **call_params,
                )
            except Exception as e:
                raise RuntimeError(f"RCP request '{name}' with model {model} failed: {e}") from e

            message = response.choices[0].message
            content = message.content.strip()

            # Capture reasoning / thinking tokens if present (modern vLLM uses `reasoning`, current RCP uses `reasoning_content`)
            reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
            update_kwargs: dict = {"output": content}
            if reasoning:
                update_kwargs["metadata"] = {"reasoning": reasoning}
            if response.usage:
                update_kwargs["usage_details"] = {
                    "input": response.usage.prompt_tokens,
                    "output": response.usage.completion_tokens,
                }
            generation.update(**update_kwargs)

        # Some models still leave the thinking trace inline in the content
        if THINKING_CLOSER in content:
            content = content.split(THINKING_CLOSER)[-1].strip()

        return content

    def text(
        self,
        prompt: str,
        *,
        model: str,
        name: str = "llm-request",
        enable_thinking: bool = True,
        params: dict[str, Any] | None = None,
    ) -> str:
        """Send a plain text prompt and return the model's answer."""
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        return self._send(model, messages, name=name, enable_thinking=enable_thinking, params=params)

    def vision(
        self,
        prompt: str,
        images: Sequence[str | Path],
        *,
        model: str,
        name: str = "vision-request",
        enable_thinking: bool = True,
        params: dict[str, Any] | None = None,
    ) -> str:
        """Send a prompt with one or more images (paths) and return the model's answer."""
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
            image = Path(image)
            mime_type = mt.guess_mime_type(str(image))
            if mime_type is None:
                raise ValueError(f"Could not determine MIME type for {image}")
            b64 = base64.b64encode(image.read_bytes()).decode("utf-8")
            content.append({"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}})

        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        return self._send(model, messages, name=name, enable_thinking=enable_thinking, params=params)

    def structured(
        self,
        prompt: str,
        schema: type[ModelT],
        *,
        model: str,
        name: str = "structured-request",
        enable_thinking: bool = True,
        params: dict[str, Any] | None = None,
    ) -> ModelT:
        """Send a prompt and parse the answer into the given pydantic schema."""
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        content = self._send(model, messages, response_format=schema, name=name, enable_thinking=enable_thinking, params=params)
        return schema.model_validate_json(content)

    def embeddings(self, texts: Sequence[str], *, model: str) -> list[list[float]]:
        """Embed a batch of texts; returns one vector per text, in order."""
        response = self._client().embeddings.create(model=model, input=list(texts))
        return [item.embedding for item in response.data]
