"""The unified RCP client: one entry point for every model invocation.

Vision, text, structured output and embeddings all go through here. A
client IS one configured model invocation: the model and its parameters
are one pairing, the endpoint and key come from the environment unless
passed in. Failures are retried a few times and then raised: either the
run completes or it stops loud.
"""

from __future__ import annotations

import base64
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

from langfuse import get_client
from openai import OpenAI
from pydantic import BaseModel

import ragtools.mime_types as mt

ModelT = TypeVar("ModelT", bound=BaseModel)

# Closers vLLM may still leave inline in the content of thinking models
THINKING_CLOSER = "</think>"


@lru_cache(maxsize=None)
def openai_client(base_url: str, api_key: str, max_retries: int, timeout: int) -> OpenAI:
    """One underlying client per config, shared across calls: no per-call connection setup."""
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)


@dataclass(frozen=True)
class RCPClient:
    """One configured model invocation: endpoint, key, model and its params.

    The model and its parameters are one pairing, chosen in
    ``ragtools.llm.defaults``; the endpoint and key come from the
    environment unless passed in. A failed request is retried a few times
    by the underlying client and then raised: a page that cannot be read
    stops the run rather than leaving a silent gap in the index.
    """

    model: str
    params: dict[str, Any] = field(default_factory=dict)
    base_url: str = field(default_factory=lambda: os.environ.get("RCP_BASE_URL", ""))
    api_key: str = field(default_factory=lambda: os.environ.get("RCP_API_KEY", ""))
    max_retries: int = 3
    timeout: int = 600

    def _client(self) -> OpenAI:
        if not self.base_url:
            raise ValueError("Missing required environment variable: RCP_BASE_URL")
        if not self.api_key:
            raise ValueError("Missing required environment variable: RCP_API_KEY")
        return openai_client(self.base_url, self.api_key, self.max_retries, self.timeout)

    def _send(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: type[BaseModel] | None = None,
        system: str | None = None,
        name: str,
    ) -> str:
        if system:
            messages = [{"role": "system", "content": system}, *messages]

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
        with get_client().start_as_current_observation(as_type="generation", name=name, model=self.model, input=messages) as generation:
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format=response_format_schema,
                    **self.params,
                )  # type: ignore[call-overload]
            except Exception as e:
                raise RuntimeError(f"RCP request '{name}' with model {self.model} failed: {e}") from e

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
        system: str | None = None,
        name: str = "llm-request",
    ) -> str:
        """Send a plain text prompt and return the model's answer."""
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        return self._send(messages, system=system, name=name)

    def vision(
        self,
        prompt: str,
        images: Sequence[str | Path],
        *,
        system: str | None = None,
        detail: str | None = None,
        name: str = "vision-request",
    ) -> str:
        """Send a prompt with one or more images (paths) and return the model's answer."""
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for image in images:
            image = Path(image)
            mime_type = mt.guess_mime_type(str(image))
            if mime_type is None:
                raise ValueError(f"Could not determine MIME type for {image}")
            b64 = base64.b64encode(image.read_bytes()).decode("utf-8")
            image_url: dict[str, Any] = {"url": f"data:{mime_type};base64,{b64}"}
            if detail:
                image_url["detail"] = detail
            content.append({"type": "image_url", "image_url": image_url})

        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        return self._send(messages, system=system, name=name)

    def structured(
        self,
        prompt: str,
        schema: type[ModelT],
        *,
        system: str | None = None,
        name: str = "structured-request",
    ) -> ModelT:
        """Send a prompt and parse the answer into the given pydantic schema."""
        messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        content = self._send(messages, response_format=schema, system=system, name=name)
        return schema.model_validate_json(content)

    def embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts; returns one vector per text, in order."""
        response = self._client().embeddings.create(model=self.model, input=list(texts))
        return [item.embedding for item in response.data]
