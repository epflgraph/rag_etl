"""The unified RCP client: one entry point for every model invocation.

A client IS one configured model invocation: the model and its parameters
are one pairing, the endpoint and key come from the environment
(`RCP_BASE_URL`, `RCP_API_KEY`) unless passed in. `generate` is the one
way to ask a chat model, its options independent of each other: images
beside the prompt, a schema for the answer, a limit on its length.
`embeddings` is the other API. Failures are retried a few times and then
raised: either the run completes or it stops loud.
"""

import base64
import io
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar, overload

from langfuse import get_client
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel

import ragtools.mime_types as mt

ModelT = TypeVar("ModelT", bound=BaseModel)

# Closers vLLM may still leave inline in the content of thinking models
THINKING_CLOSER = "</think>"


class AnswerTooLong(Exception):
    """A streamed answer passed the caller's character limit and was cut off mid-generation."""


@lru_cache(maxsize=None)
def openai_client(base_url: str, api_key: str, max_retries: int, timeout: int) -> OpenAI:
    """One underlying client per config, shared across calls: no per-call connection setup."""
    return OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)


def _pil_data_uri(img: Image.Image) -> str:
    """Encode an in-memory image as a base64 PNG data URI."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def _json_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """The strict JSON-schema response format asking the model for this pydantic schema."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": schema.model_json_schema(),
            "strict": True,
        },
    }


def _user_content(
    prompt: str, images: Sequence[str | Path | Image.Image], detail: str | None
) -> str | list[dict[str, Any]]:
    """The user message: the prompt alone, or the prompt and every image, encoded inline.

    Images arrive as file paths or as in-memory PIL images (a page still
    being worked on); both become base64 data URIs.
    """
    if not images:
        return prompt

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image in images:
        if isinstance(image, Image.Image):
            image_url: dict[str, Any] = {"url": _pil_data_uri(image)}
        else:
            image = Path(image)
            mime_type = mt.guess_mime_type(str(image))
            if mime_type is None:
                raise ValueError(f"Could not determine MIME type for {image}")
            b64 = base64.b64encode(image.read_bytes()).decode("utf-8")
            image_url = {"url": f"data:{mime_type};base64,{b64}"}
        if detail:
            image_url["detail"] = detail
        content.append({"type": "image_url", "image_url": image_url})
    return content


@dataclass(frozen=True)
class RCPClient:
    """One configured model invocation: endpoint, key, model and its params.

    The model and its parameters are one pairing, chosen in
    ``ragtools.llm.catalog``. The endpoint and key are read from the
    environment when a call is made, unless given here. A failed request is
    retried a few times by the underlying client and then raised: a page
    that cannot be read stops the run rather than leaving a silent gap in
    the index.
    """

    model: str
    params: dict[str, Any] = field(default_factory=dict)
    base_url: str | None = None  # RCP_BASE_URL when not given
    api_key: str | None = None   # RCP_API_KEY when not given
    max_retries: int = 3
    timeout: int = 600

    def _client(self) -> OpenAI:
        # Read where they are used: a missing variable fails here, by its name
        base_url = self.base_url if self.base_url is not None else os.environ["RCP_BASE_URL"]
        api_key = self.api_key if self.api_key is not None else os.environ["RCP_API_KEY"]
        return openai_client(base_url, api_key, self.max_retries, self.timeout)

    @overload
    def generate(
        self,
        prompt: str,
        *,
        images: Sequence[str | Path | Image.Image] = ...,
        schema: None = None,
        system: str | None = ...,
        detail: str | None = ...,
        max_chars: int | None = ...,
        name: str = ...,
    ) -> str: ...

    @overload
    def generate(
        self,
        prompt: str,
        *,
        images: Sequence[str | Path | Image.Image] = ...,
        schema: type[ModelT],
        system: str | None = ...,
        detail: str | None = ...,
        max_chars: int | None = ...,
        name: str = ...,
    ) -> ModelT: ...

    def generate(
        self,
        prompt: str,
        *,
        images: Sequence[str | Path | Image.Image] = (),
        schema: type[ModelT] | None = None,
        system: str | None = None,
        detail: str | None = None,
        max_chars: int | None = None,
        name: str = "llm-request",
    ) -> str | ModelT:
        """Ask the model, traced in Langfuse, and return its answer.

        The options are independent of each other:

        - ``images``: sent beside the prompt (``detail`` is their resolution hint);
        - ``schema``: the answer is asked as that pydantic model and returned
          parsed; without one, the answer's text is returned;
        - ``system``: a system prompt;
        - ``max_chars``: the answer is streamed and cut the moment it passes
          the limit (`AnswerTooLong`), instead of being waited out to the end.

        A failure is raised loud, and a thinking trace left inline is dropped.
        """
        messages: list[dict[str, Any]] = [{"role": "user", "content": _user_content(prompt, images, detail)}]
        if system:
            messages = [{"role": "system", "content": system}, *messages]

        request: dict[str, Any] = {"model": self.model, "messages": messages, **self.params}
        if schema is not None:
            request["response_format"] = _json_schema(schema)

        client = self._client()
        with get_client().start_as_current_observation(
            as_type="generation", name=name, model=self.model, input=messages
        ) as generation:
            try:
                if max_chars is None:
                    content = self._whole(client, request, generation)
                else:
                    content = self._streamed(client, request, max_chars, generation)
            except AnswerTooLong:
                raise
            except Exception as e:
                raise RuntimeError(f"RCP request '{name}' with model {self.model} failed: {e}") from e

        # Some models still leave the thinking trace inline in the content
        if THINKING_CLOSER in content:
            content = content.split(THINKING_CLOSER)[-1].strip()

        return content if schema is None else schema.model_validate_json(content)

    def _whole(self, client: OpenAI, request: dict[str, Any], generation: Any) -> str:
        """Receive the answer whole, recording its usage and reasoning in the trace."""
        response = client.chat.completions.create(**request)

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

        return content

    def _streamed(self, client: OpenAI, request: dict[str, Any], max_chars: int, generation: Any) -> str:
        """Stream the answer and cut it the moment it passes ``max_chars``.

        A non-streaming request has a total-duration timeout: a generation
        slower than it is killed and restarted from zero, and an answer
        that never fits in it kills the run. A healthy stream has no such
        ceiling — chunks keep flowing — so the answer is read incrementally
        and cut as soon as it grows past the limit.
        """
        parts: list[str] = []
        try:
            stream = client.chat.completions.create(**request, stream=True)
            total = 0
            for chunk in stream:
                if not chunk.choices:
                    continue
                piece = chunk.choices[0].delta.content
                if not piece:
                    continue
                parts.append(piece)
                total += len(piece)
                if total > max_chars:
                    stream.close()
                    raise AnswerTooLong(f"the answer was cut off at {total:,} chars")
        finally:
            generation.update(output="".join(parts).strip())

        return "".join(parts).strip()

    def embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts; returns one vector per text, in order."""
        response = self._client().embeddings.create(model=self.model, input=list(texts))
        return [item.embedding for item in response.data]
