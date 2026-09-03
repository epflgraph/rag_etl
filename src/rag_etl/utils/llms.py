import base64

from langfuse import get_client
from openai import OpenAI

import rag_etl.utils.mime_types as mt

from rag_etl.config import CONFIG


# Recommended params for each model and mode
LLM_PARAMS = {
    # https://huggingface.co/Qwen/Qwen3.6-35B-A3B
    # Recommended for precise coding tasks (e.g. WebDev).
    ("Qwen/Qwen3.6-35B-A3B", "thinking"): {
        "temperature": 0.6,
        "top_p": 0.95,
        "presence_penalty": 0.0,
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": True},
            # Value from benchmark "Reasoning Budgets vs. Structured CoT Controlling LLM Thinking Tokens"
            # (https://kaitchup.substack.com/p/reasoning-budgets-vs-structured-cot)
            "thinking_token_budget": 32000,
        },
    },
    # https://huggingface.co/Qwen/Qwen3.6-35B-A3B
    ("Qwen/Qwen3.6-35B-A3B", "instruct"): {
        "temperature": 0.7,
        "top_p": 0.8,
        "presence_penalty": 1.5,
        "extra_body": {
            "top_k": 20,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    },
}


def send_llm_request(
    model, messages, response_format=None, name: str = "llm-request", enable_thinking: bool = True, timeout: int = 600
):
    if response_format:
        response_format_schema = {
            "type": "json_schema",
            "json_schema": {
                "name": type(response_format).__name__,
                "schema": response_format.model_json_schema(),
                "strict": True,
            },
        }
    else:
        response_format_schema = None

    # Send request and trace it as a single generation
    rcp_client = OpenAI(base_url=CONFIG["RCP_BASE_URL"], api_key=CONFIG["RCP_API_KEY"], timeout=timeout)
    langfuse = get_client()
    with langfuse.start_as_current_observation(
        as_type="generation",
        name=name,
        model=model,
        input=messages,
    ) as generation:
        if enable_thinking:
            mode = "thinking"
        else:
            mode = "instruct"

        params = LLM_PARAMS[(model, mode)]

        response = rcp_client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=response_format_schema,
            **params,
        )
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

    # Strip thinking tokens
    thinking_tag = "</think>"
    if thinking_tag in response:
        content = content.split(thinking_tag)[-1].strip()

    # Return parsed result if structured output
    if response_format:
        return response_format.model_validate_json(content)

    # Return string otherwise
    return content


def generate_alt_text(path: str) -> str:
    # Guess MIME type from extension
    mime_type = mt.guess_mime_type(path)

    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for {path}")

    # Encode file to base64
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    # Build data URL
    data_url = f"data:{mime_type};base64,{b64}"

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Generate the ALT text for this image. If prominent text exists, include it briefly.",
                },
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]

    rcp_model = CONFIG["RCP_VISION_MODEL"]
    message = send_llm_request(rcp_model, messages, name="generate-alt-text")

    return message
