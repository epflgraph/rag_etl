"""The settled models: one pairing per concern, chosen per semester.

A model name means nothing without its parameters — the same Qwen with or
without thinking behaves in many ways — so each entry below IS a model: an
RCPClient built with its parameters and the endpoint from the environment.
Choosing a different option for a semester means editing this file.
"""

from ragtools.llm.rcp import RCPClient

MODELS: dict[tuple[str, str], RCPClient] = {
    # https://huggingface.co/Qwen/Qwen3.6-35B-A3B — recommended parameters per mode
    ("Qwen/Qwen3.6-35B-A3B", "thinking"): RCPClient(
        model="Qwen/Qwen3.6-35B-A3B",
        params={
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
    ),
    ("Qwen/Qwen3.6-35B-A3B", "instruct"): RCPClient(
        model="Qwen/Qwen3.6-35B-A3B",
        params={
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
    ),
    # Reasoning models: thinking maps to reasoning_effort=high
    ("deepseek-ai/DeepSeek-V4-Flash-0731", "thinking"): RCPClient(
        model="deepseek-ai/DeepSeek-V4-Flash-0731",
        params={"temperature": 1.0, "top_p": 0.95, "reasoning_effort": "high"},
    ),
    # https://huggingface.co/zai-org/GLM-5.3-Flash
    ("zai-org/GLM-5.3-Flash", "thinking"): RCPClient(
        model="zai-org/GLM-5.3-Flash",
        params={"reasoning_effort": "high", "temperature": 1.0, "top_p": 0.95},
    ),
    ("Qwen/Qwen3-Embedding-8B", "embedding"): RCPClient(model="Qwen/Qwen3-Embedding-8B"),
}

# One entry per concern, chosen from the catalog
_TASKS: dict[str, tuple[str, str]] = {
    "ocr": ("Qwen/Qwen3.6-35B-A3B", "instruct"),
    "judge": ("Qwen/Qwen3.6-35B-A3B", "thinking"),
    "split_exercises": ("Qwen/Qwen3.6-35B-A3B", "thinking"),
    "stitch": ("Qwen/Qwen3.6-35B-A3B", "instruct"),
    "embedding": ("Qwen/Qwen3-Embedding-8B", "embedding"),
}


def get_model(task: str) -> RCPClient:
    """The client serving one task, from the catalog; fails loud on an unknown task."""
    try:
        return MODELS[_TASKS[task]]
    except KeyError:
        raise ValueError(f"Unknown task '{task}': add it to ragtools/llm/defaults.py") from None


# Legacy: send_llm_request keys parameters by (model, mode)
DEFAULT_LLM_PARAMS: dict[tuple[str, str], dict] = {key: client.params for key, client in MODELS.items()}
