"""Recommended parameters per (model, mode), shared by every caller.

The project spec chooses the model; these are the parameters that come
with it. Modes: "thinking" (reasoning on) and "instruct" (reasoning off).
"""

DEFAULT_LLM_PARAMS: dict[tuple[str, str], dict] = {
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
    # Reasoning models: "thinking" mode maps to reasoning_effort=high
    ("deepseek-ai/DeepSeek-V4-Flash-0731", "thinking"): {"temperature": 1.0, "top_p": 0.95, "reasoning_effort": "high"},
    # https://huggingface.co/zai-org/GLM-5.3-Flash
    ("zai-org/GLM-5.3-Flash", "thinking"): {
        "reasoning_effort": "high",
        "temperature": 1.0,
        "top_p": 0.95,
    },
}
