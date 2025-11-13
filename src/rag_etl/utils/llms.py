import base64

from openai import OpenAI

import rag_etl.utils.mime_types as mt

from rag_etl.config import CONFIG


def send_llm_request(model, messages, response_format=None):
    rcp_client = OpenAI(base_url=CONFIG['RCP_BASE_URL'], api_key=CONFIG['RCP_API_KEY'])

    if response_format:
        response = rcp_client.chat.completions.parse(model=model, messages=messages, response_format=response_format)
        return response.choices[0].message.parsed
    else:
        response = rcp_client.chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content.strip()


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

    messages = [{'role': 'user', 'content': [
        {"type": "text", "text": "Generate the ALT text for this image. If prominent text exists, include it briefly."},
        {"type": "image_url", "image_url": {"url": data_url}}
    ]}]

    rcp_model = 'Qwen/Qwen2.5-VL-72B-Instruct'

    message = send_llm_request(rcp_model, messages)

    return message
