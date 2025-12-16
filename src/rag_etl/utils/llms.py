import base64

from openai import OpenAI

import rag_etl.utils.mime_types as mt

from rag_etl.config import CONFIG


def send_llm_request(model, messages, response_format=None):
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

    # Send request
    rcp_client = OpenAI(base_url=CONFIG['RCP_BASE_URL'], api_key=CONFIG['RCP_API_KEY'])
    response = rcp_client.chat.completions.create(model=model, messages=messages, response_format=response_format_schema)
    content = response.choices[0].message.content.strip()

    # Strip thinking tokens
    thinking_tag = '</think>'
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

    messages = [{'role': 'user', 'content': [
        {"type": "text", "text": "Generate the ALT text for this image. If prominent text exists, include it briefly."},
        {"type": "image_url", "image_url": {"url": data_url}}
    ]}]

    rcp_model = CONFIG['RCP_VISION_MODEL']
    message = send_llm_request(rcp_model, messages)

    return message
