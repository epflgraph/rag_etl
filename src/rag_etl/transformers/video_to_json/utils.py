from google.genai.types import Part
import json
import logging
from typing import Any


from rag_etl.transformers.video_to_json.schema import VideoUnderstandingResponse
from rag_etl.transformers.video_to_json.prompts import (
    video_understanding_prompt,
)

logger = logging.getLogger(__name__)


def process_video_from_url(
    input_video_url: str, model_name: str, client: Any, mime_type: str
) -> dict | None:
    """Process a single video URL with Gemini and returns a JSON output"""

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[
                Part.from_uri(
                    file_uri=input_video_url,
                    mime_type=mime_type,
                ),
                video_understanding_prompt,
            ],
            config={
                "response_mime_type": "application/json",
                "response_schema": VideoUnderstandingResponse,
            },
        )

        text = getattr(response, "text", None)
        if not text:
            logger.error("Empty response.text from model for %s", input_video_url)
            return None

        # Parses the response JSON
        try:
            response_data = json.loads(response.text)
            return response_data
        except json.JSONDecodeError:
            logger.error(
                "Non-JSON response for video %s  %r",
                input_video_url,
                text[:200],
            )
            return None

    except Exception:
        logger.exception("Error processing video %s: ", input_video_url)
        return None
