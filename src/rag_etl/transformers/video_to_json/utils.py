from google.genai.types import Part
import json
import logging
from typing import Any

from rag_etl.transformers.video_to_json.schema import VideoUnderstandingResponse
from rag_etl.transformers.video_to_json.prompts import (
    video_understanding_prompt,
)

import requests
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def process_video_from_url(
    input_video_url: str,
    model_name: str,
    client: Any,
    download_path: str | Path,
) -> dict | None:
    """
    Download a video from a given Kaltura URL, upload it to Gemini for processing, and return JSON output.
    Initially, sending the URL directly to Gemini was unreliable: it did not return errors but analyzed only part of the videos.
    """

    try:
        download_path = Path(download_path)
        download_path.parent.mkdir(parents=True, exist_ok=True)

        # Download the video
        logger.info("Downloading video from %s", input_video_url)
        with requests.get(input_video_url, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            with open(download_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)

        logger.info("Downloaded video to %s", download_path)

        # Upload video file to Gemini
        logger.info("Uploading video file...")
        video_file = client.files.upload(file=str(download_path))

        # Wait for processing
        while video_file.state.name == "PROCESSING":
            logger.info("Waiting for video to be processed...")
            time.sleep(10)
            video_file = client.files.get(name=video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError(f"Video processing failed: {video_file.state.name}")

        logger.info("Video processed successfully. Generating content...")

        # Generate content
        response = client.models.generate_content(
            model=model_name,
            contents=[
                video_file,
                "\n\n",
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
