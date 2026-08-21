from rag_etl.transformers import BaseTransformer
from rag_etl.resources import BaseResource
from google import genai
from pathlib import Path
import json
import logging
from collections.abc import Sequence
from rag_etl.transformers.video_to_json.utils import process_video_from_url


import rag_etl.utils.mime_types as mt

from rag_etl.config import CONFIG

logger = logging.getLogger(__name__)


class VideoToJSONTransformer(BaseTransformer):
    """
    Transformer that downloads a short video from kaltura, and uses Gemini to generate a JSON with all the content extracted from the video.
    """

    def transform(self, resources: Sequence[BaseResource]) -> list[BaseResource]:
        transformed_resources: list[BaseResource] = []

        # Load environment variables
        google_api_key = CONFIG["GOOGLE_API_KEY"]

        if not google_api_key:
            raise RuntimeError("Missing GOOGLE_API_KEY in CONFIG")

        client = genai.Client(api_key=google_api_key)

        for resource in resources:
            # Skip if resource is not an MP4
            if resource.mime_type != mt.MP4:
                transformed_resources.append(resource)
                continue

            p = Path(resource.path)
            if len(p.parents) < 3:
                logger.error("VideoToJSONTransformer: unexpected path depth: %s", p)
                transformed_resources.append(resource)
                continue

            safe_stem = p.stem.replace(" ", "_")

            # Build paths of JSON file
            new_filename = f"{safe_stem}_{resource.model}.json"

            downloaded_video_filename = f"{safe_stem}_{resource.model}.mp4"

            # parents[1] goes up 2 levels
            json_path = p.parents[1] / "video" / new_filename
            # json_path.parent.mkdir(parents=True, exist_ok=True)

            downloaded_video_path = p.parents[1] / "video" / downloaded_video_filename

            # Only process video if not cached
            cached = self.get_from_cache(resource.path, json_path)
            if not cached:
                logger.debug(f"Converting {resource.path} into {json_path}")
                json_data = process_video_from_url(
                    input_video_url=resource.url,
                    model_name=resource.model,
                    client=client,
                    download_path=downloaded_video_path,
                )

                if json_data is None:
                    # TODO retry on failure or at least print the content or plausible reason why it failed
                    logger.debug(
                        f"Failed to convert {resource.path} into {json_path}. Skipping to prevent a failure later on..."
                    )
                    continue

                json_path.write_text(
                    json.dumps(json_data, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                self.set_to_cache(resource.path, json_path)

            # Build transformed resource and append it
            new_resource = resource.copy_with(
                path=str(json_path),
                mime_type=mt.JSON,
                processing_method=None,
                model=None,
            )
            transformed_resources.append(new_resource)

        return transformed_resources
