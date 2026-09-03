import logging
import time

import requests

from rag_etl.config import CONFIG

logger = logging.getLogger(__name__)

POLL_SECONDS = 10

# one hour deadline to finishe task
MAX_WAIT_SECONDS = 3600

REQUEST_TIMEOUT = 120


class GraphAIError(Exception):
    """
    Raised when GraphAI cannot be reached, answers with an error, or cannot do
    what was asked of it.

    Callers get one thing to catch, and do not have to know that the service is
    reached over HTTP.
    """


def request_json(method: str, url: str, headers: dict[str, str], **kwargs) -> dict:
    """
    Send one request to GraphAI and return its decoded body.

    Every way the call can fail is reported as GraphAIError: the host being
    unreachable, a timeout, an error status, or a body that is not JSON.
    """

    try:
        response = requests.request(method, url, headers=headers, timeout=REQUEST_TIMEOUT, **kwargs)
        response.raise_for_status()
    except requests.RequestException as error:
        raise GraphAIError(f"{method} {url} failed: {error}") from error

    try:
        return response.json()
    except ValueError as error:
        raise GraphAIError(f"{method} {url} answered with a body that is not JSON") from error


def get_access_token() -> str:
    """Log in to GraphAI and return a bearer token."""

    payload = request_json(
        "POST",
        f"{CONFIG['GRAPH_AI_URL']}/token",
        headers={"accept": "application/json", "Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "",
            "username": CONFIG["GRAPH_AI_USERNAME"],
            "password": CONFIG["GRAPH_AI_PASSWORD"],
            "scope": "",
            "client_id": "",
            "client_secret": "",
        },
    )

    access_token = payload.get("access_token")
    if not access_token:
        raise GraphAIError(f"GraphAI did not return an access token: {payload}")

    return access_token


def json_headers(token: str) -> dict[str, str]:
    """Headers for a GraphAI call that sends and receives JSON."""

    return {"accept": "application/json", "Content-Type": "application/json", "Authorization": f"Bearer {token}"}


def poll_task(endpoint: str, task_id: str, token: str) -> dict:
    """
    Poll a GraphAI task until it leaves PENDING, and return its response.

    Work is queued rather than answered inline: slide detection runs at
    roughly a tenth of the video's duration, so a 90 minute lecture takes
    about nine minutes here.
    """

    headers = {"accept": "application/json", "Authorization": f"Bearer {token}"}
    deadline = time.time() + MAX_WAIT_SECONDS

    while time.time() < deadline:
        payload = request_json("GET", f"{CONFIG['GRAPH_AI_URL']}/{endpoint}/status/{task_id}", headers=headers)

        if payload["task_status"] != "PENDING":
            return payload

        logger.debug(f"{endpoint}: still working")
        time.sleep(POLL_SECONDS)

    raise GraphAIError(f"{endpoint} did not finish within {MAX_WAIT_SECONDS}s")


def retrieve_video(video_url: str, token: str) -> str:
    """Ask GraphAI to fetch the video, and return the token identifying it."""

    response = request_json(
        "POST",
        f"{CONFIG['GRAPH_AI_URL']}/video/retrieve_url",
        headers=json_headers(token),
        json={"url": video_url, "force": False, "playlist": False},
    )

    payload = poll_task("video/retrieve_url", response["task_id"], token)
    result = payload.get("task_result") or {}

    video_token = result.get("token")
    if not video_token:
        raise GraphAIError(f"GraphAI could not retrieve {video_url}: {payload}")

    return video_token


def detect_slides(video_token: str, token: str, language: str | None = None) -> list[int]:
    """
    Run slide detection on a retrieved video and return the change timestamps
    in seconds, sorted.

    The service also reports a final timestamp equal to the video duration,
    which marks the end rather than a slide, so callers get it too and should
    treat the last value as a bound.
    """

    response = request_json(
        "POST",
        f"{CONFIG['GRAPH_AI_URL']}/video/detect_slides",
        headers=json_headers(token),
        json={"token": video_token, "force_non_self": True, "force": False, "language": language or "en"},
    )

    payload = poll_task("video/detect_slides", response["task_id"], token)
    result = payload.get("task_result") or {}

    if not result.get("successful"):
        raise GraphAIError(f"Slide detection failed for {video_token}: {payload}")

    timestamps = []
    for slide in (result.get("slide_tokens") or {}).values():
        timestamps.append(int(slide["timestamp"]))

    return sorted(timestamps)
