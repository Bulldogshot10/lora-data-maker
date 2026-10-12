from __future__ import annotations

import base64
import json
import mimetypes
import os
import time

from PIL import Image, UnidentifiedImageError
from pathlib import Path
from typing import Any

from .caption_vlm import VlmCaptionerError, VlmConfig, _data_uri
from .security import MAX_IMAGE_PIXELS, MAX_IMAGE_UPLOAD_BYTES, assert_safe_outbound_url, validate_api_base_url


def _image_payload(path: Path) -> tuple[str, str]:
    try:
        size = path.stat().st_size
        if size < 1 or size > MAX_IMAGE_UPLOAD_BYTES:
            raise VlmCaptionerError("image exceeds the 25 MiB API caption limit")
        with Image.open(path) as image:
            width, height = image.size
            if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
                raise VlmCaptionerError(
                    f"unsafe image dimensions {width}x{height}; maximum is {MAX_IMAGE_PIXELS} pixels"
                )
            detected_format = (image.format or "").upper()
        payload = path.read_bytes()
    except VlmCaptionerError:
        raise
    except (OSError, UnidentifiedImageError) as exc:
        raise VlmCaptionerError(
            f"could not read a valid image {path}: {exc}"
        ) from exc

    mime_by_format = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
        "BMP": "image/bmp",
        "GIF": "image/gif",
        "TIFF": "image/tiff",
        "AVIF": "image/avif",
    }
    mime_type = mime_by_format.get(detected_format) or mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return mime_type, base64.b64encode(payload).decode("ascii")


MAX_API_RESPONSE_BYTES = 2 * 1024 * 1024


def _close_response(response: Any) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        close()


def _read_response_bytes(
    response: Any,
    *,
    limit: int,
    reject_oversized: bool,
) -> bytes | None:
    headers = getattr(response, "headers", {}) or {}
    raw_length = headers.get("Content-Length", "") if hasattr(headers, "get") else ""
    if raw_length:
        try:
            declared = int(raw_length)
        except (TypeError, ValueError) as exc:
            _close_response(response)
            raise VlmCaptionerError("VLM API returned an invalid Content-Length") from exc
        if declared < 0:
            _close_response(response)
            raise VlmCaptionerError("VLM API returned an invalid Content-Length")
        if declared > limit and reject_oversized:
            _close_response(response)
            raise VlmCaptionerError("VLM API response exceeds the 2 MiB limit")

    iterator = getattr(response, "iter_content", None)
    if not callable(iterator):
        return None

    body = bytearray()
    overflow = False
    try:
        for chunk in iterator(chunk_size=64 * 1024):
            if not chunk:
                continue
            remaining = limit - len(body)
            if len(chunk) > remaining:
                body.extend(chunk[:remaining])
                overflow = True
                break
            body.extend(chunk)
        if overflow and reject_oversized:
            raise VlmCaptionerError("VLM API response exceeds the 2 MiB limit")
        return bytes(body)
    finally:
        _close_response(response)


def _response_json(response: Any) -> Any:
    body = _read_response_bytes(
        response,
        limit=MAX_API_RESPONSE_BYTES,
        reject_oversized=True,
    )
    try:
        return json.loads(body.decode("utf-8")) if body is not None else response.json()
    except VlmCaptionerError:
        raise
    except Exception as exc:
        raise VlmCaptionerError("VLM API returned invalid or oversized JSON") from exc


def _response_error(response: Any) -> str:
    try:
        body = _read_response_bytes(
            response,
            limit=4096,
            reject_oversized=False,
        )
        data = json.loads(body.decode("utf-8")) if body is not None else response.json()
    except Exception:
        text = getattr(response, "text", "")
        return str(text).strip()[:500] or f"HTTP {response.status_code}"

    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("detail") or error.get("code")
            if message:
                return str(message)[:500]
        if error:
            return str(error)[:500]
        message = data.get("message") or data.get("detail")
        if message:
            return str(message)[:500]
    return repr(data)[:500]

def _extract_openai_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise VlmCaptionerError("OpenAI-compatible API returned no choices")

    choice = choices[0]
    if not isinstance(choice, dict):
        raise VlmCaptionerError("OpenAI-compatible API returned an invalid choice")

    message = choice.get("message")
    if not isinstance(message, dict):
        raise VlmCaptionerError("OpenAI-compatible API returned no message")

    content = message.get("content")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in {"text", "output_text"} and isinstance(
                part.get("text"), str
            ):
                text_parts.append(part["text"])
        return "".join(text_parts)

    raise VlmCaptionerError(
        "OpenAI-compatible API returned an unsupported message.content format"
    )


API_VLM_PROMPT = (
    "caption this image for a stable diffusion character lora dataset. "
    "return exactly these 5 fields, in this exact order, one field per line. "
    "copy the field labels exactly as written in the required template below. "
    "each field value must be a short phrase, maximum 7 words. "
    "in the pose field, include shot type and body pose, using visible framing only. "
    "choose from extreme close-up, close-up, medium close-up, medium shot, medium-long shot, cowboy shot, full-body shot, wide shot. "
    "examples: close-up portrait; medium shot standing; cowboy shot standing; full-body shot standing; wide shot seated. "
    "describe the visible camera angle or view; describe visible clothing; "
    "describe visible expression; describe the visible background and include lighting. "
    "do not mention hair color, eye color, face, age, or gender. "
    "do not mention image quality, watermarks, text, metadata, ai, or this task. "
    "do not guess hidden or ambiguous details. "
    "replace every placeholder with a real short phrase; never output the angle-bracket placeholders. "
    "required template, with exactly these labels and line breaks:\n"
    "pose: <body pose>\n"
    "angle camera: <camera angle or view>\n"
    "clothing: <visible clothing>\n"
    "expression: <visible expression>\n"
    "background: <background and lighting>\n"
    "return only those five lines, no introduction, explanation, bullets, markdown, or quotes."
)



class _ApiVlmCaptioner:
    def __init__(self, config: VlmConfig) -> None:
        self.config = config
        key_env = config.api_key_env.strip()
        if not key_env:
            raise VlmCaptionerError(
                f"caption.vlm.api_key_env is required for {config.runtime} runtime"
            )

        self._api_key = os.environ.get(key_env, "").strip()
        if not self._api_key:
            raise VlmCaptionerError(
                f"environment variable {key_env!r} is not set; "
                f"export your API key before starting the caption job"
            )

        self._session = None
        try:
            import requests

            self._session = requests.Session()
            # Environment-configured proxies must not silently reroute credentialed requests.
            self._session.trust_env = False
        except ImportError as exc:
            raise VlmCaptionerError(
                "API VLM captions require the requests package"
            ) from exc

    def _post(self, url: str, *, headers: dict[str, str], payload: dict[str, Any]) -> Any:
        assert self._session is not None
        retryable_statuses = {429, 500, 502, 503, 504}

        for attempt in range(3):
            try:
                safe_url = assert_safe_outbound_url(url, allow_loopback=True)
                response = self._session.post(
                    safe_url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.api_timeout,
                    allow_redirects=False,
                    stream=True,
                )
            except ValueError as exc:
                raise VlmCaptionerError(f"VLM API destination rejected: {exc}") from exc
            except Exception as exc:
                raise VlmCaptionerError(f"VLM API request failed: {exc}") from exc

            if 300 <= response.status_code < 400:
                raise VlmCaptionerError(
                    "VLM API endpoint returned a redirect; redirects are disabled to protect credentials"
                )

            if getattr(response, "ok", False):
                return _response_json(response)

            if response.status_code not in retryable_statuses or attempt == 2:
                detail = _response_error(response).replace(self._api_key, "[redacted]")
                raise VlmCaptionerError(
                    f"VLM API returned HTTP {response.status_code}: {detail}"
                )

            retry_after = ""
            headers_obj = getattr(response, "headers", None)
            if headers_obj is not None:
                retry_after = str(headers_obj.get("Retry-After", "")).strip()
            try:
                delay = min(float(retry_after), 10.0) if retry_after else 1.0 * (attempt + 1)
            except (TypeError, ValueError):
                delay = 1.0 * (attempt + 1)
            time.sleep(max(0.0, delay))

    def close(self) -> None:
        session = self._session
        self._session = None
        if session is not None:
            try:
                session.close()
            except Exception:
                pass

class GeminiVlmCaptioner(_ApiVlmCaptioner):
    """VLM caption runtime backed by the Google Gemini REST API."""

    def __init__(self, config: VlmConfig) -> None:
        super().__init__(config)
        if not config.api_model.strip():
            raise VlmCaptionerError(
                "caption.vlm.api_model is required for the Gemini runtime"
            )
        self._model = config.api_model.strip().removeprefix("models/")
        raw_base_url = (
            config.api_base_url.strip()
            or "https://generativelanguage.googleapis.com/v1beta"
        ).rstrip("/")
        try:
            self._base_url = validate_api_base_url(raw_base_url)
        except ValueError as exc:
            raise VlmCaptionerError(f"invalid Gemini API base URL: {exc}") from exc

    def caption(self, image_path: Path | str) -> str:
        path = Path(image_path).expanduser().resolve()
        mime_type, encoded = _image_payload(path)
        url = f"{self._base_url}/models/{self._model}:generateContent"

        def request_caption(prompt: str) -> str:
            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": encoded,
                                }
                            },
                            {"text": prompt},
                        ],
                    }
                ],
            }
            data = self._post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self._api_key,
                },
                payload=payload,
            )

            try:
                candidates = data["candidates"]
                parts = candidates[0]["content"]["parts"]
            except (KeyError, IndexError, TypeError) as exc:
                raise VlmCaptionerError(
                    "Gemini API returned no candidate text"
                ) from exc

            answer_parts: list[str] = []
            for part in parts:
                if not isinstance(part, dict):
                    continue
                if part.get("thought") is True:
                    continue
                value = part.get("text")
                if isinstance(value, str) and value.strip():
                    answer_parts.append(value.strip())

            text = "\n".join(answer_parts).strip()
            if not text:
                finish_reason = candidates[0].get("finishReason") if isinstance(candidates[0], dict) else None
                raise VlmCaptionerError(
                    "Gemini API returned no visible caption text"
                    + (f" (finishReason={finish_reason!r})" if finish_reason else "")
                )
            return text

        return request_caption(API_VLM_PROMPT)


class OpenAICompatibleVlmCaptioner(_ApiVlmCaptioner):
    """VLM caption runtime backed by an OpenAI-compatible Chat Completions API."""

    def __init__(self, config: VlmConfig) -> None:
        super().__init__(config)
        if not config.api_model.strip():
            raise VlmCaptionerError(
                "caption.vlm.api_model is required for the OpenAI-compatible runtime"
            )
        self._model = config.api_model.strip()
        raw_base_url = (
            config.api_base_url.strip()
            or "https://api.openai.com/v1"
        ).rstrip("/")
        try:
            base_url = validate_api_base_url(raw_base_url)
        except ValueError as exc:
            raise VlmCaptionerError(f"invalid OpenAI-compatible API base URL: {exc}") from exc
        self._url = (
            base_url
            if base_url.endswith("/chat/completions")
            else f"{base_url}/chat/completions"
        )

    def caption(self, image_path: Path | str) -> str:
        path = Path(image_path).expanduser().resolve()
        image_url = _data_uri(path)

        def request_caption(prompt: str) -> str:
            payload = {
                "model": self._model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
            }
            data = self._post(
                self._url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                payload=payload,
            )
            text = _extract_openai_text(data).strip()
            if not text:
                raise VlmCaptionerError(
                    "OpenAI-compatible API returned empty caption text"
                )
            return text

        return request_caption(API_VLM_PROMPT)
