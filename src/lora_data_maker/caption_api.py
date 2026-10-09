from __future__ import annotations

import base64
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

from .caption_vlm import VlmCaptionerError, VlmConfig, _data_uri


def _image_payload(path: Path) -> tuple[str, str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise VlmCaptionerError(
            f"could not read processed image {path}: {exc}"
        ) from exc

    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return mime_type, base64.b64encode(payload).decode("ascii")


def _response_error(response: Any) -> str:
    try:
        data = response.json()
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
    "each field value must be a short phrase, maximum 5 words. "
    "describe body pose; describe the visible camera angle or view; describe visible clothing; "
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


_CAPTION_FIELD_LABEL_RE = re.compile(
    r"""(?im)(?<![A-Za-z0-9_])["']?(pose|angle[ _-]+camera|camera[ _-]+angle|clothing|expression|background)["']?\s*:"""
)


def _has_all_caption_fields(text: str) -> bool:
    found: set[str] = set()
    for match in _CAPTION_FIELD_LABEL_RE.finditer(str(text)):
        name = re.sub(r"[ _-]+", " ", match.group(1).lower()).strip()
        if name == "camera angle":
            name = "angle camera"
        found.add(name)
    return {"pose", "angle camera", "clothing", "expression", "background"} <= found


def _format_retry_prompt(previous: str) -> str:
    return (
        "Your previous response did not follow the required format. "
        "Inspect the image again and rewrite the answer using exactly these five labeled lines, "
        "in this order. Each value must be a short phrase of at most 5 words. "
        "Do not return a paragraph or JSON. Return only the lines below, replacing the placeholders:\n"
        "pose: <body pose>\n"
        "angle camera: <camera angle or view>\n"
        "clothing: <visible clothing>\n"
        "expression: <visible expression>\n"
        "background: <background and lighting>\n"
        "Previous response to reformat: " + str(previous).strip()[:2000]
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
        except ImportError as exc:
            raise VlmCaptionerError(
                "API VLM captions require the requests package"
            ) from exc

    def _post(self, url: str, *, headers: dict[str, str], payload: dict[str, Any]) -> Any:
        assert self._session is not None
        retryable_statuses = {429, 500, 502, 503, 504}

        for attempt in range(3):
            try:
                response = self._session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.api_timeout,
                )
            except Exception as exc:
                raise VlmCaptionerError(f"VLM API request failed: {exc}") from exc

            if getattr(response, "ok", False):
                try:
                    return response.json()
                except Exception as exc:
                    raise VlmCaptionerError(
                        "VLM API returned invalid JSON"
                    ) from exc

            if response.status_code not in retryable_statuses or attempt == 2:
                detail = _response_error(response)
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
        self._base_url = (
            config.api_base_url.strip()
            or "https://generativelanguage.googleapis.com/v1beta"
        ).rstrip("/")

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

        text = request_caption(API_VLM_PROMPT)
        if not _has_all_caption_fields(text):
            text = request_caption(_format_retry_prompt(text))
        return text


class OpenAICompatibleVlmCaptioner(_ApiVlmCaptioner):
    """VLM caption runtime backed by an OpenAI-compatible Chat Completions API."""

    def __init__(self, config: VlmConfig) -> None:
        super().__init__(config)
        if not config.api_model.strip():
            raise VlmCaptionerError(
                "caption.vlm.api_model is required for the OpenAI-compatible runtime"
            )
        self._model = config.api_model.strip()
        base_url = (
            config.api_base_url.strip()
            or "https://api.openai.com/v1"
        ).rstrip("/")
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

        text = request_caption(API_VLM_PROMPT)
        if not _has_all_caption_fields(text):
            text = request_caption(_format_retry_prompt(text))
        return text
