from __future__ import annotations

import base64
import mimetypes
import os
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
        try:
            response = self._session.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.config.api_timeout,
            )
        except Exception as exc:
            raise VlmCaptionerError(f"VLM API request failed: {exc}") from exc

        if not getattr(response, "ok", False):
            detail = _response_error(response)
            raise VlmCaptionerError(
                f"VLM API returned HTTP {response.status_code}: {detail}"
            )

        try:
            return response.json()
        except Exception as exc:
            raise VlmCaptionerError(
                "VLM API returned invalid JSON"
            ) from exc

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
        self._model = config.api_model.strip()
        self._base_url = (
            config.api_base_url.strip()
            or "https://generativelanguage.googleapis.com/v1beta"
        ).rstrip("/")

    def caption(self, image_path: Path | str) -> str:
        path = Path(image_path).expanduser().resolve()
        mime_type, encoded = _image_payload(path)

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": self.config.prompt},
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": encoded,
                            }
                        },
                    ],
                }
            ],
            "generationConfig": {
                "temperature": self.config.temperature,
                "topP": self.config.top_p,
                "topK": self.config.top_k,
                "maxOutputTokens": self.config.max_tokens,
            },
        }
        url = f"{self._base_url}/models/{self._model}:generateContent"

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

        text = "".join(
            str(part["text"])
            for part in parts
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        ).strip()

        if not text:
            raise VlmCaptionerError("Gemini API returned empty caption text")
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
                            "text": self.config.prompt,
                        },
                    ],
                }
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
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
