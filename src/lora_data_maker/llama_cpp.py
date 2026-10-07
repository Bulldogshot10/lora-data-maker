from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests


class LlamaCppError(RuntimeError):
    """Raised when a llama.cpp server request fails."""


class LlamaCppClient:
    """Small OpenAI-compatible client for a local llama-server instance."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8080/v1",
        model: str = "local-model",
        timeout: float = 120.0,
        session: requests.Session | None = None,
    ) -> None:
        value = str(base_url).strip().rstrip("/")
        if not value:
            raise ValueError("base_url must not be empty")
        if not str(model).strip():
            raise ValueError("model must not be empty")
        if timeout <= 0:
            raise ValueError("timeout must be greater than zero")

        self.base_url = value
        self.model = str(model).strip()
        self.timeout = float(timeout)
        self.session = session or requests.Session()

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    @property
    def models_url(self) -> str:
        return f"{self.base_url}/models"

    def health(self) -> bool:
        """Return whether the configured llama-server responds successfully."""
        try:
            response = self.session.get(self.models_url, timeout=self.timeout)
            return response.ok
        except requests.RequestException:
            return False

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 128,
        response_format: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> str:
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        if response_format is not None:
            payload["response_format"] = dict(response_format)
        if seed is not None:
            payload["seed"] = int(seed)

        try:
            response = self.session.post(
                self.chat_url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except (requests.RequestException, OSError) as exc:
            raise LlamaCppError(
                f"llama.cpp request failed at {self.chat_url}: {exc}"
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise LlamaCppError(
                "llama.cpp returned a non-JSON response"
            ) from exc

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlamaCppError(
                "llama.cpp response is missing choices[0].message.content"
            ) from exc

        if isinstance(content, str):
            result = content.strip()
        elif isinstance(content, list):
            text_parts = [
                str(item.get("text", ""))
                for item in content
                if isinstance(item, Mapping) and item.get("type") == "text"
            ]
            result = "".join(text_parts).strip()
        else:
            result = ""

        if not result:
            raise LlamaCppError("llama.cpp returned an empty completion")
        return result

    def chat_with_image(
        self,
        image_path: Path | str,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 128,
        response_format: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> str:
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        data_url = _image_data_url(path)
        messages: list[dict[str, Any]] = []

        if system_prompt and system_prompt.strip():
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt.strip(),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                    {
                        "type": "text",
                        "text": prompt.strip(),
                    },
                ],
            }
        )

        return self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            seed=seed,
        )


def _image_data_url(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/jpeg"

    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
