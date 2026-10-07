from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Sequence


class LlamaCppError(RuntimeError):
    """Raised when the embedded llama.cpp runtime cannot infer."""


class LlamaCppRuntime:
    """Embedded llama.cpp runtime backed by llama-cpp-python."""

    def __init__(
        self,
        *,
        model_path: Path | str,
        mmproj_path: Path | str | None = None,
        n_ctx: int = 4096,
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
        seed: int | None = None,
        temperature: float = 0.2,
        max_tokens: int = 128,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.mmproj_path = (
            Path(mmproj_path).expanduser().resolve()
            if mmproj_path
            else None
        )
        if not self.model_path.is_file():
            raise LlamaCppError(f"llama.cpp model not found: {self.model_path}")
        if self.mmproj_path is not None and not self.mmproj_path.is_file():
            raise LlamaCppError(
                f"llama.cpp mmproj not found: {self.mmproj_path}"
            )
        if n_ctx <= 0:
            raise ValueError("n_ctx must be greater than zero")
        if n_threads is not None and n_threads <= 0:
            raise ValueError("n_threads must be greater than zero")
        if n_gpu_layers < -1:
            raise ValueError("n_gpu_layers must be -1 or greater")
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        self.n_ctx = int(n_ctx)
        self.n_threads = n_threads
        self.n_gpu_layers = int(n_gpu_layers)
        self.verbose = bool(verbose)
        self.seed = seed
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

        self._llm = None
        self._lock = RLock()

    @property
    def loaded(self) -> bool:
        return self._llm is not None

    def load(self) -> "LlamaCppRuntime":
        if self._llm is not None:
            return self

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise LlamaCppError(
                "embedded llama.cpp runtime requires llama-cpp-python; "
                "install the optional llama_cpp dependency"
            ) from exc

        chat_handler = None
        if self.mmproj_path is not None:
            try:
                from llama_cpp.llama_chat_format import MTMDChatHandler

                chat_handler = MTMDChatHandler(
                    clip_model_path=str(self.mmproj_path),
                    verbose=self.verbose,
                    use_gpu=self.n_gpu_layers != 0,
                )
            except ImportError as exc:
                raise LlamaCppError(
                    "this llama-cpp-python build does not provide MTMD "
                    "multimodal support"
                ) from exc

        kwargs: dict[str, Any] = {
            "model_path": str(self.model_path),
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "verbose": self.verbose,
        }
        if self.n_threads is not None:
            kwargs["n_threads"] = self.n_threads
        if chat_handler is not None:
            kwargs["chat_handler"] = chat_handler

        try:
            self._llm = Llama(**kwargs)
        except Exception as exc:
            self._llm = None
            raise LlamaCppError(
                f"failed to load embedded llama.cpp model: {exc}"
            ) from exc

        return self

    def close(self) -> None:
        with self._lock:
            llm = self._llm
            self._llm = None
            if llm is not None:
                close = getattr(llm, "close", None)
                if callable(close):
                    close()

    def __enter__(self) -> "LlamaCppRuntime":
        return self.load()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> str:
        with self._lock:
            llm = self.load()._llm
            try:
                response = llm.create_chat_completion(
                    messages=list(messages),
                    temperature=(
                        self.temperature
                        if temperature is None
                        else float(temperature)
                    ),
                    max_tokens=(
                        self.max_tokens
                        if max_tokens is None
                        else int(max_tokens)
                    ),
                    response_format=response_format,
                    seed=self.seed if seed is None else seed,
                )
            except Exception as exc:
                raise LlamaCppError(
                    f"embedded llama.cpp inference failed: {exc}"
                ) from exc

        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LlamaCppError(
                "llama.cpp response is missing choices[0].message.content"
            ) from exc

        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            text = "".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, Mapping) and item.get("type") == "text"
            ).strip()
        else:
            text = ""

        if not text:
            raise LlamaCppError("llama.cpp returned an empty completion")
        return text

    def chat_with_image(
        self,
        image_path: Path | str,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: Mapping[str, Any] | None = None,
        seed: int | None = None,
    ) -> str:
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        import base64
        import mimetypes

        mime_type, _ = mimetypes.guess_type(path.name)
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/jpeg"

        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        data_url = f"data:{mime_type};base64,{encoded}"

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
                    {"type": "text", "text": prompt.strip()},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
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
