from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .caption_vlm import VlmCaptionerError, VlmConfig


def _numpy_dtype(ort_type: str) -> np.dtype:
    value = str(ort_type).lower()
    if "float16" in value:
        return np.dtype(np.float16)
    if "float" in value:
        return np.dtype(np.float32)
    if "int64" in value:
        return np.dtype(np.int64)
    if "int32" in value:
        return np.dtype(np.int32)
    if "bool" in value:
        return np.dtype(np.bool_)
    raise VlmCaptionerError(f"unsupported ONNX input type: {ort_type}")


def _session_inputs(session: Any) -> dict[str, Any]:
    return {item.name: item for item in session.get_inputs()}


def _session_outputs(session: Any) -> list[str]:
    return [item.name for item in session.get_outputs()]


def _pick_input(inputs: dict[str, Any], *names: str) -> Any | None:
    for name in names:
        if name in inputs:
            return inputs[name]
    return None


class SmolVlmOnnxCaptioner:
    """SmolVLM2 caption runtime using the official ONNX components and ONNX Runtime."""

    def __init__(self, config: VlmConfig) -> None:
        if not config.model_path.strip():
            raise VlmCaptionerError("VLM ONNX model directory is required")

        root = Path(config.model_path).expanduser().resolve()
        if not root.is_dir():
            raise VlmCaptionerError(f"VLM ONNX model directory not found: {root}")

        onnx_dir = root / "onnx"
        decoder_path = onnx_dir / "decoder_model_merged_q4f16.onnx"
        embed_path = onnx_dir / "embed_tokens_q4f16.onnx"
        vision_path = onnx_dir / "vision_encoder_q4f16.onnx"

        missing = [p for p in (decoder_path, embed_path, vision_path) if not p.is_file()]
        if missing:
            names = ", ".join(str(p) for p in missing)
            raise VlmCaptionerError(
                f"missing SmolVLM2 ONNX files: {names}"
            )

        try:
            import onnxruntime as ort
            from transformers import AutoProcessor
        except ImportError as exc:
            raise VlmCaptionerError(
                "ONNX VLM captions require onnxruntime and transformers; "
                "install lora-data-maker[caption-onnx]"
            ) from exc

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = config.n_threads
        options.inter_op_num_threads = 1

        providers = ["CPUExecutionProvider"]
        try:
            self._vision = ort.InferenceSession(
                str(vision_path), sess_options=options, providers=providers
            )
            self._embed = ort.InferenceSession(
                str(embed_path), sess_options=options, providers=providers
            )
            self._decoder = ort.InferenceSession(
                str(decoder_path), sess_options=options, providers=providers
            )
            self._processor = AutoProcessor.from_pretrained(
                str(root), local_files_only=True
            )
        except Exception as exc:
            raise VlmCaptionerError(
                f"could not initialize SmolVLM2 ONNX runtime: {exc}"
            ) from exc

        self.root = root
        self._prompt = config.prompt
        self._max_new_tokens = config.max_tokens
        self._decoder_inputs = _session_inputs(self._decoder)
        self._decoder_outputs = _session_outputs(self._decoder)
        self._embed_input_name = self._embed.get_inputs()[0].name
        self._embed_output_name = self._embed.get_outputs()[0].name

        config_path = root / "config.json"
        generation_path = root / "generation_config.json"
        try:
            model_config = json.loads(config_path.read_text(encoding="utf-8"))
            generation_config = (
                json.loads(generation_path.read_text(encoding="utf-8"))
                if generation_path.is_file()
                else {}
            )
        except (OSError, ValueError) as exc:
            raise VlmCaptionerError(
                f"could not read SmolVLM2 config files: {exc}"
            ) from exc

        text_config = model_config.get("text_config", {})
        self._num_layers = int(text_config.get("num_hidden_layers", 32))
        self._num_kv_heads = int(text_config.get("num_key_value_heads", 5))
        self._head_dim = int(text_config.get("head_dim", 64))
        self._image_token_id = int(model_config.get("image_token_id", 49190))
        self._eos_token_id = int(
            generation_config.get(
                "eos_token_id",
                getattr(self._processor.tokenizer, "eos_token_id", 49279),
            )
            or 49279
        )

    def _processor_prompt(self, image: Image.Image) -> dict[str, np.ndarray]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self._prompt},
                ],
            }
        ]
        prompt = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        inputs = self._processor(
            images=image,
            text=prompt,
            return_tensors="np",
        )
        return {
            str(key): np.asarray(value)
            for key, value in inputs.items()
            if isinstance(value, np.ndarray)
        }

    def _run_vision(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        names = _session_inputs(self._vision)
        feed: dict[str, np.ndarray] = {}

        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            raise VlmCaptionerError("SmolVLM2 processor returned no pixel_values")

        pixel_input = _pick_input(names, "pixel_values")
        if pixel_input is None:
            raise VlmCaptionerError("SmolVLM2 vision ONNX has no pixel_values input")

        pixels = pixel_values
        if len(pixel_input.shape) == 4 and pixels.ndim == 5:
            pixels = pixels.reshape(
                pixels.shape[0] * pixels.shape[1], *pixels.shape[2:]
            )
        pixels = pixels.astype(_numpy_dtype(pixel_input.type), copy=False)
        feed[pixel_input.name] = pixels

        attention_input = _pick_input(
            names,
            "pixel_attention_mask",
            "patch_attention_mask",
        )
        if attention_input is not None:
            mask = inputs.get("pixel_attention_mask")
            if mask is None:
                height, width = pixels.shape[-2:]
                patch = 16
                mask = np.ones(
                    (pixels.shape[0], height // patch, width // patch),
                    dtype=np.bool_,
                )
            elif mask.ndim == 4:
                mask = mask.reshape(mask.shape[0] * mask.shape[1], *mask.shape[2:])
            elif mask.ndim == 3 and mask.shape[0] != pixels.shape[0]:
                mask = np.broadcast_to(mask, (pixels.shape[0], *mask.shape[1:]))
            if attention_input.name == "patch_attention_mask" and mask.ndim == 3:
                mask = mask.astype(bool, copy=False)
                ph = pixels.shape[-2] // 16
                pw = pixels.shape[-1] // 16
                mask = mask[:, :ph * 16, :pw * 16]
                mask = mask.reshape(mask.shape[0], ph, 16, pw, 16).sum(
                    axis=(2, 4)
                ) > 0
            feed[attention_input.name] = mask.astype(
                _numpy_dtype(attention_input.type), copy=False
            )

        output = self._vision.run(_session_outputs(self._vision), feed)[0]
        features = np.asarray(output)
        if features.ndim == 2:
            features = features[None, ...]
        if features.ndim != 3:
            raise VlmCaptionerError(
                f"unexpected SmolVLM2 vision output shape: {features.shape}"
            )
        return features

    def _embed_tokens(self, token_ids: np.ndarray) -> np.ndarray:
        output = self._embed.run(
            [self._embed_output_name],
            {self._embed_input_name: token_ids.astype(np.int64, copy=False)},
        )[0]
        return np.asarray(output)

    def _merge_image_features(
        self,
        input_ids: np.ndarray,
        token_embeddings: np.ndarray,
        image_features: np.ndarray,
    ) -> np.ndarray:
        positions = np.argwhere(input_ids == self._image_token_id)
        token_count = int(positions.shape[0])
        flat_features = image_features.reshape(-1, image_features.shape[-1])
        if token_count != flat_features.shape[0]:
            raise VlmCaptionerError(
                "SmolVLM2 image/token mismatch: "
                f"tokens={token_count}, features={flat_features.shape[0]}"
            )
        merged = np.asarray(token_embeddings).copy()
        for index, (batch, position) in enumerate(positions):
            merged[batch, position] = flat_features[index]
        return merged.astype(
            np.float16 if "float16" in str(self._decoder_inputs["inputs_embeds"].type) else np.float32,
            copy=False,
        )

    def _empty_past(self, batch_size: int) -> dict[str, np.ndarray]:
        past: dict[str, np.ndarray] = {}
        for item in self._decoder.get_inputs():
            if not item.name.startswith("past_key_values."):
                continue
            dtype = _numpy_dtype(item.type)
            past[item.name] = np.zeros(
                (batch_size, self._num_kv_heads, 0, self._head_dim),
                dtype=dtype,
            )
        return past

    def _next_past(
        self,
        outputs: list[np.ndarray],
    ) -> dict[str, np.ndarray]:
        past: dict[str, np.ndarray] = {}
        for name, value in zip(self._decoder_outputs[1:], outputs[1:]):
            if name.startswith("present."):
                past[name.replace("present.", "past_key_values.", 1)] = np.asarray(value)
        return past

    def _decode_tokens(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        return self._processor.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        ).strip()

    def caption(self, image_path: Path | str) -> str:
        path = Path(image_path).expanduser().resolve()
        try:
            image = Image.open(path).convert("RGB")
        except OSError as exc:
            raise VlmCaptionerError(
                f"could not open image {path}: {exc}"
            ) from exc

        self._prompt = self.config_prompt
