from __future__ import annotations

import json
import re
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
    raise VlmCaptionerError(f"unsupported ONNX tensor type: {ort_type}")


def _session_inputs(session: Any) -> dict[str, Any]:
    return {item.name: item for item in session.get_inputs()}


def _session_outputs(session: Any) -> list[str]:
    return [item.name for item in session.get_outputs()]


def _pick_input(inputs: dict[str, Any], *names: str) -> Any | None:
    for name in names:
        if name in inputs:
            return inputs[name]
    return None


def _pick_output(session: Any, *names: str) -> tuple[str, int]:
    outputs = _session_outputs(session)
    for name in names:
        if name in outputs:
            return name, outputs.index(name)
    if not outputs:
        raise VlmCaptionerError("ONNX session has no outputs")
    return outputs[0], 0


def _cast_for_meta(value: np.ndarray, meta: Any) -> np.ndarray:
    return np.asarray(value).astype(_numpy_dtype(meta.type), copy=False)


def _scalar_for_meta(value: int | bool, meta: Any) -> np.ndarray:
    dtype = _numpy_dtype(meta.type)
    if len(meta.shape) == 0:
        return np.asarray(value, dtype=dtype)
    if len(meta.shape) == 1:
        return np.asarray([value], dtype=dtype)
    raise VlmCaptionerError(
        f"unsupported scalar ONNX input shape for {meta.name}: {meta.shape}"
    )


def _resolve_empty_cache_shape(
    metadata_shape: Any,
    *,
    batch_size: int,
    num_kv_heads: int,
) -> tuple[int, ...]:
    resolved: list[int] = []
    for dim in metadata_shape:
        if isinstance(dim, (int, np.integer)):
            resolved.append(int(dim))
            continue

        name = str(dim).lower()
        if name in {"batch_size", "batch"}:
            resolved.append(batch_size)
        elif "batch_size" in name and "num_heads" in name:
            resolved.append(batch_size * num_kv_heads)
        elif "num_heads" in name or name in {"num_kv_heads", "heads"}:
            resolved.append(num_kv_heads)
        else:
            # Empty cache: sequence-like symbolic dimensions start at zero.
            resolved.append(0)

    return tuple(resolved)


class SmolVlmOnnxCaptioner:
    """SmolVLM2 caption runtime using the official ONNX components and ONNX Runtime."""

    _DEFAULT_REPO_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    _ONNX_FILES = (
        "onnx/decoder_model_merged_q4f16.onnx",
        "onnx/embed_tokens_q4f16.onnx",
        "onnx/vision_encoder_q4f16.onnx",
    )
    _PROCESSOR_PATTERNS = (
        "added_tokens.json",
        "chat_template.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    )

    @classmethod
    def _resolve_model_root(cls, model_path: str) -> Path:
        raw_model_path = str(model_path).strip()

        if raw_model_path:
            candidate = Path(raw_model_path).expanduser()
            if candidate.is_dir():
                return candidate.resolve()
        else:
            candidate = None

        repo_id = raw_model_path or cls._DEFAULT_REPO_ID
        if re.fullmatch(r"[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", repo_id):
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise VlmCaptionerError(
                    "ONNX VLM auto-download requires huggingface-hub; "
                    "install lora-data-maker[caption-onnx]"
                ) from exc

            try:
                return Path(
                    snapshot_download(
                        repo_id=repo_id,
                        allow_patterns=[
                            *cls._PROCESSOR_PATTERNS,
                            *cls._ONNX_FILES,
                        ],
                    )
                ).resolve()
            except Exception as exc:
                raise VlmCaptionerError(
                    f"could not download SmolVLM2 model {repo_id!r}: {exc}"
                ) from exc

        display_path = (
            str(candidate.resolve()) if candidate is not None else raw_model_path
        )
        raise VlmCaptionerError(
            f"VLM ONNX model directory not found: {display_path}; "
            "use a local directory or a Hugging Face repo ID such as "
            f"{cls._DEFAULT_REPO_ID}"
        )

    def __init__(self, config: VlmConfig) -> None:
        if config.runtime != "onnx":
            raise VlmCaptionerError(
                "SmolVlmOnnxCaptioner requires caption.vlm.runtime = 'onnx'"
            )

        root = self._resolve_model_root(config.model_path)
        onnx_dir = root / "onnx"

        decoder_path = onnx_dir / "decoder_model_merged_q4f16.onnx"
        embed_path = onnx_dir / "embed_tokens_q4f16.onnx"
        vision_path = onnx_dir / "vision_encoder_q4f16.onnx"

        missing = [p for p in (decoder_path, embed_path, vision_path) if not p.is_file()]
        if missing:
            names = ", ".join(str(p) for p in missing)
            raise VlmCaptionerError(f"missing SmolVLM2 ONNX files: {names}")

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

        try:
            providers = ["CPUExecutionProvider"]
            self._vision = ort.InferenceSession(
                str(vision_path),
                sess_options=options,
                providers=providers,
            )
            self._embed = ort.InferenceSession(
                str(embed_path),
                sess_options=options,
                providers=providers,
            )
            self._decoder = ort.InferenceSession(
                str(decoder_path),
                sess_options=options,
                providers=providers,
            )
            self._processor = AutoProcessor.from_pretrained(
                str(root),
                local_files_only=True,
            )
        except Exception as exc:
            raise VlmCaptionerError(
                f"could not initialize SmolVLM2 ONNX runtime: {exc}"
            ) from exc

        self.root = root
        self.config = config
        self._prompt = config.prompt
        self._max_new_tokens = config.max_tokens

        self._vision_inputs = _session_inputs(self._vision)
        self._decoder_inputs = _session_inputs(self._decoder)
        self._decoder_outputs = _session_outputs(self._decoder)

        self._embed_input_name = self._embed.get_inputs()[0].name
        self._embed_output_name, _ = _pick_output(
            self._embed,
            "inputs_embeds",
        )

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

        if "inputs_embeds" not in self._decoder_inputs:
            raise VlmCaptionerError(
                "SmolVLM2 decoder ONNX does not expose inputs_embeds"
            )

    def _processor_prompt(self, image: Image.Image) -> dict[str, np.ndarray]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self._prompt},
                ],
            }
        ]
        try:
            inputs = self._processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="np",
            )
        except Exception as exc:
            raise VlmCaptionerError(
                f"SmolVLM2 processor failed to prepare multimodal inputs: {exc}"
            ) from exc

        return {
            str(key): np.asarray(value)
            for key, value in inputs.items()
            if isinstance(value, np.ndarray)
        }

    def _run_vision(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        pixel_values = inputs.get("pixel_values")
        if pixel_values is None:
            raise VlmCaptionerError("SmolVLM2 processor returned no pixel_values")

        pixel_input = _pick_input(self._vision_inputs, "pixel_values")
        if pixel_input is None:
            raise VlmCaptionerError(
                "SmolVLM2 vision ONNX has no pixel_values input"
            )

        pixels = np.asarray(pixel_values)
        expected_rank = len(pixel_input.shape)
        pixels_were_flattened = expected_rank == 4 and pixels.ndim == 5
        pixels_were_batched = expected_rank == 5 and pixels.ndim == 4
        if pixels_were_flattened:
            pixels = pixels.reshape(
                pixels.shape[0] * pixels.shape[1],
                *pixels.shape[2:],
            )
        elif pixels_were_batched:
            pixels = pixels[:, None, ...]

        feed: dict[str, np.ndarray] = {
            pixel_input.name: _cast_for_meta(pixels, pixel_input),
        }

        attention_input = _pick_input(
            self._vision_inputs,
            "pixel_attention_mask",
            "patch_attention_mask",
        )
        if attention_input is not None:
            mask = inputs.get("pixel_attention_mask")

            if mask is None:
                if expected_rank == 5:
                    mask = np.ones(
                        (
                            pixels.shape[0],
                            pixels.shape[1],
                            pixels.shape[-2],
                            pixels.shape[-1],
                        ),
                        dtype=np.bool_,
                    )
                else:
                    mask = np.ones(
                        (pixels.shape[0], pixels.shape[-2], pixels.shape[-1]),
                        dtype=np.bool_,
                    )
            else:
                mask = np.asarray(mask)
                if pixels_were_flattened and mask.ndim == 4:
                    mask = mask.reshape(
                        mask.shape[0] * mask.shape[1],
                        *mask.shape[2:],
                    )
                elif pixels_were_batched and mask.ndim == 3:
                    mask = mask[:, None, ...]

            if attention_input.name == "patch_attention_mask":
                if mask.ndim == 4:
                    mask = mask.reshape(
                        mask.shape[0] * mask.shape[1],
                        *mask.shape[2:],
                    )
                height, width = pixels.shape[-2:]
                patch_h = height // 16
                patch_w = width // 16
                if mask.shape[-2:] != (patch_h, patch_w):
                    mask = mask[..., : patch_h * 16, : patch_w * 16]
                    mask = mask.reshape(
                        mask.shape[0],
                        patch_h,
                        16,
                        patch_w,
                        16,
                    ).any(axis=(2, 4))

            feed[attention_input.name] = _cast_for_meta(mask, attention_input)

        output_name, output_index = _pick_output(self._vision, "image_features")
        try:
            outputs = self._vision.run([output_name], feed)
        except Exception as exc:
            raise VlmCaptionerError(
                f"SmolVLM2 vision inference failed: {exc}"
            ) from exc

        features = np.asarray(outputs[0])
        if features.ndim < 2:
            raise VlmCaptionerError(
                f"unexpected SmolVLM2 vision output shape: {features.shape}"
            )
        return features

    def _embed_tokens(self, token_ids: np.ndarray) -> np.ndarray:
        input_meta = self._embed.get_inputs()[0]
        inputs = {input_meta.name: _cast_for_meta(token_ids, input_meta)}
        try:
            outputs = self._embed.run(
                [self._embed_output_name],
                inputs,
            )
        except Exception as exc:
            raise VlmCaptionerError(
                f"SmolVLM2 token embedding inference failed: {exc}"
            ) from exc
        return np.asarray(outputs[0])

    def _merge_image_features(
        self,
        input_ids: np.ndarray,
        token_embeddings: np.ndarray,
        image_features: np.ndarray,
    ) -> np.ndarray:
        image_mask = np.asarray(input_ids) == self._image_token_id
        token_count = int(image_mask.sum())
        if token_count == 0:
            raise VlmCaptionerError(
                "SmolVLM2 processor produced no image tokens"
            )

        hidden_size = int(image_features.shape[-1])
        flat_features = np.asarray(image_features).reshape(-1, hidden_size)
        if token_count != flat_features.shape[0]:
            raise VlmCaptionerError(
                "SmolVLM2 image/token mismatch: "
                f"tokens={token_count}, features={flat_features.shape[0]}"
            )

        merged = np.asarray(token_embeddings).copy()
        if merged.ndim != 3:
            raise VlmCaptionerError(
                f"unexpected token embedding shape: {merged.shape}"
            )

        if merged.shape[-1] != hidden_size:
            raise VlmCaptionerError(
                "SmolVLM2 hidden-size mismatch: "
                f"text={merged.shape[-1]}, image={hidden_size}"
            )

        merged[image_mask] = flat_features
        decoder_meta = self._decoder_inputs["inputs_embeds"]
        return _cast_for_meta(merged, decoder_meta)

    def _empty_past(self, batch_size: int) -> dict[str, np.ndarray]:
        past: dict[str, np.ndarray] = {}
        for meta in self._decoder.get_inputs():
            if not meta.name.startswith("past_key_values."):
                continue
            shape = _resolve_empty_cache_shape(
                meta.shape,
                batch_size=batch_size,
                num_kv_heads=self._num_kv_heads,
            )
            dtype = _numpy_dtype(meta.type)
            past[meta.name] = np.zeros(shape, dtype=dtype)
        if not past:
            raise VlmCaptionerError(
                "SmolVLM2 decoder ONNX exposes no past_key_values inputs"
            )
        return past

    def _next_past(self, outputs: list[np.ndarray]) -> dict[str, np.ndarray]:
        output_map = {
            name: np.asarray(value)
            for name, value in zip(self._decoder_outputs, outputs, strict=True)
        }
        past: dict[str, np.ndarray] = {}
        for input_name in self._decoder_inputs:
            if not input_name.startswith("past_key_values."):
                continue
            present_name = input_name.replace("past_key_values.", "present.", 1)
            value = output_map.get(present_name)
            if value is None:
                raise VlmCaptionerError(
                    f"SmolVLM2 decoder output missing {present_name}"
                )
            past[input_name] = value

        if not past:
            raise VlmCaptionerError(
                "SmolVLM2 decoder produced no present key/value cache"
            )
        return past

    def _position_ids(
        self,
        attention_mask: np.ndarray,
        *,
        last_only: bool = False,
    ) -> np.ndarray:
        mask = np.asarray(attention_mask).astype(np.int64, copy=False)
        positions = mask.cumsum(axis=-1) - 1
        positions = np.where(mask == 0, 1, positions)
        if last_only:
            positions = positions[:, -1:]
        return positions

    def _decoder_feed(
        self,
        *,
        inputs_embeds: np.ndarray,
        attention_mask: np.ndarray,
        past: dict[str, np.ndarray],
        use_cache: bool,
    ) -> dict[str, np.ndarray]:
        feed: dict[str, np.ndarray] = {
            self._decoder_inputs["inputs_embeds"].name: _cast_for_meta(
                inputs_embeds,
                self._decoder_inputs["inputs_embeds"],
            )
        }

        attention_meta = self._decoder_inputs.get("attention_mask")
        if attention_meta is not None:
            feed[attention_meta.name] = _cast_for_meta(
                attention_mask,
                attention_meta,
            )

        position_meta = self._decoder_inputs.get("position_ids")
        if position_meta is not None:
            positions = self._position_ids(
                attention_mask,
                last_only=use_cache,
            )
            feed[position_meta.name] = _cast_for_meta(positions, position_meta)

        cache_branch_meta = self._decoder_inputs.get("use_cache_branch")
        if cache_branch_meta is not None:
            feed[cache_branch_meta.name] = _scalar_for_meta(
                use_cache,
                cache_branch_meta,
            )

        logits_keep_meta = self._decoder_inputs.get("num_logits_to_keep")
        if logits_keep_meta is not None:
            feed[logits_keep_meta.name] = _scalar_for_meta(
                1,
                logits_keep_meta,
            )

        feed.update(past)
        return feed

    def _decoder_logits(self, outputs: list[np.ndarray]) -> np.ndarray:
        output_name, output_index = _pick_output(self._decoder, "logits")
        if output_index >= len(outputs):
            raise VlmCaptionerError(
                f"SmolVLM2 decoder returned {len(outputs)} outputs, "
                f"cannot read {output_name}"
            )
        return np.asarray(outputs[output_index])

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
            with Image.open(path) as source:
                image = source.convert("RGB")
        except OSError as exc:
            raise VlmCaptionerError(
                f"could not open image {path}: {exc}"
            ) from exc

        try:
            inputs = self._processor_prompt(image)
            input_ids = inputs.get("input_ids")
            attention_mask = inputs.get("attention_mask")
            if input_ids is None:
                raise VlmCaptionerError(
                    "SmolVLM2 processor returned no input_ids"
                )

            token_ids = np.asarray(input_ids, dtype=np.int64)
            if token_ids.ndim != 2:
                raise VlmCaptionerError(
                    f"unexpected SmolVLM2 input_ids shape: {token_ids.shape}"
                )

            if attention_mask is None:
                attention_mask = np.ones(
                    token_ids.shape,
                    dtype=np.int64,
                )
            else:
                attention_mask = np.asarray(attention_mask)

            image_features = self._run_vision(inputs)
            token_embeddings = self._embed_tokens(token_ids)
            merged_embeddings = self._merge_image_features(
                token_ids,
                token_embeddings,
                image_features,
            )

            batch_size = int(token_ids.shape[0])
            past = self._empty_past(batch_size)
            generated: list[int] = []

            for step in range(self._max_new_tokens):
                if step == 0:
                    step_embeddings = merged_embeddings
                    step_attention = attention_mask
                else:
                    generated_array = np.asarray([[generated[-1]]], dtype=np.int64)
                    step_embeddings = self._embed_tokens(generated_array)
                    step_attention = np.ones(
                        (batch_size, token_ids.shape[1] + len(generated)),
                        dtype=np.int64,
                    )

                feed = self._decoder_feed(
                    inputs_embeds=step_embeddings,
                    attention_mask=step_attention,
                    past=past,
                    use_cache=step > 0,
                )

                try:
                    outputs = self._decoder.run(self._decoder_outputs, feed)
                except Exception as exc:
                    raise VlmCaptionerError(
                        f"SmolVLM2 decoder inference failed at token {step}: {exc}"
                    ) from exc

                logits = self._decoder_logits(outputs)
                if logits.ndim != 3 or logits.shape[0] != batch_size:
                    raise VlmCaptionerError(
                        f"unexpected SmolVLM2 decoder logits shape: {logits.shape}"
                    )

                next_token = int(np.argmax(logits[:, -1, :], axis=-1)[0])
                generated.append(next_token)
                past = self._next_past(outputs)

                if next_token == self._eos_token_id:
                    break

            text = self._decode_tokens(generated)
            if not text:
                raise VlmCaptionerError(
                    "SmolVLM2 ONNX returned an empty caption; "
                    f"generated_tokens={len(generated)}; "
                    f"eos_token_id={self._eos_token_id}"
                )
            return text
        finally:
            try:
                image.close()
            except Exception:
                pass

    def close(self) -> None:
        for name in ("_vision", "_embed", "_decoder"):
            session = getattr(self, name, None)
            if session is None:
                continue
            try:
                release = getattr(session, "release", None)
                if callable(release):
                    release()
            except Exception:
                pass
            setattr(self, name, None)
