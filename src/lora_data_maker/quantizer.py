from __future__ import annotations

import json
import os
import queue
import shutil
import struct
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable

SUPPORTED_QUANT_TYPES = ("q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "f16", "f32")
COMPONENT_NAMES = ("unet", "clip", "vae", "other")
ARCHITECTURE_LABELS = {
    "sd15": "SD 1.x / SD 1.5-like",
    "sd2": "SD 2.x",
    "sdxl": "SDXL (including Lightning)",
    "sd3": "SD3 / SD3.5",
    "flux": "FLUX",
    "unknown": "Unknown / custom architecture",
}
SUPPORTED_ARCHITECTURES = frozenset({"sd15", "sd2", "sdxl", "sd3", "flux"})
_COPY_CHUNK = 4 * 1024 * 1024
_MAX_HEADER_BYTES = 256 * 1024 * 1024


class QuantizerError(RuntimeError):
    """A stable-diffusion.cpp or safetensors quantization error."""


class QuantizerCancelledError(QuantizerError):
    """Raised when a quantization process is cancelled."""


def find_sd_cli() -> str | None:
    """Find the stable-diffusion.cpp CLI without executing any generation mode."""
    configured = os.environ.get("LORA_DATA_MAKER_SD_CPP_CLI", "").strip()
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
        resolved = shutil.which(configured)
        return str(Path(resolved).resolve()) if resolved else None

    for name in ("sd-cli", "sd"):
        resolved = shutil.which(name)
        if resolved:
            return str(Path(resolved).resolve())
    return None


def require_sd_cli() -> str:
    cli = find_sd_cli()
    if cli:
        return cli
    raise QuantizerError(
        "stable-diffusion.cpp CLI was not found. Build or install sd.cpp, then set "
        "LORA_DATA_MAKER_SD_CPP_CLI=/absolute/path/to/sd-cli before starting the WebUI."
    )


def _read_header(path: Path) -> tuple[dict[str, Any], int, int]:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise QuantizerError("file is too small to be a safetensors file")
            header_size = struct.unpack("<Q", prefix)[0]
            if header_size < 2 or header_size > _MAX_HEADER_BYTES:
                raise QuantizerError(
                    f"invalid safetensors header size: {header_size} bytes"
                )
            data_start = 8 + header_size
            if data_start > size:
                raise QuantizerError("safetensors header extends beyond the file")
            raw_header = handle.read(header_size)
            if len(raw_header) != header_size:
                raise QuantizerError("safetensors header is truncated")
        header = json.loads(raw_header.decode("utf-8"))
    except QuantizerError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, struct.error) as exc:
        raise QuantizerError(f"could not read safetensors header: {exc}") from exc

    if not isinstance(header, dict):
        raise QuantizerError("safetensors header must be a JSON object")
    return header, data_start, size - data_start


def _tensor_entries(
    header: dict[str, Any], payload_size: int
) -> list[tuple[str, dict[str, Any], int, int]]:
    entries: list[tuple[str, dict[str, Any], int, int]] = []
    for name, raw in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(raw, dict):
            raise QuantizerError("safetensors header contains an invalid tensor entry")
        offsets = raw.get("data_offsets")
        shape = raw.get("shape")
        dtype = raw.get("dtype")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(v, bool) or not isinstance(v, int) for v in offsets)
            or not isinstance(shape, list)
            or not isinstance(dtype, str)
        ):
            raise QuantizerError(f"invalid tensor metadata for {name!r}")
        start, end = offsets
        if start < 0 or end < start or end > payload_size:
            raise QuantizerError(f"invalid data offsets for tensor {name!r}")
        entries.append((name, raw, start, end))
    if not entries:
        raise QuantizerError("safetensors file contains no tensors")
    return entries


def _has_component_prefix(key: str, prefixes: tuple[str, ...]) -> bool:
    """Match both root-level Diffusers keys and keys nested under a model namespace."""
    return any(key.startswith(prefix) or ("." + prefix) in key for prefix in prefixes)


def _component_for_tensor(name: str) -> str:
    key = name.replace("\\", "/").lower()

    if key.startswith(("model.diffusion_model.", "diffusion_model.", "unet.", "unet/")):
        return "unet"
    if key.startswith(("first_stage_model.", "vae.", "vae_ema.", "autoencoder.")):
        return "vae"
    if key.startswith(
        (
            "cond_stage_model.",
            "conditioner.embedders.",
            "text_encoder.",
            "text_encoder_2.",
            "text_encoders.",
            "clip_l.",
            "clip_g.",
            "text_model.",
            "token_embedding.",
            "transformer.text_model.",
        )
    ):
        return "clip"
    if key.startswith(("encoder.", "decoder.", "quant_conv.", "post_quant_conv.")):
        return "vae"

    # Root-level UNet and DiT/diffusion-transformer namespaces. SD3 and FLUX
    # use transformer backbones, so they must be included alongside classic UNets.
    if _has_component_prefix(
        key,
        (
            "down_blocks.",
            "up_blocks.",
            "mid_block.",
            "conv_in.",
            "conv_out.",
            "time_embedding.",
            "add_embedding.",
            "input_blocks.",
            "output_blocks.",
            "middle_block.",
            "double_blocks.",
            "single_blocks.",
            "single_transformer_blocks.",
            "joint_blocks.",
            "transformer_blocks.",
            "x_embedder.",
            "context_embedder.",
            "y_embedder.",
            "img_in.",
            "txt_in.",
            "vector_in.",
            "guidance_in.",
            "final_layer.",
        ),
    ):
        return "unet"
    return "other"


def _detect_architecture(
    entries: list[tuple[str, dict[str, Any], int, int]],
) -> str:
    """Best-effort architecture classification using tensor names and shapes only."""
    keys = [name.replace("\\", "/").lower() for name, _meta, _start, _end in entries]

    def has_any_token(tokens: tuple[str, ...]) -> bool:
        return any(
            key.startswith(token) or ("." + token) in key
            for key in keys
            for token in tokens
        )

    # Test FLUX first: some of its transformer naming overlaps with SD3.
    if has_any_token(
        (
            "double_blocks.",
            "single_blocks.",
            "single_transformer_blocks.",
            "img_in.",
            "txt_in.",
            "vector_in.",
            "guidance_in.",
        )
    ):
        return "flux"

    if has_any_token(("joint_blocks.", "y_embedder.")):
        return "sd3"

    if has_any_token(
        (
            "conditioner.embedders.1.",
            "text_encoder_2.",
            "text_encoders.clip_g.",
            "label_emb.0.0.",
            "add_embedding.linear_1.",
        )
    ):
        return "sdxl"

    cross_attention_dims: set[int] = set()
    for name, meta, _start, _end in entries:
        key = name.lower()
        if "attn2.to_k.weight" not in key and "attn2.to_v.weight" not in key:
            continue
        shape = meta.get("shape")
        if isinstance(shape, list) and len(shape) >= 2 and isinstance(shape[-1], int):
            cross_attention_dims.add(shape[-1])

    if 2048 in cross_attention_dims:
        return "sdxl"
    if 1024 in cross_attention_dims:
        return "sd2"
    if 768 in cross_attention_dims:
        return "sd15"

    # A generic transformer-block namespace with an SD3-like context embedder is
    # most likely an SD3-family checkpoint if no FLUX-specific signature appeared.
    if has_any_token(("context_embedder.",)) and has_any_token(("transformer_blocks.",)):
        return "sd3"

    return "unknown"


def inspect_safetensors(path: Path | str) -> dict[str, Any]:
    """Inspect names and byte offsets only; tensor weights are never loaded into RAM."""
    target = Path(path).expanduser().resolve()
    header, _data_start, payload_size = _read_header(target)
    entries = _tensor_entries(header, payload_size)
    components = {
        name: {"tensor_count": 0, "bytes": 0}
        for name in COMPONENT_NAMES
    }
    for name, _meta, start, end in entries:
        component = _component_for_tensor(name)
        components[component]["tensor_count"] += 1
        components[component]["bytes"] += end - start

    architecture = _detect_architecture(entries)
    has_model_diffusion_namespace = any(
        name.lower().startswith("model.diffusion_model.")
        for name, _meta, _start, _end in entries
    )
    is_standalone_backbone = (
        not has_model_diffusion_namespace
        and components["unet"]["tensor_count"] > 0
        and components["clip"]["tensor_count"] == 0
        and components["vae"]["tensor_count"] == 0
        and components["other"]["tensor_count"] == 0
    )
    return {
        "size_bytes": target.stat().st_size,
        "tensor_count": len(entries),
        "components": components,
        "can_quantize_unet": components["unet"]["tensor_count"] > 0,
        "is_standalone_backbone": is_standalone_backbone,
        "architecture": architecture,
        "architecture_label": ARCHITECTURE_LABELS[architecture],
        "architecture_recognized": architecture in SUPPORTED_ARCHITECTURES,
        "architecture_note": (
            "Architecture detection is a best-effort estimate from tensor names and shapes; "
            "stable-diffusion.cpp remains the final compatibility check."
            if architecture != "unknown"
            else "Could not confidently identify this architecture from tensor names/shapes. "
                 "stable-diffusion.cpp may still support it, but conversion can fail."
        ),
    }


def _normalized_unet_key(name: str) -> str:
    # --diffusion-model adds the model.diffusion_model. namespace itself. Strip
    # that prefix here to avoid double-prefixing extracted checkpoint tensors.
    for prefix in ("model.diffusion_model.", "diffusion_model.", "unet.", "transformer."):
        if name.startswith(prefix):
            return name[len(prefix):]
    for prefix in ("model.diffusion_model/", "diffusion_model/", "unet/"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def extract_component_safetensors(
    source: Path | str,
    destination: Path | str,
    component: str = "unet",
) -> Path:
    """Stream one selected component into a new safetensors file."""
    if component not in {"unet", "clip", "vae"}:
        raise QuantizerError(f"unsupported component: {component!r}")

    source_path = Path(source).expanduser().resolve()
    target_path = Path(destination).expanduser().resolve()
    header, data_start, payload_size = _read_header(source_path)
    entries = _tensor_entries(header, payload_size)
    selected = [
        (name, meta, start, end)
        for name, meta, start, end in entries
        if _component_for_tensor(name) == component
    ]
    if not selected:
        label = {"unet": "UNet", "clip": "CLIP", "vae": "VAE"}[component]
        raise QuantizerError(
            f"no {label} tensors were detected in the uploaded safetensors file"
        )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_header: dict[str, Any] = {}
    metadata = header.get("__metadata__")
    if isinstance(metadata, dict):
        target_header["__metadata__"] = metadata

    copied: list[tuple[str, str, dict[str, Any], int, int]] = []
    cursor = 0
    seen_names: set[str] = set()
    for name, meta, start, end in sorted(selected, key=lambda item: (item[2], item[3], item[0])):
        out_name = _normalized_unet_key(name) if component == "unet" else name
        if out_name in seen_names:
            raise QuantizerError(
                f"component extraction produced a duplicate tensor name: {out_name!r}"
            )
        seen_names.add(out_name)
        copied.append((name, out_name, meta, start, end))
        new_meta = dict(meta)
        new_meta["data_offsets"] = [cursor, cursor + (end - start)]
        target_header[out_name] = new_meta
        cursor += end - start

    encoded_header = json.dumps(
        target_header,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    encoded_header += b" " * ((-len(encoded_header)) % 8)

    temp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{target_path.name}.",
            suffix=".part",
            dir=target_path.parent,
            delete=False,
        ) as out:
            temp_name = out.name
            out.write(struct.pack("<Q", len(encoded_header)))
            out.write(encoded_header)
            with source_path.open("rb") as inp:
                for _old_name, _new_name, _meta, start, end in copied:
                    inp.seek(data_start + start)
                    remaining = end - start
                    while remaining:
                        chunk = inp.read(min(_COPY_CHUNK, remaining))
                        if not chunk:
                            raise QuantizerError("safetensors tensor payload was truncated")
                        out.write(chunk)
                        remaining -= len(chunk)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp_name, target_path)
    except Exception:
        if temp_name:
            try:
                Path(temp_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return target_path


def build_sd_cpp_command(
    cli_path: str,
    source_path: Path | str,
    output_path: Path | str,
    quant_type: str,
    scope: str,
    *,
    unet_path: Path | str | None = None,
    standalone_diffusion_model: bool = False,
) -> list[str]:
    if quant_type not in SUPPORTED_QUANT_TYPES:
        raise QuantizerError(
            f"unsupported weight type {quant_type!r}; choose one of {', '.join(SUPPORTED_QUANT_TYPES)}"
        )
    if scope not in {"all", "unet"}:
        raise QuantizerError("scope must be 'all' or 'unet'")

    if scope == "all":
        if standalone_diffusion_model:
            # Standalone UNet/DiT files need the diffusion-model namespace supplied by sd.cpp.
            input_args = ["--diffusion-model", str(Path(source_path).resolve())]
        else:
            # Monolithic checkpoints already contain their own component namespaces.
            input_args = ["-m", str(Path(source_path).resolve())]
    else:
        if unet_path is None:
            raise QuantizerError("UNet-only conversion requires an extracted UNet safetensors file")
        input_args = ["--diffusion-model", str(Path(unet_path).resolve())]

    # Force sd.cpp conversion mode. No generation arguments are passed.
    return [
        cli_path,
        "-M",
        "convert",
        *input_args,
        "-o",
        str(Path(output_path).resolve()),
        "--type",
        quant_type,
    ]


def _run_sd_cpp(
    command: list[str],
    *,
    progress: Callable[[int, int, str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> None:
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
    except OSError as exc:
        raise QuantizerError(f"could not start stable-diffusion.cpp: {exc}") from exc

    output_queue: queue.Queue[str | None] = queue.Queue()

    def collect_output() -> None:
        try:
            if process.stdout is not None:
                for line in process.stdout:
                    output_queue.put(line.rstrip())
        finally:
            output_queue.put(None)

    reader = threading.Thread(target=collect_output, daemon=True)
    reader.start()
    recent: list[str] = []
    ended = False

    while process.poll() is None or not ended or not output_queue.empty():
        if cancel_event is not None and cancel_event.is_set():
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise QuantizerCancelledError("quantization cancelled")

        try:
            line = output_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if line is None:
            ended = True
            continue
        if line:
            recent.append(line)
            if len(recent) > 12:
                recent.pop(0)
            if progress is not None:
                progress(0, 0, ("sd.cpp: " + line)[:240])

    return_code = process.wait()
    reader.join(timeout=1)
    if return_code != 0:
        detail = "\n".join(recent[-6:])
        raise QuantizerError(
            f"sd.cpp conversion failed with exit code {return_code}"
            + (f": {detail}" if detail else "")
        )


def quantize_safetensors(
    source: Path | str,
    output: Path | str,
    *,
    scope: str = "all",
    quant_type: str = "q4_0",
    cli_path: str | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> Path:
    """Quantize safetensors to GGUF through sd.cpp conversion mode only."""
    source_path = Path(source).expanduser().resolve()
    output_path = Path(output).expanduser().resolve()
    if not source_path.is_file():
        raise QuantizerError("uploaded safetensors model was not found")
    if output_path.exists():
        raise QuantizerError("output file already exists")
    if scope not in {"all", "unet"}:
        raise QuantizerError("scope must be 'all' or 'unet'")
    if quant_type not in SUPPORTED_QUANT_TYPES:
        raise QuantizerError(
            f"unsupported weight type {quant_type!r}; choose one of {', '.join(SUPPORTED_QUANT_TYPES)}"
        )

    summary = inspect_safetensors(source_path)
    if scope == "unet" and not summary["can_quantize_unet"]:
        raise QuantizerError("no UNet tensors were detected in this checkpoint")

    executable = cli_path or require_sd_cli()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    try:
        unet_path: Path | None = None
        if scope == "unet":
            temp_dir = tempfile.TemporaryDirectory(prefix="ldm-unet-quant-")
            unet_path = Path(temp_dir.name) / "unet-only.safetensors"
            extract_component_safetensors(source_path, unet_path, "unet")

        standalone_backbone = (
            scope == "all" and bool(summary.get("is_standalone_backbone"))
        )
        command = build_sd_cpp_command(
            executable,
            source_path,
            output_path,
            quant_type,
            scope,
            unet_path=unet_path,
            standalone_diffusion_model=standalone_backbone,
        )
        if progress is not None:
            progress(0, 0, f"starting sd.cpp conversion ({scope}, {quant_type})")
        _run_sd_cpp(command, progress=progress, cancel_event=cancel_event)
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise QuantizerError("sd.cpp reported success but produced no GGUF output")
        if progress is not None:
            progress(1, 1, f"GGUF ready: {output_path.name}")
        return output_path
    except QuantizerCancelledError:
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    except Exception:
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()
