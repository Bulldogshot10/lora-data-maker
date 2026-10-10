import json
import struct
from pathlib import Path

import pytest

from lora_data_maker.quantizer import (
    QuantizerError,
    build_sd_cpp_command,
    extract_component_safetensors,
    inspect_safetensors,
)


def write_safetensors(path: Path, tensors: dict[str, bytes]) -> None:
    header = {}
    payload = bytearray()
    for name, value in tensors.items():
        start = len(payload)
        payload.extend(value)
        header[name] = {
            "dtype": "U8",
            "shape": [len(value)],
            "data_offsets": [start, len(payload)],
        }
    header["__metadata__"] = {"format": "pt"}
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def test_scan_detects_unet_clip_and_vae_tensor_counts(tmp_path):
    source = tmp_path / "source.safetensors"
    write_safetensors(
        source,
        {
            "model.diffusion_model.input_blocks.0.weight": b"1234",
            "cond_stage_model.transformer.text_model.embeddings.weight": b"ab",
            "first_stage_model.encoder.conv_in.weight": b"xyz",
            "misc_tensor": b"0",
        },
    )

    result = inspect_safetensors(source)

    assert result["tensor_count"] == 4
    assert result["components"]["unet"] == {"tensor_count": 1, "bytes": 4}
    assert result["components"]["clip"] == {"tensor_count": 1, "bytes": 2}
    assert result["components"]["vae"] == {"tensor_count": 1, "bytes": 3}
    assert result["components"]["other"] == {"tensor_count": 1, "bytes": 1}
    assert result["can_quantize_unet"] is True


def test_extract_unet_writes_only_unet_tensors_without_loading_weights(tmp_path):
    source = tmp_path / "source.safetensors"
    output = tmp_path / "unet-only.safetensors"
    write_safetensors(
        source,
        {
            "model.diffusion_model.input_blocks.0.weight": b"UNET",
            "cond_stage_model.transformer.text_model.embeddings.weight": b"CLIP",
            "first_stage_model.encoder.conv_in.weight": b"VAE",
        },
    )

    extract_component_safetensors(source, output, "unet")
    result = inspect_safetensors(output)

    assert result["tensor_count"] == 1
    assert result["components"]["unet"] == {"tensor_count": 1, "bytes": 4}
    assert result["components"]["clip"]["tensor_count"] == 0
    assert result["components"]["vae"]["tensor_count"] == 0

    content = output.read_bytes()
    header_size = struct.unpack("<Q", content[:8])[0]
    header = json.loads(content[8:8 + header_size])
    assert list(k for k in header if k != "__metadata__") == [
        "model.diffusion_model.input_blocks.0.weight"
    ]
    assert content[8 + header_size:] == b"UNET"
    assert header["__metadata__"] == {"format": "pt"}


def test_extract_unet_strips_component_prefix_when_needed(tmp_path):
    source = tmp_path / "diffusers-unet.safetensors"
    output = tmp_path / "unet-only.safetensors"
    write_safetensors(source, {"diffusion_model.down_blocks.0.weight": b"abcd"})

    extract_component_safetensors(source, output, "unet")

    content = output.read_bytes()
    header_size = struct.unpack("<Q", content[:8])[0]
    header = json.loads(content[8:8 + header_size])
    assert "down_blocks.0.weight" in header
    assert "diffusion_model.down_blocks.0.weight" not in header


def test_build_sd_cpp_command_always_uses_convert_mode(tmp_path):
    source = tmp_path / "model.safetensors"
    output = tmp_path / "model-q4_0.gguf"
    command = build_sd_cpp_command(
        "/opt/sd.cpp/bin/sd-cli",
        source,
        output,
        "q4_0",
        "all",
    )

    assert command[:3] == ["/opt/sd.cpp/bin/sd-cli", "-M", "convert"]
    assert "-m" in command
    assert "--diffusion-model" not in command
    assert "--type" in command
    assert "q4_0" in command


def test_build_sd_cpp_unet_command_passes_only_extracted_unet(tmp_path):
    unet = tmp_path / "unet-only.safetensors"
    output = tmp_path / "unet-q8_0.gguf"
    command = build_sd_cpp_command(
        "sd-cli",
        tmp_path / "original.safetensors",
        output,
        "q8_0",
        "unet",
        unet_path=unet,
    )

    assert command[:3] == ["sd-cli", "-M", "convert"]
    assert "--diffusion-model" in command
    assert str(unet.resolve()) in command
    assert "-m" not in command


def test_unet_extraction_rejects_checkpoint_without_unet(tmp_path):
    source = tmp_path / "clip-only.safetensors"
    output = tmp_path / "out.safetensors"
    write_safetensors(source, {"cond_stage_model.transformer.weight": b"1234"})

    with pytest.raises(QuantizerError, match="no UNet tensors"):
        extract_component_safetensors(source, output, "unet")
