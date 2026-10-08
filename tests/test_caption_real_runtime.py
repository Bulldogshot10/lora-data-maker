import pytest

pytestmark = pytest.mark.real_caption_runtime

from lora_data_maker.clip_tokenizer import (
    CLIP_CONTEXT_LENGTH,
    CLIP_TOKENIZER_ID,
    count_clip_tokens,
    load_clip_tokenizer,
    pack_clip_caption,
)


def test_real_clip_tokenizer_smoke():
    tokenizer = load_clip_tokenizer()

    assert tokenizer.__class__.__name__ == "CLIPTokenizer"
    assert tokenizer.name_or_path == CLIP_TOKENIZER_ID
    assert tokenizer.model_max_length == CLIP_CONTEXT_LENGTH

    text = "julia, standing, black dress, smiling, indoors"
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        input_ids = input_ids[0]

    assert input_ids[0] == tokenizer.bos_token_id
    assert input_ids[-1] == tokenizer.eos_token_id
    assert count_clip_tokens(text, tokenizer=tokenizer) == len(input_ids)


def test_real_clip_packer_respects_context_length():
    tokenizer = load_clip_tokenizer()
    caption = pack_clip_caption(
        [
            "julia",
            "standing with arms raised",
            "white halter top black skirt",
            "slight smile",
            "indoors with soft window lighting",
        ],
        CLIP_CONTEXT_LENGTH,
        tokenizer=tokenizer,
    )

    assert caption
    assert count_clip_tokens(caption, tokenizer=tokenizer) <= CLIP_CONTEXT_LENGTH


def test_real_llama_cpp_multimodal_handlers():
    from llama_cpp import llama_chat_format

    handlers = {
        "mtmd": "MTMDChatHandler",
        "qwen2.5-vl": "Qwen25VLChatHandler",
        "gemma4": "Gemma4ChatHandler",
        "qwen3.5": "MTMDChatHandler",
    }

    for chat_format, class_name in handlers.items():
        handler_class = getattr(llama_chat_format, class_name, None)
        assert callable(handler_class), (
            f"{class_name} is unavailable for chat format {chat_format!r}"
        )
