import pytest

pytestmark = pytest.mark.real_caption_runtime


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
