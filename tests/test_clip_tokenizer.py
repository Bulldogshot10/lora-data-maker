from lora_data_maker.clip_tokenizer import count_clip_tokens, pack_clip_caption


class FakeClipTokenizer:
    def __call__(self, text, **kwargs):
        value = str(text)
        return {"input_ids": list(range(len(value.split()) + 2))}


def test_count_clip_tokens_preserves_bos_and_eos():
    tokenizer = FakeClipTokenizer()
    assert count_clip_tokens("julia portrait", tokenizer=tokenizer) == 4


def test_pack_clip_caption_preserves_field_order_and_budget():
    tokenizer = FakeClipTokenizer()
    caption = pack_clip_caption(
        ["julia", "standing", "black dress", "smiling", "indoors soft lighting"],
        8,
        tokenizer=tokenizer,
    )
    assert caption == "julia, standing, black dress, smiling, indoors"
    assert count_clip_tokens(caption, tokenizer=tokenizer) == 8


def test_pack_clip_caption_uses_custom_separator():
    tokenizer = FakeClipTokenizer()
    caption = pack_clip_caption(
        ["julia", "standing", "black dress"],
        8,
        separator="; ",
        tokenizer=tokenizer,
    )
    assert caption == "julia; standing; black dress"
