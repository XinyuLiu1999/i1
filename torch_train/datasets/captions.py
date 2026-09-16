"""Explicit caption limits shared by SFT and inference."""


def tokenize_captions(tokenizer, captions, token_len, overflow="error"):
    if overflow not in ("error", "truncate"):
        raise ValueError("caption_overflow must be 'error' or 'truncate'.")
    if token_len <= 0:
        raise ValueError("token_len must be positive.")
    if overflow == "error":
        lengths = [len(ids) for ids in tokenizer(
            captions, truncation=False, padding=False, add_special_tokens=True)["input_ids"]]
        if any(n > token_len for n in lengths):
            raise ValueError(
                f"Caption has {max(lengths)} tokens, exceeding token_len={token_len}. "
                "Increase token_len for a new SFT run or explicitly select caption_overflow='truncate'.")
    return tokenizer(captions, max_length=token_len, padding="max_length", truncation=True,
                     return_attention_mask=True, return_tensors="pt", add_special_tokens=True)
