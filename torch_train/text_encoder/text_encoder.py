from __future__ import annotations

import torch
from transformers import AutoTokenizer, T5GemmaModel

TEXT_ENCODER_CONFIGS = {
    "T5Gemma": {
        "model_name": "google/t5gemma-2b-2b-ul2-it",
        "hidden_dim": 2304,
        "token_len": 256,
    },
}


class TextEncoder:
    def __init__(self, config=None, text_encoder_type: str = "T5Gemma", text_token_len=None,
                 weight_dtype: torch.dtype = torch.bfloat16, device: torch.device = "cpu"):
        if text_encoder_type not in TEXT_ENCODER_CONFIGS:
            raise ValueError(f"Unsupported text encoder type: {text_encoder_type} (only T5Gemma).")
        spec = TEXT_ENCODER_CONFIGS[text_encoder_type]
        if text_token_len is not None and text_token_len <= 0:
            raise ValueError("text_token_len must be positive.")
        self.text_encoder_type = text_encoder_type
        self.model_name = spec["model_name"]
        self.hidden_dim = spec["hidden_dim"]
        self.text_token_len = text_token_len or spec["token_len"]
        self.drop_prefix_token_len = 0
        self.weight_dtype = weight_dtype

        print(f"Loading text encoder {text_encoder_type} and tokenizer.")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.text_encoder = (
            T5GemmaModel.from_pretrained(self.model_name, dtype=weight_dtype).encoder.to(device).eval()
        )
        max_context = getattr(self.text_encoder.config, "max_position_embeddings", None)
        if max_context is not None and self.text_token_len > max_context:
            raise ValueError(f"token_len={self.text_token_len} exceeds encoder context={max_context}.")
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)
        print(f"Successfully loaded text encoder {text_encoder_type} and tokenizer.")

    def tokenize(self, prompts):
        return self.tokenizer(
            prompts,
            max_length=self.text_token_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
            add_special_tokens=True,
        )


@torch.no_grad()
def encode_text_encoder(text_encoder, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    outputs = text_encoder(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.last_hidden_state.float()
