from __future__ import annotations

import math
import gc
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import os
from pathlib import Path
import json
from PIL import Image
from tqdm import tqdm

# Share geometry and caption limits with training, including when run as a script.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from torch_train.models.geometry import VariableGeometryMixin
from torch_train.datasets.captions import tokenize_captions

PROMPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "jax", "inference", "prompts"))
PROMPT_SET_CHOICES = (
    "geneval",
    "geneval_simple_rewrite",
    "geneval_complex_rewrite",
    "dpg",
    "dpg_simple_rewrite",
    "dpg_complex_rewrite",
    "prism",
    "prism_simple_rewrite",
    "prism_complex_rewrite",
    "CVTG-2K",
    "CVTG-2K_simple_rewrite",
    "CVTG-2K_complex_rewrite",
    "longtext",
    "longtext_simple_rewrite",
    "longtext_complex_rewrite",
)

MODEL_SIZE_TO_REPO_ID = {
    "1B": "zlab-princeton/i1-1B",
    "3B": "zlab-princeton/i1-3B",
}

def _get_1d_pos_embed(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.outer(pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _get_interpolated_pos_embed(
    embed_dim: int,
    grid_size: int,
    image_resolution: int,
    base_image_resolution: int = 256,
) -> np.ndarray:
    scale = float(base_image_resolution) / float(image_resolution)
    grid_h = np.arange(grid_size, dtype=np.float32) * scale
    grid_w = np.arange(grid_size, dtype=np.float32) * scale
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = _get_1d_pos_embed(embed_dim // 2, grid[0])
    emb_w = _get_1d_pos_embed(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1).astype(np.float32)


def _default_rope_axes_dims(head_dim: int) -> tuple[int, int, int]:
    if head_dim % 2 != 0:
        raise ValueError("Head dimension must be even for RoPE.")
    time_dim = head_dim // 2
    if time_dim % 2 != 0:
        time_dim -= 1
    remaining = head_dim - time_dim
    row_dim = remaining // 2
    col_dim = remaining - row_dim
    if row_dim % 2 != 0:
        row_dim -= 1
        col_dim += 1
    if col_dim % 2 != 0:
        col_dim -= 1
        row_dim += 1
    if min(time_dim, row_dim, col_dim) <= 0:
        raise ValueError("Each RoPE axis must receive at least two dimensions.")
    return time_dim, row_dim, col_dim


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        x_float = x_float * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x_float * self.scale.float()).to(dtype)


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_float = x.float()
        mean = x_float.mean(dim=-1, keepdim=True)
        var = (x_float - mean).square().mean(dim=-1, keepdim=True)
        x_float = (x_float - mean) * torch.rsqrt(var + self.eps)
        return (x_float * self.scale.float() + self.bias.float()).to(dtype)


class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int, hidden_size: int, in_channels: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.linear1 = nn.Linear(frequency_embedding_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.linear2(F.silu(self.linear1(x)))


class SwiGLUFFN(nn.Module):
    def __init__(self, hidden_size: int, hidden_features: int) -> None:
        super().__init__()
        self.w12 = nn.Linear(hidden_size, 2 * hidden_features)
        self.w3 = nn.Linear(hidden_features, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class MlpBlock(nn.Module):
    def __init__(self, hidden_size: int, hidden_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, qk_norm: bool, use_rmsnorm: bool) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        norm = RMSNorm if use_rmsnorm else LayerNorm
        self.q_norm = norm(self.head_dim) if qk_norm else None
        self.k_norm = norm(self.head_dim) if qk_norm else None
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(bsz, seq_len, self.hidden_size)
        return self.proj(out)


class TextEncoderAdapterTransformer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_size: int,
        drop_text_prob: float,
        num_heads: int,
        mlp_ratio: float,
        use_qknorm: bool,
        use_swiglu: bool,
        use_rmsnorm: bool,
        token_len: int,
    ) -> None:
        super().__init__()
        del drop_text_prob
        self.learnable_null_caption = nn.Parameter(torch.empty(1, token_len, in_channels))
        self.connector_in = nn.Linear(in_channels, hidden_size)
        norm = RMSNorm if use_rmsnorm else LayerNorm
        self.connector_norm1 = norm(hidden_size)
        self.connector_norm2 = norm(hidden_size)
        self.connector_attn = Attention(hidden_size, num_heads, use_qknorm, use_rmsnorm)
        hidden_features = int(2 / 3 * int(hidden_size * mlp_ratio)) if use_swiglu else int(hidden_size * mlp_ratio)
        self.connector_mlp = (
            SwiGLUFFN(hidden_size, hidden_features)
            if use_swiglu
            else MlpBlock(hidden_size, hidden_features)
        )
        self.connector_norm3 = norm(hidden_size)
        self.connector_norm4 = norm(hidden_size)
        self.connector_attn2 = Attention(hidden_size, num_heads, use_qknorm, use_rmsnorm)
        self.connector_mlp2 = (
            SwiGLUFFN(hidden_size, hidden_features)
            if use_swiglu
            else MlpBlock(hidden_size, hidden_features)
        )

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        x = self.connector_in(caption)
        x = x + self.connector_attn(self.connector_norm1(x))
        x = x + self.connector_mlp(self.connector_norm2(x))
        x = x + self.connector_attn2(self.connector_norm3(x))
        return x + self.connector_mlp2(self.connector_norm4(x))


class MultimodalRopeEmbedder(nn.Module):
    def __init__(
        self,
        axes_dims: tuple[int, ...],
        axes_lens: tuple[int, ...],
        axes_scales: tuple[float, ...],
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        cos_tables = []
        sin_tables = []
        for dim, axis_len, axis_scale in zip(axes_dims, axes_lens, axes_scales):
            steps = torch.arange(0, dim, 2, dtype=torch.float32)
            base = 1.0 / (theta ** (steps / dim))
            positions = torch.arange(axis_len, dtype=torch.float32) * axis_scale
            angles = positions[:, None] * base[None, :]
            cos_tables.append(angles.cos())
            sin_tables.append(angles.sin())
        self.cos_tables = nn.ParameterList([nn.Parameter(t, requires_grad=False) for t in cos_tables])
        self.sin_tables = nn.ParameterList([nn.Parameter(t, requires_grad=False) for t in sin_tables])

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos = []
        sin = []
        for axis_idx, (cos_table, sin_table) in enumerate(zip(self.cos_tables, self.sin_tables)):
            pos = position_ids[:, :, axis_idx].clamp(0, cos_table.shape[0] - 1)
            cos.append(cos_table[pos])
            sin.append(sin_table[pos])
        return torch.cat(cos, dim=-1), torch.cat(sin, dim=-1)


def _apply_multimodal_rope(
    x: torch.Tensor,
    freqs: Optional[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    if freqs is None:
        return x
    cos, sin = freqs
    dtype = x.dtype
    x_pair = x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    x0, x1 = x_pair.unbind(dim=-1)
    cos = cos[:, None].float()
    sin = sin[:, None].float()
    out = torch.stack((x0 * cos - x1 * sin, x0 * sin + x1 * cos), dim=-1)
    return out.reshape_as(x).to(dtype)


@dataclass(frozen=True)
class i1DiTForwardCache:
    text_tokens: torch.Tensor
    text_mask: Optional[torch.Tensor]
    image_freqs: tuple[torch.Tensor, torch.Tensor]
    text_freqs: tuple[torch.Tensor, torch.Tensor]
    grid_shape: tuple[int, int]


class MMDiTAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, qk_norm: bool, use_rmsnorm: bool) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv_image = nn.Linear(hidden_size, 3 * hidden_size)
        self.qkv_text = nn.Linear(hidden_size, 3 * hidden_size)
        norm = RMSNorm if use_rmsnorm else LayerNorm
        self.q_norm = norm(self.head_dim) if qk_norm else None
        self.k_norm = norm(self.head_dim) if qk_norm else None
        self.proj_image = nn.Linear(hidden_size, hidden_size)
        self.proj_text = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        image_freqs: Optional[tuple[torch.Tensor, torch.Tensor]],
        text_freqs: Optional[tuple[torch.Tensor, torch.Tensor]],
        text_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, image_len, _ = image_tokens.shape
        text_len = text_tokens.shape[1]

        def project(linear: nn.Linear, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            qkv = linear(x).reshape(bsz, x.shape[1], 3, self.num_heads, self.head_dim)
            q, k, v = qkv.unbind(dim=2)
            return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        q_image, k_image, v_image = project(self.qkv_image, image_tokens)
        q_text, k_text, v_text = project(self.qkv_text, text_tokens)
        if self.q_norm is not None:
            q_image = self.q_norm(q_image)
            k_image = self.k_norm(k_image)
            q_text = self.q_norm(q_text)
            k_text = self.k_norm(k_text)
        q_image = _apply_multimodal_rope(q_image, image_freqs)
        k_image = _apply_multimodal_rope(k_image, image_freqs)
        q_text = _apply_multimodal_rope(q_text, text_freqs)
        k_text = _apply_multimodal_rope(k_text, text_freqs)
        q = torch.cat([q_image, q_text], dim=2)
        k = torch.cat([k_image, k_text], dim=2)
        v = torch.cat([v_image, v_text], dim=2)
        key_mask = None
        attn_mask = None
        if text_mask is not None:
            image_mask = torch.ones((bsz, image_len), dtype=torch.bool, device=text_tokens.device)
            key_mask = torch.cat([image_mask, text_mask.bool()], dim=1)
            attn_mask = key_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(bsz, image_len + text_len, self.hidden_size)
        if key_mask is not None:
            out = out * key_mask[:, :, None].to(out.dtype)
        return self.proj_image(out[:, :image_len]), self.proj_text(out[:, image_len:])


class i1DiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float,
        use_qknorm: bool,
        use_swiglu: bool,
        use_rmsnorm: bool,
        use_skip: bool = False,
    ) -> None:
        super().__init__()
        self.use_skip = use_skip
        if use_skip:
            self.skip_linear_image = nn.Linear(2 * hidden_size, hidden_size)
            self.skip_linear_text = nn.Linear(2 * hidden_size, hidden_size)
        norm = RMSNorm if use_rmsnorm else LayerNorm
        self.norm1 = norm(hidden_size)
        self.norm2 = norm(hidden_size)
        self.norm3 = norm(hidden_size)
        self.norm4 = norm(hidden_size)
        self.attn = MMDiTAttention(hidden_size, num_heads, use_qknorm, use_rmsnorm)
        hidden_features = int(2 / 3 * int(hidden_size * mlp_ratio)) if use_swiglu else int(hidden_size * mlp_ratio)
        self.mlp_image = SwiGLUFFN(hidden_size, hidden_features) if use_swiglu else MlpBlock(hidden_size, hidden_features)
        self.mlp_text = SwiGLUFFN(hidden_size, hidden_features) if use_swiglu else MlpBlock(hidden_size, hidden_features)

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        image_freqs: Optional[tuple[torch.Tensor, torch.Tensor]],
        text_freqs: Optional[tuple[torch.Tensor, torch.Tensor]],
        text_mask: Optional[torch.Tensor],
        skip: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.use_skip:
            if skip is None:
                raise ValueError("Skip connection is required.")
            image_tokens = self.skip_linear_image(torch.cat([image_tokens, skip[0]], dim=-1))
            text_tokens = self.skip_linear_text(torch.cat([text_tokens, skip[1]], dim=-1))
        image_attn, text_attn = self.attn(
            self.norm1(image_tokens),
            self.norm1(text_tokens),
            image_freqs,
            text_freqs,
            text_mask,
        )
        image_tokens = image_tokens + self.norm3(image_attn)
        text_tokens = text_tokens + self.norm3(text_attn)
        image_tokens = image_tokens + self.norm4(self.mlp_image(self.norm2(image_tokens)))
        text_tokens = text_tokens + self.norm4(self.mlp_text(self.norm2(text_tokens)))
        if text_mask is not None:
            text_tokens = text_tokens * text_mask[:, :, None].to(text_tokens.dtype)
        return image_tokens, text_tokens


class FinalLayerNoAdaLN(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, use_rmsnorm: bool) -> None:
        super().__init__()
        norm = RMSNorm if use_rmsnorm else LayerNorm
        self.norm_final = norm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm_final(x))


class i1DiT(VariableGeometryMixin, nn.Module):
    def __init__(
        self,
        input_size: int = 1024 // 8,
        image_resolution: int = 1024,
        patch_size: int = 2,
        in_channels: int = 32,
        hidden_size: int = 2016,
        depth: int = 29,
        num_heads: int = 28,
        mlp_ratio: float = 4.0,
        text_embed_dim: int = 2304,
        text_num_tokens: int = 256,
        rope_theta: float = 10000.0,
        position_scale: str = "area",
        **_: object,
    ) -> None:
        super().__init__()
        if position_scale != "area":
            raise ValueError(f"Unsupported position_scale: {position_scale}")
        self.text_num_tokens = text_num_tokens
        self.input_size = input_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.x_embedder = PatchEmbed(patch_size, hidden_size, in_channels)
        hw = input_size // patch_size
        self.hw = hw
        pos = _get_interpolated_pos_embed(hidden_size, hw, image_resolution)
        self.pos_embed = nn.Parameter(torch.from_numpy(pos.reshape(1, hw * hw, hidden_size)))
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.text_encoder_adapter = TextEncoderAdapterTransformer(
            text_embed_dim,
            hidden_size,
            0.1,
            num_heads,
            mlp_ratio,
            True,
            True,
            True,
            text_num_tokens,
        )
        head_dim = hidden_size // num_heads
        axes_dims = _default_rope_axes_dims(head_dim)
        axes_lens = (text_num_tokens + 1, hw, hw)
        image_scale = 256.0 / image_resolution
        self.rope_embedder = MultimodalRopeEmbedder(
            axes_dims,
            axes_lens,
            (1.0, image_scale, image_scale),
            theta=rope_theta,
        )
        self.init_geometry(image_resolution, hidden_size, axes_dims, rope_theta)
        self.register_buffer("image_row_ids", torch.repeat_interleave(torch.arange(hw), hw), persistent=False)
        self.register_buffer("image_col_ids", torch.tile(torch.arange(hw), (hw,)), persistent=False)
        num_in_blocks = depth // 2
        self.in_blocks = nn.ModuleList(
            [
                i1DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio,
                    True,
                    True,
                    True,
                )
                for _ in range(num_in_blocks)
            ]
        )
        self.mid_block = i1DiTBlock(
            hidden_size,
            num_heads,
            mlp_ratio,
            True,
            True,
            True,
        )
        self.out_blocks = nn.ModuleList(
            [
                i1DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio,
                    True,
                    True,
                    True,
                    use_skip=True,
                )
                for _ in range(num_in_blocks)
            ]
        )
        self.final_layer = FinalLayerNoAdaLN(
            hidden_size,
            patch_size,
            self.out_channels,
            True,
        )

    def prepare_forward_cache(
        self,
        caption: torch.Tensor,
        mask: Optional[torch.Tensor],
        num_image_tokens: int,
        grid_shape: Optional[tuple[int, int]] = None,
    ) -> i1DiTForwardCache:
        h, w = grid_shape if grid_shape is not None else (self.hw, self.hw)
        if h * w != num_image_tokens:
            raise ValueError("Pass the actual grid_shape when caching a rectangular image.")
        text_tokens = self.text_encoder_adapter(caption)
        text_mask = mask.bool() if mask is not None else None
        image_freqs, text_freqs = self.grid_rope_freqs(text_tokens, text_mask, h, w)
        return i1DiTForwardCache(text_tokens, text_mask, image_freqs, text_freqs, (h, w))

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        caption: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        forward_cache: Optional[i1DiTForwardCache] = None,
    ) -> torch.Tensor:
        del t
        h, w = self.patch_grid(x)
        pos = self.grid_geometry(h, w, x.device)[0]
        tokens = self.x_embedder(x) + pos.to(dtype=x.dtype, device=x.device)
        cache = forward_cache if forward_cache is not None else self.prepare_forward_cache(caption, mask, tokens.shape[1], (h, w))
        if cache.grid_shape != (h, w):
            raise ValueError("Forward cache belongs to a different image shape.")
        text_tokens = cache.text_tokens
        text_mask = cache.text_mask
        text_freqs = cache.text_freqs
        image_freqs = cache.image_freqs
        image_tokens = tokens
        skips = []
        for block in self.in_blocks:
            image_tokens, text_tokens = block(image_tokens, text_tokens, image_freqs, text_freqs, text_mask)
            skips.append((image_tokens, text_tokens))
        image_tokens, text_tokens = self.mid_block(image_tokens, text_tokens, image_freqs, text_freqs, text_mask)
        for block in self.out_blocks:
            image_tokens, text_tokens = block(image_tokens, text_tokens, image_freqs, text_freqs, text_mask, skips.pop())
        tokens = self.final_layer(image_tokens)
        bsz = x.shape[0]
        p = self.patch_size
        tokens = tokens.reshape(bsz, h, w, p, p, self.out_channels)
        tokens = tokens.permute(0, 1, 3, 2, 4, 5).reshape(bsz, h * p, w * p, self.out_channels)
        image = tokens.permute(0, 3, 1, 2)
        return image


FLUX2_LATENTS_MEAN = [-0.06761776655912399, -0.07152235507965088, -0.07534133642911911, -0.07449393719434738, 0.022278539836406708, 0.017995379865169525, 0.014197370037436485, 0.01836133562028408, -6.275518535403535e-05, -0.006251443177461624, -0.00021015340462327003, -0.0031394739635288715, -0.027202727273106575, -0.02810601517558098, -0.027645578607916832, -0.029033277183771133, -0.0768895298242569, -0.06717019528150558, -0.09018829464912415, -0.08921381831169128, 0.016836659982800484, 0.015206480398774147, 0.00790204294025898, 0.008579261600971222, 0.008347982540726662, 0.0015409095212817192, 0.0002583497844170779, -0.004281752277165651, -0.043877143412828445, -0.04189559817314148, -0.04378034919500351, -0.043148837983608246, -0.010246668942272663, -0.013186423107981682, -0.006620197091251612, -0.004766239318996668, -0.031062893569469452, -0.03055436909198761, -0.027904054149985313, -0.01795399747788906, 0.0030211929697543383, 0.001502539962530136, 0.012592565268278122, 0.0144742326810956, 0.034720875322818756, 0.03376586362719536, 0.033663298934698105, 0.02829528972506523, 0.0019797170534729958, 0.004728920292109251, 0.004654144402593374, 0.004963618237525225, 0.012272646650671959, 0.008096166886389256, 0.00805679615586996, 0.014576919376850128, 0.06810732930898666, 0.06790295243263245, 0.07665354013442993, 0.07318653911352158, -0.04621443152427673, -0.04739413782954216, -0.03918757662177086, -0.05109340697526932, -0.05277586728334427, -0.04773825407028198, -0.047003958374261856, -0.0517151840031147, -0.03170523792505264, -0.03163386881351471, -0.03446723148226738, -0.02825590781867504, 0.050968676805496216, 0.04450491443276405, 0.057813018560409546, 0.04580356180667877, -0.0411602221429348, -0.04582904279232025, -0.048741210252046585, -0.04673927649855614, -0.008838738314807415, -0.010627646930515766, -0.008805501274764538, -0.004613492637872696, -0.03758484125137329, -0.043219830840826035, -0.043574366718530655, -0.049890533089637756, 0.011846445500850677, 0.016636915504932404, 0.020284568890929222, 0.027899663895368576, 0.011271224357187748, 0.01290129590779543, 0.0015593513380736113, 0.007155619561672211, -0.01180021371692419, -0.0018362690461799502, -0.014141527935862541, -0.005370706785470247, -0.009097136557102203, -0.013795508071780205, -0.014467928558588028, -0.01869881898164749, 0.03225415572524071, 0.030501458793878555, 0.02587026357650757, 0.02995659038424492, 0.05399540066719055, 0.06144390255212784, 0.049539074301719666, 0.05898929387331009, -0.051080696284770966, -0.06032619997859001, -0.047775182873010635, -0.052397292107343674, -0.022676242515444756, -0.027419250458478928, -0.015365149825811386, -0.025462470948696136, -0.05720777437090874, -0.056476689875125885, -0.05176353082060814, -0.049556463956832886, 0.011585467495024204, 0.0054222596809268, 0.01630038022994995, 0.010384724475443363]
FLUX2_LATENTS_VAR = [3.2502119541168213, 3.163407325744629, 3.192434072494507, 3.1813714504241943, 3.1389076709747314, 3.0941381454467773, 3.1011831760406494, 3.0550901889801025, 3.0051753520965576, 3.0179455280303955, 3.0067572593688965, 3.0076351165771484, 3.4690163135528564, 3.432523727416992, 3.470231533050537, 3.45538592338562, 3.0949840545654297, 3.071377754211426, 3.0819239616394043, 3.091344118118286, 3.014709711074829, 3.027461051940918, 3.01198673248291, 3.0252928733825684, 3.0074563026428223, 2.9741339683532715, 3.024878978729248, 2.9940483570098877, 3.080418586730957, 3.0669093132019043, 3.0831477642059326, 3.058147430419922, 3.403618097305298, 3.4055330753326416, 3.44087290763855, 3.435497283935547, 3.326714277267456, 3.1730010509490967, 3.1874520778656006, 3.22017240524292, 3.2569847106933594, 3.1953234672546387, 3.130955457687378, 3.124211549758911, 3.1620266437530518, 3.1209557056427, 3.2129595279693604, 3.185375690460205, 3.090271472930908, 3.030029058456421, 3.0565788745880127, 3.0162465572357178, 3.225846767425537, 3.2391276359558105, 3.211076259613037, 3.21309494972229, 3.161032199859619, 3.149500846862793, 3.142376184463501, 3.150174379348755, 3.071641206741333, 3.0439963340759277, 3.1177477836608887, 3.0607917308807373, 3.1593689918518066, 3.139946222305298, 3.1729917526245117, 3.1730189323425293, 3.2984564304351807, 3.244508981704712, 3.248305559158325, 3.251725673675537, 3.0720319747924805, 3.00360369682312, 3.084465742111206, 3.056194543838501, 3.100954532623291, 3.064960479736328, 3.1261374950408936, 3.102006435394287, 3.120508909225464, 3.0782599449157715, 3.178100109100342, 3.141893148422241, 3.2024238109588623, 3.2396669387817383, 3.1909685134887695, 3.1540026664733887, 3.102187395095825, 3.106377601623535, 3.08341121673584, 3.0892975330352783, 3.1621134281158447, 3.1226611137390137, 3.1719861030578613, 3.168121337890625, 2.958735942840576, 2.9129180908203125, 2.980844497680664, 2.9209375381469727, 3.165689706802368, 3.08971905708313, 3.0632121562957764, 3.0465474128723145, 3.0928444862365723, 3.0622732639312744, 3.0709831714630127, 3.014193534851074, 3.103145122528076, 3.087780714035034, 3.042872667312622, 3.0380074977874756, 3.065497875213623, 3.10084867477417, 3.109544038772583, 3.101743698120117, 2.976869583129883, 2.935845136642456, 2.999986171722412, 2.9673469066619873, 3.1200692653656006, 3.105872631072998, 3.139338493347168, 3.12007999420166, 3.0474750995635986, 3.0419390201568604, 3.086534261703491, 3.072920083999634]


def _prompt_path(filename):
    return os.path.join(PROMPT_DIR, filename)


def prompts_from_args(args) -> list[str]:
    prompts = list(args.prompt or [])
    if args.prompts_file:
        with open(os.path.expanduser(args.prompts_file), "r", encoding="utf-8") as handle:
            prompts.extend(line.strip() for line in handle if line.strip())
    elif args.prompts_jsonl:
        with open(os.path.expanduser(args.prompts_jsonl), "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = row.get(args.jsonl_prompt_key)
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(
                        f"{args.prompts_jsonl}:{line_number} has no non-empty "
                        f"{args.jsonl_prompt_key!r} string."
                    )
                prompts.append(prompt.strip())
    elif args.prompt_set == "geneval":
        with open(_prompt_path("geneval.jsonl")) as fp:
            metadatas = [json.loads(line) for line in fp]
        prompts = [metadata["prompt"] for metadata in metadatas]
    elif args.prompt_set == "geneval_simple_rewrite":
        with open(_prompt_path("geneval_simple_rewrite.txt"), "r") as fp:
            prompts = fp.readlines()
        prompts = [item.strip() for item in prompts]
    elif args.prompt_set == "geneval_complex_rewrite":
        with open(_prompt_path("geneval_complex_rewrite.jsonl")) as fp:
            metadatas = [json.loads(line) for line in fp]
        prompts = [metadata["prompt"] for metadata in metadatas]
    elif args.prompt_set == "dpg":
        with open(_prompt_path("dpg.json"), "r") as f:
            prompts = json.load(f)
        prompts = [item[1].strip() for item in prompts]
    elif args.prompt_set == "dpg_simple_rewrite":
        with open(_prompt_path("dpg_simple_rewrite.json"), "r") as f:
            prompts = json.load(f)
        prompts = [item[1].strip() for item in prompts]
    elif args.prompt_set == "dpg_complex_rewrite":
        with open(_prompt_path("dpg_complex_rewrite.json"), "r") as f:
            prompts = json.load(f)
        prompts = [item[1].strip() for item in prompts]
    elif args.prompt_set == "prism":
        with open(_prompt_path("prism.json"), "r") as f:
            prompts = json.load(f)
    elif args.prompt_set == "prism_simple_rewrite":
        with open(_prompt_path("prism_simple_rewrite.json"), "r") as f:
            prompts = json.load(f)
    elif args.prompt_set == "prism_complex_rewrite":
        with open(_prompt_path("prism_complex_rewrite.json"), "r") as f:
            prompts = json.load(f)
    elif args.prompt_set == "CVTG-2K":
        with open(_prompt_path("CVTG-2K.json"), "r") as f:
            prompts = json.load(f)
        prompts = [item[1].strip() for item in prompts]
    elif args.prompt_set == "CVTG-2K_simple_rewrite":
        with open(_prompt_path("CVTG-2K_simple_rewrite.json"), "r") as f:
            prompts = json.load(f)
        prompts = [item[1].strip() for item in prompts]
    elif args.prompt_set == "CVTG-2K_complex_rewrite":
        with open(_prompt_path("CVTG-2K_complex_rewrite.json"), "r") as f:
            prompts = json.load(f)
        prompts = [item[1].strip() for item in prompts]
    elif args.prompt_set == "longtext":
        prompts = []
        with open(_prompt_path("longtext.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                prompts.append(obj["prompt"])
    elif args.prompt_set == "longtext_simple_rewrite":
        prompts = []
        with open(_prompt_path("longtext_simple_rewrite.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                prompts.append(obj["prompt"])
    elif args.prompt_set == "longtext_complex_rewrite":
        prompts = []
        with open(_prompt_path("longtext_complex_rewrite.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                prompts.append(obj["prompt"])
    return prompts


def prepare_rewrite_prompts(tokenizer, texts: list[str], system_prompt: str) -> list[str]:
    formatted_prompts = []
    for user_input in texts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Input to Rewrite:\n{user_input}"},
        ]
        formatted_prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        )
    return formatted_prompts


def rewrite_prompts(prompts: list[str], device: torch.device, model_name: str, batch_size: int) -> list[str]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    with open(Path(__file__).with_name("metaprompt.txt"), 'r', encoding='utf-8') as f:
        system_prompt = f.read().strip()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
    ).to(device).eval()
    rewritten = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(prompts), batch_size), desc="Rewriting prompts"):
            batch_prompts = prompts[start : start + batch_size]
            formatted_prompts = prepare_rewrite_prompts(tokenizer, batch_prompts, system_prompt)
            tokenized = tokenizer(formatted_prompts, padding=True, return_tensors="pt")
            inputs = {key: value.to(device) for key, value in tokenized.items()}
            outputs = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                max_new_tokens=16384,
                pad_token_id=tokenizer.eos_token_id,
            )
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            batch_rewritten = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            rewritten.extend(text.strip() or prompt for text, prompt in zip(batch_rewritten, batch_prompts))

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rewritten


def encode_prompt(tokenizer, text_encoder, prompts: list[str], device: torch.device,
                  token_len: int = 256, caption_overflow: str = "error"):
    tokenized = tokenize_captions(tokenizer, prompts, token_len, caption_overflow)
    inputs = {key: value.to(device) for key, value in tokenized.items()}
    with torch.inference_mode():
        outputs = text_encoder(**inputs)
    hidden = outputs.last_hidden_state.float()
    mask = inputs.get("attention_mask")
    if mask is None:
        mask = torch.ones(hidden.shape[:2], dtype=torch.bool, device=device)
    return hidden, mask.bool()



def time_grid(num_steps: int, shift: float, device: torch.device) -> torch.Tensor:
    times = torch.linspace(0.0, 1.0, num_steps + 1, dtype=torch.bfloat16, device=device)
    if shift != 0.0:
        times = (shift * times) / (1.0 + (shift - 1.0) * times)
    return times


def prepare_cfg_conditioning(model, text: torch.Tensor, mask: torch.Tensor):
    batch, cond_len, _ = text.shape
    uncond = model.text_encoder_adapter.learnable_null_caption.to(device=text.device, dtype=text.dtype)
    if uncond.shape[0] == 1 and batch > 1:
        uncond = uncond.repeat(batch, 1, 1)
    uncond_len = uncond.shape[1]
    if uncond_len < cond_len:
        uncond = torch.cat([uncond, torch.zeros(batch, cond_len - uncond_len, uncond.shape[2], device=text.device)], dim=1)
        uncond_mask = mask & (torch.arange(cond_len, device=text.device)[None] < uncond_len)
    else:
        uncond = uncond[:, :cond_len]
        uncond_mask = mask
    return torch.cat([text, uncond], dim=0), torch.cat([mask, uncond_mask], dim=0)


def denoise_latents(model, text, mask, args, device, generator=None, height=None, width=None):
    height = height or getattr(args, "height", None) or model.image_resolution
    width = width or getattr(args, "width", None) or model.image_resolution
    multiple = 8 * model.patch_size
    if height <= 0 or width <= 0 or height % multiple or width % multiple:
        raise ValueError(f"Image height and width must be positive multiples of {multiple}.")
    shape = (text.shape[0], model.in_channels, height // 8, width // 8)
    gen = generator if generator is not None else torch.Generator(device=device)
    latents = torch.randn(shape, generator=gen, device=device, dtype=torch.bfloat16)
    text = text.to(dtype=torch.bfloat16)

    cfg_text, cfg_mask = prepare_cfg_conditioning(model, text, mask)
    grid = model.patch_grid(latents)
    forward_cache = model.prepare_forward_cache(cfg_text, cfg_mask, grid[0] * grid[1], grid)
    times = time_grid(args.num_steps, args.inference_timestep_shift, device)
    guidance = torch.full((text.shape[0], 1, 1, 1), args.cfg_scale, device=device, dtype=torch.bfloat16)

    for idx in tqdm(range(args.num_steps), desc="Denoising"):
        t = times[idx].expand(latents.shape[0])
        latent_input = torch.cat([latents, latents], dim=0)
        t_input = torch.cat([t, t], dim=0)
        velocity = model(latent_input, t_input, cfg_text, cfg_mask, forward_cache)
        cond, uncond = velocity.chunk(2, dim=0)
        velocity = cond + (guidance - 1.0) * (cond - uncond)
        if args.cfg_rescale is not None:
            axes = tuple(range(1, velocity.ndim))
            std_c = torch.std(cond.float(), dim=axes, keepdim=True)
            std_g = torch.std(velocity.float(), dim=axes, keepdim=True)
            factor = (std_c / (std_g + 1e-8)).to(dtype=velocity.dtype)
            velocity = velocity * (1.0 - args.cfg_rescale + args.cfg_rescale * factor)
        latents = latents + (times[idx + 1] - times[idx]) * velocity
    return latents


def reverse_scale_flux2_latents(latents: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = latents.shape
    latents = latents.reshape(batch, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4).reshape(batch, channels * 4, height // 2, width // 2)
    mean = torch.tensor(FLUX2_LATENTS_MEAN, device=latents.device, dtype=latents.dtype).reshape(1, -1, 1, 1)
    var = torch.tensor(FLUX2_LATENTS_VAR, device=latents.device, dtype=latents.dtype).reshape(1, -1, 1, 1)
    latents = latents * torch.sqrt(var + 0.0001) + mean
    batch, channels, height, width = latents.shape
    latents = latents.reshape(batch, channels // 4, 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3).reshape(batch, channels // 4, height * 2, width * 2)
    return latents


def decode_vae(vae, latents: torch.Tensor, batch_size: int):
    images = []
    if latents.device.type == "cuda":
        torch.cuda.empty_cache()
    for start in range(0, latents.shape[0], batch_size):
        latent_batch = reverse_scale_flux2_latents(latents[start : start + batch_size])
        decoded = vae.decode(latent_batch).sample
        image_batch = (decoded / 2 + 0.5).clamp(0, 1)
        image_batch = (image_batch.permute(0, 2, 3, 1) * 255).round().to(torch.uint8).cpu().numpy()
        images.append(image_batch)
        del latent_batch, decoded, image_batch
    return np.concatenate(images, axis=0)


def build_model(device, checkpoint_path: str, dtype=torch.bfloat16, text_num_tokens: int | None = None):
    # mmap avoids materializing the checkpoint's optimizer/train state in host
    # RAM when an SFT checkpoint also contains resumable training state.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    config = dict(checkpoint["config"])
    checkpoint_text_num_tokens = int(config.get("text_num_tokens", 256))
    if text_num_tokens is not None:
        if text_num_tokens <= 0:
            raise ValueError("text_num_tokens must be positive.")
        if text_num_tokens < checkpoint_text_num_tokens:
            raise ValueError(
                f"text_num_tokens={text_num_tokens} is smaller than the checkpoint's "
                f"native context ({checkpoint_text_num_tokens})."
            )
        config["text_num_tokens"] = text_num_tokens

    model = i1DiT(**config).to(device=device, dtype=dtype).eval()
    model.checkpoint_text_num_tokens = checkpoint_text_num_tokens
    state = checkpoint["model"]
    if text_num_tokens is None or text_num_tokens == checkpoint_text_num_tokens:
        model.load_state_dict(state, strict=True)
        return model

    # Match the SFT initialization compatibility path: rebuild deterministic
    # geometry tensors and tile the learned null-caption prefix to the new
    # token length. All other learned weights load unchanged.
    filtered = {
        name: value
        for name, value in state.items()
        if "pos_embed" not in name and "rope_embedder" not in name
    }
    null_name = "text_encoder_adapter.learnable_null_caption"
    source_null = filtered[null_name]
    target_null = model.state_dict()[null_name]
    if source_null.shape != target_null.shape:
        if source_null.shape[0] != target_null.shape[0] or source_null.shape[2] != target_null.shape[2]:
            raise ValueError(
                f"Cannot resize null caption from {tuple(source_null.shape)} "
                f"to {tuple(target_null.shape)}."
            )
        repeats = (target_null.shape[1] + source_null.shape[1] - 1) // source_null.shape[1]
        filtered[null_name] = source_null.repeat(1, repeats, 1)[:, : target_null.shape[1]].clone()

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    unexpected = list(unexpected)
    disallowed_missing = [
        name for name in missing if "pos_embed" not in name and "rope_embedder" not in name
    ]
    if disallowed_missing or unexpected:
        raise RuntimeError(
            f"Extended checkpoint mismatch: missing={disallowed_missing[:8]} "
            f"unexpected={unexpected[:8]}"
        )
    print(
        f"Extended checkpoint text context: {checkpoint_text_num_tokens} -> "
        f"{text_num_tokens} tokens (tiled null caption; rebuilt RoPE)."
    )
    return model


def select_text_context_length(tokenizer, prompts: list[str], model, dynamic: bool) -> int:
    """Choose either the checkpoint-native context or the configured maximum."""
    maximum = model.text_num_tokens
    native = getattr(model, "checkpoint_text_num_tokens", maximum)
    if not dynamic or native >= maximum:
        return maximum
    raw_lengths = [
        len(token_ids)
        for token_ids in tokenizer(
            prompts,
            truncation=False,
            padding=False,
            add_special_tokens=True,
        )["input_ids"]
    ]
    return native if max(raw_lengths, default=0) <= native else maximum


def jsonl_image_sizes(
    path: str,
    height_key: str | None,
    width_key: str | None,
) -> list[tuple[int, int]] | None:
    """Read optional per-prompt image sizes prepared by benchmark pipelines."""
    if height_key is None and width_key is None:
        return None
    if not height_key or not width_key:
        raise ValueError("--jsonl-height-key and --jsonl-width-key must be set together.")

    sizes = []
    with open(os.path.expanduser(path), "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                height = int(row[height_key])
                width = int(row[width_key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{line_number} must contain positive integer fields "
                    f"{height_key!r} and {width_key!r}."
                ) from exc
            if height <= 0 or width <= 0:
                raise ValueError(
                    f"{path}:{line_number} has invalid image size {height}x{width}."
                )
            sizes.append((height, width))
    return sizes


def str2bool(v):
    """
    Converts string to bool type; enables command line 
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="samples")
    parser.add_argument("--skip-existing", action="store_true", help="Resume without regenerating existing PNG files.")
    parser.add_argument("--checkpoint", help="Local original or SFT checkpoint; otherwise download the selected release.")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--caption-overflow", choices=("error", "truncate"), default="error")
    parser.add_argument("--text-num-tokens", type=int, default=None, help="Override checkpoint text context length.")
    parser.add_argument(
        "--dynamic-text-context",
        action="store_true",
        help="Use the checkpoint's native context for short prompts and the override for longer prompts.",
    )
    parser.add_argument("--prompt", action="append")
    parser.add_argument("--prompts-file")
    parser.add_argument("--prompts-jsonl", help="JSONL prompts; reads the field selected by --jsonl-prompt-key.")
    parser.add_argument("--jsonl-prompt-key", default="prompt")
    parser.add_argument("--jsonl-height-key", help="Optional JSONL field containing each prompt's image height.")
    parser.add_argument("--jsonl-width-key", help="Optional JSONL field containing each prompt's image width.")
    parser.add_argument("--output-names-file", help="Optional line-delimited output filenames matching the prompts.")
    parser.add_argument("--prompt-set", type=str, choices=PROMPT_SET_CHOICES)
    parser.add_argument("--rewrite-batch-size", type=int, default=1)
    parser.add_argument("--diffusion-batch-size", type=int, default=1)
    parser.add_argument("--cfg-scale", type=float, default=12)
    parser.add_argument("--cfg-rescale", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument("--inference-timestep-shift", type=float, default=0.3)
    parser.add_argument("--rewrite-prompt", type=str2bool, default=True)
    parser.add_argument("--rewriter-model", default="Qwen/Qwen3-30B-A3B", choices=["Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-4B-Instruct-2507"])
    parser.add_argument("--vae-batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None, help="Seed for the initial diffusion latents.")
    parser.add_argument("--model-size", choices=tuple(MODEL_SIZE_TO_REPO_ID), default="3B")
    parser.add_argument("--resolution", type=int, choices=(256, 512, 1024), default=1024)
    parser.add_argument("--start-idx", type=int, default=None)
    parser.add_argument("--end-idx", type=int, default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer, T5GemmaModel
    from diffusers import AutoencoderKL
    from huggingface_hub import hf_hub_download

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    os.makedirs(args.outdir, exist_ok=True)

    prompts = prompts_from_args(args)
    image_sizes = None
    if args.jsonl_height_key or args.jsonl_width_key:
        if not args.prompts_jsonl:
            raise ValueError("Per-prompt image sizes require --prompts-jsonl.")
        if args.prompt:
            raise ValueError("Do not combine --prompt with per-prompt JSONL image sizes.")
        image_sizes = jsonl_image_sizes(
            args.prompts_jsonl,
            args.jsonl_height_key,
            args.jsonl_width_key,
        )
        if len(image_sizes) != len(prompts):
            raise ValueError(
                f"Expected one image size per prompt ({len(prompts)}), got {len(image_sizes)}."
            )
    if args.rewrite_prompt:
        prompts = rewrite_prompts(prompts, device, args.rewriter_model, args.rewrite_batch_size)
        print(prompts)
    if args.prompt_set is not None:
        if ("geneval" in args.prompt_set) or ("dpg" in args.prompt_set) or ("longtext" in args.prompt_set):
            prompts = [item for item in prompts for _ in range(4)]
    output_names = None
    if args.output_names_file:
        with open(os.path.expanduser(args.output_names_file), "r", encoding="utf-8") as handle:
            output_names = [line.strip() for line in handle if line.strip()]
        if len(output_names) != len(prompts):
            raise ValueError(
                f"Expected one output name per prompt ({len(prompts)}), got {len(output_names)}."
            )
        if len(set(output_names)) != len(output_names):
            raise ValueError("Output names must be unique.")
        for name in output_names:
            if Path(name).name != name or not name.lower().endswith(".png"):
                raise ValueError(f"Output names must be PNG basenames, got: {name!r}")

    def output_path(image_index: int) -> str:
        if output_names is not None:
            return os.path.join(args.outdir, output_names[image_index])
        if args.prompt_set is not None and (("geneval" in args.prompt_set) or ("dpg" in args.prompt_set)):
            prompt_index, repeat_index = divmod(image_index, 4)
            return os.path.join(args.outdir, f"{prompt_index:05d}", "samples", f"{repeat_index:05d}.png")
        if args.prompt_set is not None and (("prism" in args.prompt_set) or ("CVTG-2K" in args.prompt_set)):
            return os.path.join(args.outdir, f"{image_index:05d}.png")
        if args.prompt_set is not None and "longtext" in args.prompt_set:
            prompt_index, repeat_index = divmod(image_index, 4)
            return os.path.join(args.outdir, f"{prompt_index:04d}_{repeat_index}.png")
        return os.path.join(args.outdir, f"{image_index:06d}.png")

    def output_exists(image_index: int) -> bool:
        """Treat a PNG with the wrong requested geometry as incomplete."""
        path = output_path(image_index)
        if not os.path.isfile(path):
            return False
        expected = None
        if image_sizes is not None:
            height, width = image_sizes[image_index]
            expected = (width, height)
        elif args.height is not None and args.width is not None:
            expected = (args.width, args.height)
        if expected is None:
            return True
        try:
            with Image.open(path) as existing_image:
                return existing_image.size == expected
        except (OSError, ValueError):
            return False

    start_idx = max(0, args.start_idx) if args.start_idx is not None else 0
    end_idx = min(len(prompts), args.end_idx) if args.end_idx is not None else len(prompts)
    if args.skip_existing and all(output_exists(index) for index in range(start_idx, end_idx)):
        print(f"Skipping sample indices [{start_idx}, {end_idx}): all output files already exist.")
        sys.exit(0)

    checkpoint_path = args.checkpoint or hf_hub_download(
        repo_id=MODEL_SIZE_TO_REPO_ID[args.model_size],
        filename=f"{args.resolution}_resolution_checkpoint_torch.pt",
        repo_type="model",
    )
    model = build_model(device, checkpoint_path, text_num_tokens=args.text_num_tokens)

    tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-2b-2b-ul2-it")
    text_encoder = T5GemmaModel.from_pretrained(
        "google/t5gemma-2b-2b-ul2-it",
        dtype=torch.bfloat16,
    ).encoder.to(device).eval()
    vae = AutoencoderKL.from_pretrained("black-forest-labs/FLUX.2-dev", subfolder="vae").to(device=device, dtype=torch.bfloat16).eval()

    latent_generator = None
    if args.seed is not None:
        latent_generator = torch.Generator(device=device).manual_seed(args.seed)
    reported_text_lengths = set()
    with torch.inference_mode():
        start = start_idx
        while start < end_idx:
            batch_end = min(start + args.diffusion_batch_size, end_idx)
            if image_sizes is not None:
                # A tensor batch must share one spatial shape. Shorten the
                # batch at a resolution boundary while retaining one model load.
                batch_size = image_sizes[start]
                while batch_end > start + 1 and any(
                    size != batch_size for size in image_sizes[start:batch_end]
                ):
                    batch_end -= 1
                height, width = batch_size
            else:
                height = args.height or model.image_resolution
                width = args.width or model.image_resolution
            batch_prompts = prompts[start:batch_end]
            batch_indices = range(start, start + len(batch_prompts))
            if args.skip_existing and all(output_exists(index) for index in batch_indices):
                # Preserve the seeded latent stream so a partially resumed run
                # produces the same later samples as an uninterrupted run.
                if latent_generator is not None:
                    skipped = torch.randn(
                        (len(batch_prompts), model.in_channels, height // 8, width // 8),
                        generator=latent_generator,
                        device=device,
                        dtype=torch.bfloat16,
                    )
                    del skipped
                print(f"Skipping existing sample indices [{start}, {start + len(batch_prompts)}).")
                start = batch_end
                continue
            native_text_num_tokens = getattr(model, "checkpoint_text_num_tokens", model.text_num_tokens)
            batch_text_num_tokens = select_text_context_length(
                tokenizer,
                batch_prompts,
                model,
                args.dynamic_text_context,
            )
            if batch_text_num_tokens not in reported_text_lengths:
                print(
                    f"Using text context length {batch_text_num_tokens} for this checkpoint "
                    f"(native={native_text_num_tokens}, maximum={model.text_num_tokens})."
                )
                reported_text_lengths.add(batch_text_num_tokens)
            text, mask = encode_prompt(
                tokenizer,
                text_encoder,
                batch_prompts,
                device,
                batch_text_num_tokens,
                args.caption_overflow,
            )
            latents = denoise_latents(
                model,
                text,
                mask,
                args,
                device,
                latent_generator,
                height=height,
                width=width,
            )
            images = decode_vae(vae, latents, args.vae_batch_size)
            
            for offset, image in enumerate(images):
                curr_image_index = start + offset
                save_path = output_path(curr_image_index)
                if args.skip_existing and output_exists(curr_image_index):
                    continue
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                Image.fromarray(image).save(save_path)
            del text, mask, latents, images
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            start = batch_end
