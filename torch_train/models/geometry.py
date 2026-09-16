"""Shared, checkpoint-compatible image geometry for training and inference."""
from collections import OrderedDict
import math

import numpy as np
import torch


class VariableGeometryMixin:
    def init_geometry(self, image_resolution, hidden_size, axes_dims, rope_theta):
        self.image_resolution = image_resolution
        self.hidden_size = hidden_size
        self.rope_axes_dims = tuple(axes_dims)
        self.rope_theta = rope_theta
        # Derived tensors are not parameters or checkpoint state.
        self._geometry_cache = OrderedDict()

    def patch_grid(self, x):
        h, w = x.shape[-2:]
        if h <= 0 or w <= 0 or h % self.patch_size or w % self.patch_size:
            raise ValueError(f"Latent shape {h}x{w} must be divisible by patch_size={self.patch_size}.")
        return h // self.patch_size, w // self.patch_size

    def grid_geometry(self, h, w, device):
        if (h, w) == (self.hw, self.hw):
            # Preserve the original square calculation and checkpoint tensors.
            return (self.pos_embed, self.image_row_ids, self.image_col_ids,
                    self.rope_embedder.cos_tables, self.rope_embedder.sin_tables)
        key = (h, w, device, self.pos_embed.dtype)
        if key in self._geometry_cache:
            self._geometry_cache.move_to_end(key)
            pos, rows, cols, cos, sin = self._geometry_cache[key]
            # FSDP may replace parameter views between forwards. Never retain
            # a gathered text-table parameter inside the derived-tensor cache.
            return (pos, rows, cols, [self.rope_embedder.cos_tables[0], *cos],
                    [self.rope_embedder.sin_tables[0], *sin])

        pixels_per_token = self.image_resolution / self.hw
        scale = 256.0 / (math.sqrt(h * w) * pixels_per_token)
        # Match the legacy sinusoidal axis order: columns, then rows.
        xx, yy = np.meshgrid(np.arange(w, dtype=np.float32) * scale,
                             np.arange(h, dtype=np.float32) * scale)
        axis_dim = self.hidden_size // 2
        omega = 1.0 / (10000 ** (np.arange(axis_dim // 2, dtype=np.float64) / (axis_dim / 2)))
        embeds = []
        for grid in (xx, yy):
            angles = np.outer(grid.reshape(-1), omega)
            embeds.extend((np.sin(angles), np.cos(angles)))
        pos = torch.from_numpy(np.concatenate(embeds, axis=1).astype(np.float32))
        pos = pos[None].to(device=device, dtype=self.pos_embed.dtype)
        rows = torch.arange(h, device=device).repeat_interleave(w)
        cols = torch.arange(w, device=device).repeat(h)
        cos = [self.rope_embedder.cos_tables[0]]
        sin = [self.rope_embedder.sin_tables[0]]
        for dim, length in zip(self.rope_axes_dims[1:], (h, w)):
            base = 1.0 / (self.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
            angles = (torch.arange(length, device=device, dtype=torch.float32) * scale)[:, None] * base[None]
            cos.append(angles.cos().to(cos[0].dtype))
            sin.append(angles.sin().to(sin[0].dtype))
        result = pos, rows, cols, cos, sin
        self._geometry_cache[key] = pos, rows, cols, cos[1:], sin[1:]
        if len(self._geometry_cache) > 16:
            self._geometry_cache.popitem(last=False)
        return result

    def grid_rope_freqs(self, text_tokens, text_mask, h, w):
        bsz, text_len = text_tokens.shape[:2]
        if text_len >= self.rope_embedder.cos_tables[0].shape[0]:
            raise ValueError("Caption exceeds the model's configured text RoPE range.")
        mask = text_mask if text_mask is not None else torch.ones(
            (bsz, text_len), device=text_tokens.device, dtype=torch.bool)
        _, rows, cols, cos_tables, sin_tables = self.grid_geometry(h, w, text_tokens.device)
        positions = torch.arange(text_len, device=text_tokens.device)[None].expand(bsz, -1)
        positions = torch.where(mask.bool(), positions, torch.zeros_like(positions))
        zeros = torch.zeros_like(positions)
        caption_ids = torch.stack((positions, zeros, zeros), dim=-1)
        image_ids = torch.stack((mask.to(torch.int32).sum(dim=1)[:, None].expand(-1, h * w),
                                 rows[None].expand(bsz, -1), cols[None].expand(bsz, -1)), dim=-1)
        ids = torch.cat((caption_ids, image_ids), dim=1)
        cos = torch.cat([table[ids[:, :, axis]] for axis, table in enumerate(cos_tables)], dim=-1)
        sin = torch.cat([table[ids[:, :, axis]] for axis, table in enumerate(sin_tables)], dim=-1)
        return (cos[:, text_len:], sin[:, text_len:]), (cos[:, :text_len], sin[:, :text_len])
