"""Small deterministic frozen-VAE reconstruction diagnostic on CPU, using cached weights."""

import argparse
import html
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from .bucketed import BucketedImages
from vae.vae import load_vae, encode_images_to_latents, scale_latents, reverse_scale_latents


def inspect(manifest, output_dir, count=2):
    from configs.sft_512 import get_config as config512
    from configs.sft_1024 import get_config as config1024
    if count < 1:
        raise ValueError('count must be positive.')
    torch.set_num_threads(1)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    vae = load_vae(config1024(), torch.device('cpu'), dtype=torch.float32)
    results = []
    with torch.inference_mode():
        for get_config in (config512, config1024):
            config = get_config()
            config.input.manifest = str(manifest)
            data = BucketedImages(config.input)
            groups = [group for group in data.groups if group]
            chosen = np.linspace(0, len(groups) - 1, min(count, len(groups)), dtype=int)
            for ordinal, group_index in enumerate(chosen):
                index = groups[group_index][0]
                pixels, _ = data[index]
                record = data.records[index][0]
                started = time.monotonic()
                latents = encode_images_to_latents(vae, pixels.unsqueeze(0), sample=False)
                restored = reverse_scale_latents(scale_latents(latents, config), config.vae_type)
                torch.testing.assert_close(restored, latents, rtol=1e-5, atol=1e-5)
                decoded = vae.decode(restored).sample
                if not torch.isfinite(latents).all() or not torch.isfinite(decoded).all():
                    raise ValueError(f'Nonfinite VAE output for {record.identifier}')
                reconstruction = decoded[0].permute(1, 2, 0)
                if reconstruction.shape != pixels.shape:
                    raise ValueError('VAE reconstruction shape mismatch.')
                original = (pixels + 1) / 2
                reconstructed = ((reconstruction + 1) / 2).clamp(0, 1)
                mse = (original - reconstructed).square().mean().item()
                stem = f'{config.image_size}_{ordinal:02d}'
                for label, array in [('processed', original), ('reconstructed', reconstructed)]:
                    Image.fromarray((array.numpy() * 255).round().astype(np.uint8)).save(output / f'{stem}_{label}.png')
                result = dict(id=record.identifier, resolution=config.image_size, image_shape=list(pixels.shape),
                              latent_shape=list(latents.shape), mse=mse,
                              psnr_db=float(-10 * np.log10(max(mse, 1e-12))),
                              normalization_roundtrip_passed=True, finite=True,
                              seconds=time.monotonic() - started,
                              processed=f'{stem}_processed.png', reconstructed=f'{stem}_reconstructed.png')
                results.append(result)
                print(json.dumps(result), flush=True)
    (output / 'report.json').write_text(json.dumps(dict(device='cpu', dtype='float32', posterior='mode',
                                                     note='Pixel metrics do not establish text transcription accuracy.',
                                                     examples=results), indent=2) + '\n')
    cards = ''.join(f'<h2>{row["resolution"]}: {html.escape(row["id"])}</h2>'
                    f'<p>Processed image / VAE reconstruction (click for native pixels)</p>'
                    + ''.join(f'<a href="{row[key]}"><img style="width:48%;vertical-align:top" src="{row[key]}"></a>'
                              for key in ['processed', 'reconstructed']) for row in results)
    (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>VAE inspection</title>' + cards)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--count', type=int, default=2, help='Examples per resolution, 512 and 1024.')
    args = parser.parse_args()
    inspect(args.manifest, args.output_dir, args.count)


if __name__ == '__main__':
    main()
