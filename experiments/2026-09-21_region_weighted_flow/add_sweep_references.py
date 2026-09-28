#!/usr/bin/env python3
"""Add existing BizGenEval image sets to a completed sweep gallery without inference."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from PIL import Image

from evaluate_sweep import PROJECT, write_comparison

SETTINGS = {
    '01': ('Base', '256 tokens · square'),
    '03': ('Full SFT · step 6262', '1024 tokens · native buckets'),
    '04': ('Captioned SFT · step 6245', '1024 tokens · native buckets'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=PROJECT /
                        'artifacts/region_weighted_flow_2026-09-21/sweep_500/evaluation_40prompts_50steps')
    parser.add_argument('--images-root', type=Path, default=PROJECT / 'artifacts/bizgeneval_evaluation/images')
    parser.add_argument('--sets', nargs='+', choices=SETTINGS, default=['01', '03', '04'])
    args = parser.parse_args()
    output = args.output_root.expanduser().resolve()
    images = args.images_root.expanduser().resolve()
    manifest = json.loads((output / 'evaluation.json').read_text())
    inputs = {name: (output / 'inputs' / name).read_bytes() for name in manifest['inputs_sha256']}
    for name, payload in inputs.items():
        if hashlib.sha256(payload).hexdigest() != manifest['inputs_sha256'][name]:
            raise ValueError(f'Evaluation input checksum mismatch: {name}')
    names = inputs['output_names.txt'].decode().splitlines()
    rows = [json.loads(line) for line in inputs['metadata.jsonl'].decode().splitlines()]
    if len(rows) != len(names) or len(set(names)) != len(names):
        raise ValueError('Evaluation rows and filenames do not match')
    if any(Path(name).name != name for name in names):
        raise ValueError('Expected plain image filenames')
    source_rows = [json.loads(line) for line in
                   (images.parent / 'inputs/bizgeneval_i1.jsonl').read_text().splitlines() if line.strip()]
    source_by_name = {f"{row['domain']}_{row['dimension']}_{row['id']}.png": row for row in source_rows}
    for name, row in zip(names, rows):
        if name not in source_by_name or source_by_name[name]['prompt'] != row['prompt']:
            raise ValueError(f'Reference prompt differs or is missing: {name}')
    run_config = dict(line.split('=', 1) for line in (images.parent / 'run_config.txt').read_text().splitlines()
                      if '=' in line)
    columns = []
    copies = []
    for prefix in dict.fromkeys(args.sets):
        directories = sorted(path for path in images.glob(prefix + '_*') if path.is_dir())
        if len(directories) != 1:
            raise ValueError(f'Expected one image set for {prefix}: {directories}')
        source = directories[0]
        title, detail = SETTINGS[prefix]
        directory = Path('references') / source.name
        sizes = {}
        for name in names:
            with Image.open(source / name) as image:
                image.load()
                sizes[name] = list(image.size)
            copies.append((source / name, output / directory / name))
        columns.append(dict(directory=str(directory), title=f'{prefix} · {title}',
                            detail=f"{detail} · {run_config.get('num_steps', 'unknown')} denoising steps",
                            source_directory=str(source), image_sizes=sizes))
    # Validate all selected references before changing the existing gallery.
    for source, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix('.png.tmp')
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    note = (f"Reference columns {' / '.join(dict.fromkeys(args.sets))} reuse existing renders with "
            f"{run_config.get('num_steps', 'unknown')} denoising steps and base seed "
            f"{run_config.get('seed', 'unknown')} (original GPU IDs: {run_config.get('gpu_ids', 'unknown')}). "
            'Prompt IDs and text match. Seeds/context lengths/output geometry differ from the sweep; '
            'these columns are historical references, not matched loss ablations. '
            'Original image aspect ratios are preserved.')
    references = dict(format='region-sweep-references-v1', columns=columns, note=note,
                      source_run_config=run_config, matched_prompts=len(names))
    path = output / 'comparison_references.json'
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(references, indent=2) + '\n')
    temporary.replace(path)
    settings = SimpleNamespace(output_root=output, checkpoint_step=manifest['checkpoint_step'],
                               num_prompts=manifest['num_prompts'], num_steps=manifest['num_steps'])
    page = write_comparison(settings, list(manifest['checkpoints']), inputs)
    print(f'Added {len(columns)} reference sets x {len(names)} matched prompts: {page}')


if __name__ == '__main__':
    main()
