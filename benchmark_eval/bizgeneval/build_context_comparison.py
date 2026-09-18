#!/usr/bin/env python3
"""Build a side-by-side HTML viewer for the SFT context comparison."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    rows_path = root / "inputs" / "bizgeneval_i1.jsonl"
    names_path = root / "inputs" / "output_names.txt"
    rows = [json.loads(line) for line in rows_path.open(encoding="utf-8") if line.strip()]
    names = [line.strip() for line in names_path.open(encoding="utf-8") if line.strip()]
    if len(rows) != len(names):
        raise ValueError("Prepared rows and output names have different lengths")

    cards: list[str] = []
    for selected_index, (row, name) in enumerate(zip(rows, names)):
        truncated = root / "images" / "truncate_1024" / name
        all_tokens = root / "images" / "all_tokens" / name
        if not truncated.is_file() or not all_tokens.is_file():
            raise FileNotFoundError(f"Missing paired outputs for {name}")
        prompt = html.escape(row["prompt"])
        metadata = html.escape(
            f"pair {selected_index + 1}/{len(rows)} · source index {row['_i1_source_index']} · "
            f"id {row.get('id', '')} · {row['_i1_text_tokens']} tokens · "
            f"{row.get('domain', '')}/{row.get('dimension', '')}"
        )
        escaped_name = html.escape(name, quote=True)
        cards.append(
            f"""
<article>
  <header><strong>{metadata}</strong></header>
  <div class="pair">
    <figure><figcaption>truncate at 1024</figcaption>
      <a href="images/truncate_1024/{escaped_name}"><img loading="lazy" src="images/truncate_1024/{escaped_name}" alt="1024-token truncated output"></a>
    </figure>
    <figure><figcaption>all {row['_i1_text_tokens']} input tokens</figcaption>
      <a href="images/all_tokens/{escaped_name}"><img loading="lazy" src="images/all_tokens/{escaped_name}" alt="all-token output"></a>
    </figure>
  </div>
  <details><summary>prompt</summary><p>{prompt}</p></details>
</article>"""
        )

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SFT context comparison</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui,sans-serif; }}
body {{ margin: 0 auto; max-width: 1600px; padding: 24px; background: #17191d; color: #eee; }}
h1 {{ margin: 0 0 8px; }} .note {{ color: #b8bec9; margin: 0 0 24px; }}
article {{ margin: 0 0 28px; padding: 16px; border: 1px solid #3b4049; border-radius: 8px; background: #202329; }}
article header {{ margin-bottom: 12px; }} .pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
figure {{ margin: 0; }} figcaption {{ margin-bottom: 7px; font-weight: 700; color: #8fd4ff; }}
img {{ display: block; width: 100%; height: auto; background: white; }}
details {{ margin-top: 12px; }} details p {{ white-space: pre-wrap; line-height: 1.45; color: #cdd2da; }}
@media (max-width: 800px) {{ .pair {{ grid-template-columns: 1fr; }} }}
</style></head><body>
<h1>SFT checkpoint: 1024-token truncation vs. all input tokens</h1>
<p class="note">{len(rows)} prompts whose T5Gemma tokenized length exceeds 1024. Click an image for its full-resolution render.</p>
{''.join(cards)}
</body></html>"""
    destination = root / "comparison.html"
    destination.write_text(document, encoding="utf-8")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
