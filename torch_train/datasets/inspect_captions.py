"""Measure actual T5Gemma token lengths without loading any model weights."""
import argparse
import heapq
import json
from pathlib import Path

import numpy as np

from .data_sources import iter_image_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True,
                        help="JSONL manifest, Parquet shard, or GPT-Image output directory.")
    parser.add_argument("--token_len", type=int, default=1024)
    parser.add_argument("--tokenizer", default="google/t5gemma-2b-2b-ul2-it")
    parser.add_argument("--report", default=None, help="Optional path for the JSON audit report.")
    parser.add_argument("--longest", type=int, default=10,
                        help="Include this many longest captions and decoded round trips in the report.")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    lengths, pending, longest = [], [], []

    def flush():
        if pending:
            encoded = tokenizer([item[1] for item in pending], padding=False, truncation=False,
                                add_special_tokens=True)
            for (identifier, caption), ids in zip(pending, encoded["input_ids"]):
                length = len(ids)
                lengths.append(length)
                item = (length, len(lengths), identifier, caption, ids)
                if args.longest > 0:
                    if len(longest) < args.longest:
                        heapq.heappush(longest, item)
                    else:
                        heapq.heappushpop(longest, item)
            pending.clear()

    for record in iter_image_records(args.manifest):
        pending.append((record.identifier, record.caption))
        if len(pending) == 128:
            flush()
    flush()
    if not lengths:
        raise ValueError("No captions found.")
    lengths = np.array(lengths)
    report = dict(
        source=str(Path(args.manifest).expanduser().resolve()), tokenizer=args.tokenizer,
        count=len(lengths), p50=float(np.percentile(lengths, 50)),
        p95=float(np.percentile(lengths, 95)), p99=float(np.percentile(lengths, 99)),
        maximum=int(lengths.max()), token_limit=args.token_len,
        exceeding_limit=int((lengths > args.token_len).sum()),
        longest=[dict(id=identifier, tokens=length, caption=caption,
                      decoded=tokenizer.decode(ids, skip_special_tokens=True))
                 for length, _, identifier, caption, ids in sorted(longest, reverse=True)],
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.report:
        path = Path(args.report).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n")
    raise SystemExit(int((lengths > args.token_len).any()))


if __name__ == "__main__":
    main()
