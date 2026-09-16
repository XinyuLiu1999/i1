"""Measure actual T5Gemma token lengths without loading any model weights."""
import argparse
import json

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--token_len", type=int, default=1024)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-2b-2b-ul2-it")
    lengths, pending = [], []

    def flush():
        if pending:
            encoded = tokenizer(pending, padding=False, truncation=False, add_special_tokens=True)
            lengths.extend(len(ids) for ids in encoded["input_ids"])
            pending.clear()

    with open(args.manifest) as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            caption = record.get("caption", record.get("prompt"))
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError(f"Invalid caption at line {line_no}.")
            pending.append(caption)
            if len(pending) == 128:
                flush()
    flush()
    if not lengths:
        raise ValueError("No captions found.")
    lengths = np.array(lengths)
    print(json.dumps(dict(count=len(lengths), p50=float(np.percentile(lengths, 50)),
                          p95=float(np.percentile(lengths, 95)), p99=float(np.percentile(lengths, 99)),
                          maximum=int(lengths.max()), token_limit=args.token_len,
                          exceeding_limit=int((lengths > args.token_len).sum())), indent=2))
    raise SystemExit(int((lengths > args.token_len).any()))


if __name__ == "__main__":
    main()
