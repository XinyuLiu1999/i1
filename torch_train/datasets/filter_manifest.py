"""Drop manifest records whose caption exceeds the SFT token budget.

SFT uses caption_overflow="error", so one over-length caption in a batch stops
training. This writes a filtered copy of a JSONL manifest, keeping each accepted
line byte-for-byte (Parquet/image stale checks compare captions exactly).

The output must live next to the input: relative parquet_path/image_path/
cache_path values are resolved from the manifest's directory.

    python -m datasets.filter_manifest /data/DenseText-merged/i1_manifest.jsonl
"""

import argparse
from bisect import bisect_right
from datetime import datetime, timezone
import hashlib
from itertools import islice
import json
from multiprocessing import get_context
import os
from pathlib import Path

DEFAULT_TOKENIZER = "google/t5gemma-2b-2b-ul2-it"
_TOKENIZER = None


def _init_worker(name):
    global _TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    _TOKENIZER = AutoTokenizer.from_pretrained(name)


def _count(lines, tokenizer=None):
    """Return (identifier, token count) per line, counted like tokenize_captions."""
    records = [json.loads(line) for line in lines]
    captions = []
    for number, record in enumerate(records):
        caption = record.get("caption")
        if not isinstance(caption, str):
            raise ValueError(f"Record {record.get('id', number)!r} has no string caption.")
        captions.append(caption)
    encoded = (tokenizer or _TOKENIZER)(captions, truncation=False, padding=False,
                                        add_special_tokens=True)["input_ids"]
    return [(str(record.get("id", "")), len(ids)) for record, ids in zip(records, encoded)]


def _chunks(handle, size):
    while chunk := list(islice(handle, size)):
        yield chunk


def _percentile(sorted_values, q):
    if not sorted_values:
        return None
    return sorted_values[min(len(sorted_values) - 1, int(q / 100 * len(sorted_values)))]


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def default_output(source, token_len):
    return source.with_name(f"{source.stem}.max{token_len}tok.jsonl")


def filter_manifest(source, output=None, token_len=1024, tokenizer_name=DEFAULT_TOKENIZER,
                    workers=8, chunk_lines=2048, overwrite=False, tokenizer=None, log=print):
    """Write kept lines to output and a report to <output>.report.json; return the report.

    workers=0 tokenizes in-process (with `tokenizer` if given, as in tests).
    """
    if token_len <= 0:
        raise ValueError("token_len must be positive.")
    source = Path(source).resolve()
    output = Path(output).resolve() if output else default_output(source, token_len)
    report_path = output.with_name(output.name + ".report.json")
    if output.parent != source.parent:
        raise ValueError("Write the filtered manifest next to the input so relative data paths still resolve.")
    if output == source:
        raise ValueError("Refusing to overwrite the input manifest.")
    if not overwrite and (output.exists() or report_path.exists()):
        raise FileExistsError(f"{output} or its report already exists; pass --overwrite.")

    lengths, dropped, kept = [], [], 0
    temporary = output.with_name(output.name + f".tmp{os.getpid()}")
    pool = None
    try:
        if workers:
            pool = get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(tokenizer_name,))
        elif tokenizer is None:
            _init_worker(tokenizer_name)
        with open(source, "rb") as reader, open(temporary, "wb") as writer:
            # Keep line order; blank lines carry no record and are skipped.
            chunks = ([line for line in chunk if line.strip()] for chunk in _chunks(reader, chunk_lines))
            if pool:
                counted = pool.imap(_count_lines, chunks)
            else:
                counted = ((lines, _count(lines, tokenizer)) for lines in chunks)
            for lines, counts in counted:
                for line, (identifier, count) in zip(lines, counts):
                    lengths.append(count)
                    if count > token_len:
                        dropped.append(dict(id=identifier, tokens=count))
                    else:
                        writer.write(line if line.endswith(b"\n") else line + b"\n")
                        kept += 1
                if len(lengths) % (chunk_lines * 250) < chunk_lines:
                    log(f"{len(lengths):,} records counted, {len(dropped):,} over {token_len} tokens")
            writer.flush()
            os.fsync(writer.fileno())
        if not kept:
            raise ValueError("No record fits token_len; refusing to write an empty manifest.")
        os.replace(temporary, output)
    finally:
        if pool:
            pool.terminate()
        temporary.unlink(missing_ok=True)

    lengths.sort()
    edges = list(range(0, lengths[-1] + 129, 128))
    histogram = [0] * len(edges)
    for count in lengths:
        histogram[bisect_right(edges, count) - 1] += 1
    report = dict(
        created=datetime.now(timezone.utc).isoformat(),
        input=str(source), input_sha256=_sha256(source),
        output=str(output), output_sha256=_sha256(output),
        tokenizer=tokenizer_name, token_len=token_len, add_special_tokens=True,
        records=len(lengths), kept=kept, dropped=len(dropped),
        dropped_fraction=len(dropped) / len(lengths),
        percentiles={f"p{q}": _percentile(lengths, q) for q in (50, 90, 95, 99, 99.9)} | {"max": lengths[-1]},
        histogram_128=[dict(lower=edge, upper=edge + 127, count=n) for edge, n in zip(edges, histogram) if n],
        dropped_records=dropped,
    )
    temporary = report_path.with_name(report_path.name + f".tmp{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, report_path)
    log(f"kept {kept:,}/{len(lengths):,}, dropped {len(dropped):,} (> {token_len} tokens); "
        f"wrote {output} and {report_path.name}")
    return report


def _count_lines(lines):
    return lines, _count(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", help="Input JSONL manifest, e.g. <merged>/i1_manifest.jsonl.")
    parser.add_argument("--output", help="Filtered JSONL in the same directory "
                                         "(default: <stem>.max<token_len>tok.jsonl).")
    parser.add_argument("--token_len", type=int, default=1024, help="Must match the SFT token_len.")
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1),
                        help="Tokenizer processes; 0 runs in-process.")
    parser.add_argument("--chunk_lines", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    filter_manifest(args.manifest, args.output, args.token_len, args.tokenizer,
                    args.workers, args.chunk_lines, args.overwrite)


if __name__ == "__main__":
    main()
