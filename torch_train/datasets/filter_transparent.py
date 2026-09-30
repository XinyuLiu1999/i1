"""Drop manifest records whose image is substantially transparent.

The SFT loader flattens images with .convert("RGB"), which discards alpha, so
transparent regions take whatever colour is stored underneath (often black).
Captions can describe a different background, so such rows are dropped.

Step 1 scans every Parquet file the manifest references and writes one row per
image (format, mode, share of pixels with alpha < 128) to a scan table next to
the manifest. JPEG rows are recognised from their magic bytes and never decoded.
The scan is reused on later runs, so a new threshold only reruns step 2.

Step 2 copies kept lines byte-for-byte to <stem>.opaque.jsonl next to the input
(relative parquet_path values resolve from there) and writes a report.

    python -m datasets.filter_transparent /data/DenseText-merged/i1_manifest.max1024tok.jsonl
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import io
import json
from multiprocessing import get_context
import os
from pathlib import Path
import time
import warnings

import numpy as np

from .filter_manifest import _sha256

ALPHA_CUTOFF = 128  # A pixel counts as transparent below this alpha value.
ALPHA_MODES = {"RGBA", "LA", "PA", "RGBa", "La"}
SCAN_NAME = "transparency_scan.parquet"


def image_transparency(data):
    """Return (format, mode, share of pixels with alpha < ALPHA_CUTOFF) for encoded bytes."""
    if data[:3] == b"\xff\xd8\xff":  # JPEG has no alpha channel.
        return "JPEG", None, 0.0
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    with warnings.catch_warnings(), Image.open(io.BytesIO(data)) as image:
        warnings.simplefilter("ignore")  # PIL warns on palette transparency.
        if image.mode not in ALPHA_MODES and "transparency" not in image.info:
            return image.format, image.mode, 0.0
        alpha = np.asarray(image.convert("RGBA").getchannel("A"))
        return image.format, image.mode, float((alpha < ALPHA_CUTOFF).mean())


def scan_parquet(path):
    """Scan one Parquet file; row-group granularity matches the export layout."""
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    rows = []
    for group in range(parquet.num_row_groups):
        table = parquet.read_row_group(group, columns=["id", "image_bytes"])
        for index, (identifier, data) in enumerate(zip(table["id"].to_pylist(), table["image_bytes"].to_pylist())):
            try:
                fmt, mode, fraction = image_transparency(data)
            except Exception as error:
                raise ValueError(f"{path} row group {group} row {index} ({identifier}): {error!r}") from error
            rows.append(dict(id=str(identifier), format=fmt, mode=mode, transparent_frac=fraction))
    return str(path), rows, sum(parquet.metadata.row_group(g).total_byte_size for g in range(parquet.num_row_groups))


def manifest_parquet_files(source):
    base = source.parent
    files = set()
    with open(source, "rb") as reader:
        for line in reader:
            if line.strip():
                files.add(str((base / json.loads(line)["parquet_path"]).resolve()))
    return sorted(files)


def scan(source, scan_path, workers=16, log=print):
    import pyarrow as pa
    import pyarrow.parquet as pq
    files = manifest_parquet_files(source)
    log(f"scanning {len(files):,} Parquet files with {workers} workers")
    base = source.parent.resolve()
    columns = {name: [] for name in ("id", "parquet_path", "format", "mode", "transparent_frac")}
    started, scanned_bytes = time.time(), 0
    pool = get_context("fork").Pool(workers) if workers else None
    try:
        results = pool.imap_unordered(scan_parquet, files) if pool else map(scan_parquet, files)
        for done, (path, rows, size) in enumerate(results, 1):
            relative = os.path.relpath(path, base)
            for row in rows:
                columns["parquet_path"].append(relative)
                for key, value in row.items():
                    columns[key].append(value)
            scanned_bytes += size
            if done % max(1, len(files) // 20) == 0 or done == len(files):
                elapsed = time.time() - started
                log(f"{done:,}/{len(files):,} files, {len(columns['id']):,} images, "
                    f"{scanned_bytes / 1e9:.0f} GB in {elapsed:.0f}s ({scanned_bytes / 1e6 / max(elapsed, 1e-9):.0f} MB/s)")
    finally:
        if pool:
            pool.terminate()
    temporary = scan_path.with_name(scan_path.name + f".tmp{os.getpid()}")
    pq.write_table(pa.table(columns), temporary, compression="zstd")
    os.replace(temporary, scan_path)
    log(f"wrote {scan_path}")


def filter_transparent(source, output=None, max_transparent=0.01, workers=16, rescan=False,
                       overwrite=False, log=print):
    """Write kept lines to output and a report to <output>.report.json; return the report."""
    import pyarrow.parquet as pq
    if not 0 <= max_transparent < 1:
        raise ValueError("max_transparent must be in [0, 1).")
    source = Path(source).resolve()
    output = Path(output).resolve() if output else source.with_name(f"{source.stem}.opaque.jsonl")
    report_path = output.with_name(output.name + ".report.json")
    scan_path = source.with_name(SCAN_NAME)
    if output.parent != source.parent:
        raise ValueError("Write the filtered manifest next to the input so relative data paths still resolve.")
    if output == source:
        raise ValueError("Refusing to overwrite the input manifest.")
    if not overwrite and (output.exists() or report_path.exists()):
        raise FileExistsError(f"{output} or its report already exists; pass --overwrite.")

    if rescan or not scan_path.exists():
        scan(source, scan_path, workers, log)
    else:
        log(f"reusing {scan_path} (pass --rescan to rebuild it)")
    table = pq.read_table(scan_path, columns=["id", "format", "mode", "transparent_frac"]).to_pydict()
    fractions = dict(zip(table["id"], table["transparent_frac"]))
    if len(fractions) != len(table["id"]):
        raise ValueError(f"{scan_path} contains duplicate ids.")
    modes = Counter(f"{fmt}/{mode}" if mode else fmt for fmt, mode in zip(table["format"], table["mode"]))
    del table

    kept, dropped, records = 0, [], 0
    temporary = output.with_name(output.name + f".tmp{os.getpid()}")
    try:
        with open(source, "rb") as reader, open(temporary, "wb") as writer:
            for line in reader:
                if not line.strip():
                    continue
                records += 1
                identifier = json.loads(line)["id"]
                fraction = fractions.get(identifier)
                if fraction is None:
                    raise ValueError(f"{identifier} is missing from {scan_path}; rerun with --rescan.")
                if fraction > max_transparent:
                    dropped.append(dict(id=identifier, transparent_frac=round(fraction, 4)))
                else:
                    writer.write(line if line.endswith(b"\n") else line + b"\n")
                    kept += 1
            writer.flush()
            os.fsync(writer.fileno())
        if not kept:
            raise ValueError("No record passes the threshold; refusing to write an empty manifest.")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    report = dict(
        created=datetime.now(timezone.utc).isoformat(),
        input=str(source), input_sha256=_sha256(source),
        output=str(output), output_sha256=_sha256(output),
        scan=str(scan_path), scan_sha256=_sha256(scan_path),
        alpha_cutoff=ALPHA_CUTOFF, max_transparent=max_transparent,
        records=records, kept=kept, dropped=len(dropped), dropped_fraction=len(dropped) / records,
        dropped_by_id_prefix=dict(Counter(r["id"].split("/", 1)[0] for r in dropped).most_common()),
        scanned_formats=dict(modes.most_common()),
        dropped_records=dropped,
    )
    temporary = report_path.with_name(report_path.name + f".tmp{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, report_path)
    log(f"kept {kept:,}/{records:,}, dropped {len(dropped):,} (> {max_transparent:.1%} of pixels transparent); "
        f"wrote {output} and {report_path.name}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", help="Input JSONL manifest, e.g. <merged>/i1_manifest.max1024tok.jsonl.")
    parser.add_argument("--output", help="Filtered JSONL in the same directory (default: <stem>.opaque.jsonl).")
    parser.add_argument("--max_transparent", type=float, default=0.01,
                        help=f"Drop images with more than this share of pixels at alpha < {ALPHA_CUTOFF}.")
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1),
                        help="Scan processes; 0 runs in-process.")
    parser.add_argument("--rescan", action="store_true", help=f"Rebuild {SCAN_NAME} even if it exists.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    filter_transparent(args.manifest, args.output, args.max_transparent, args.workers, args.rescan, args.overwrite)


if __name__ == "__main__":
    main()
