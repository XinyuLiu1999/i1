"""Read SFT records from JSONL manifests or GPT-Image Parquet shards."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from itertools import islice
import json
import os
from pathlib import Path

from PIL import Image, ImageOps


@dataclass(frozen=True)
class ImageRecord:
    identifier: str
    caption: str
    width: int | None = None
    height: int | None = None
    image_path: str | None = None
    parquet_path: str | None = None
    row_group: int | None = None
    row_in_group: int | None = None
    cache_path: str | None = None
    cache_offset: int | None = None
    cache_height: int | None = None
    cache_width: int | None = None
    transform_fingerprint: str | None = None


def _import_parquet():
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise ImportError(
            "Reading GPT-Image Parquet data requires pyarrow; install the SFT dependencies from SFT.md."
        ) from error
    return pq


@lru_cache(maxsize=8)
def _parquet_reader(path):
    """Reuse shard footers within each data-loader process."""
    return _import_parquet().ParquetFile(path)


@lru_cache(maxsize=1)
def _parquet_group(path, row_group):
    # Sequential preprocessing consumes nearby images without reading the group twice.
    parquet = _parquet_reader(path)
    names = set(parquet.schema_arrow.names)
    if {"id", "prompt", "image_bytes"}.issubset(names):
        columns = ["id", "prompt", "image_bytes"]
    elif {"id", "caption", "image_bytes"}.issubset(names):
        columns = ["id", "caption", "image_bytes"]
    else:
        raise ValueError(f"Unsupported image Parquet schema at {path}")
    return parquet.read_row_group(row_group, columns=columns)


def parquet_files(source):
    source = Path(source).expanduser().resolve()
    if source.is_file() and source.suffix == ".parquet":
        return [source]
    if source.is_dir():
        files = sorted(source.glob("shard_*/shard_*.parquet"))
        if not files:
            files = sorted(source.glob("*.parquet"))
        if files:
            return files
    return []


def _parse_size(value, location):
    try:
        width, height = (int(part) for part in value.lower().split("x", 1))
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid size {value!r} at {location}; expected WIDTHxHEIGHT.") from error
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid size {value!r} at {location}.")
    return width, height


def _iter_parquet(path):
    pq = _import_parquet()
    parquet = pq.ParquetFile(path)
    required = {"id", "prompt", "size", "image_bytes"}
    missing = required.difference(parquet.schema.names)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    metadata = parquet.read(columns=["id", "prompt", "size"]).to_pydict()
    row_group = 0
    row_start = 0
    row_end = parquet.metadata.row_group(0).num_rows if parquet.num_row_groups else 0
    for row, (identifier, caption, size) in enumerate(
            zip(metadata["id"], metadata["prompt"], metadata["size"])):
        while row >= row_end:
            row_start = row_end
            row_group += 1
            row_end += parquet.metadata.row_group(row_group).num_rows
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"Invalid prompt in {path} at row {row}.")
        width, height = _parse_size(size, f"{path}:row {row}")
        yield ImageRecord(
            identifier=str(identifier), caption=caption, width=width, height=height,
            parquet_path=str(path), row_group=row_group, row_in_group=row - row_start,
        )


def _read_parquet_records(path):
    return list(_iter_parquet(path))


def _iter_jsonl(path, image_root=None):
    root = Path(image_root).expanduser().resolve() if image_root else path.parent
    with path.open() as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            caption = record.get("caption", record.get("prompt"))
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError(f"{path}:{line_no}: expected a nonempty caption or prompt string.")
            if 'cache_path' in record:
                for key in ('width', 'height', 'cache_height', 'cache_width', 'cache_offset'):
                    minimum = 0 if key == 'cache_offset' else 1
                    if type(record.get(key)) is not int or record[key] < minimum:
                        raise ValueError(f'{path}:{line_no}: invalid or missing {key}.')
                if not record.get('transform_fingerprint') or not record.get('id'):
                    raise ValueError(f'{path}:{line_no}: cached images require id and transform_fingerprint.')
                cache_path = Path(record['cache_path']).expanduser()
                if not cache_path.is_absolute():
                    cache_path = path.parent / cache_path
                yield ImageRecord(
                    identifier=str(record['id']), caption=caption,
                    width=record['width'], height=record['height'],
                    cache_path=str(cache_path.resolve()), cache_offset=record['cache_offset'],
                    cache_height=record['cache_height'], cache_width=record['cache_width'],
                    transform_fingerprint=record['transform_fingerprint'],
                )
                continue
            if "parquet_path" in record:
                if "image_path" in record:
                    raise ValueError(f"{path}:{line_no}: specify image_path or parquet_path, not both.")
                parquet_path = Path(record["parquet_path"]).expanduser()
                if not parquet_path.is_absolute():
                    parquet_path = path.parent / parquet_path
                for key in ("width", "height", "row_group", "row_in_group"):
                    value = record.get(key)
                    minimum = 1 if key in ("width", "height") else 0
                    if type(value) is not int or value < minimum:
                        raise ValueError(f"{path}:{line_no}: invalid or missing {key}.")
                if not isinstance(record.get("id"), str) or not record["id"]:
                    raise ValueError(f"{path}:{line_no}: Parquet references require a nonempty id.")
                yield ImageRecord(
                    identifier=record["id"], caption=caption,
                    width=record["width"], height=record["height"],
                    parquet_path=str(parquet_path.resolve()),
                    row_group=record["row_group"], row_in_group=record["row_in_group"],
                )
                continue
            image_path = Path(record["image_path"]).expanduser()
            image_path = image_path if image_path.is_absolute() else root / image_path
            width = int(record["width"]) if "width" in record else None
            height = int(record["height"]) if "height" in record else None
            yield ImageRecord(
                identifier=str(record.get("id", f"line-{line_no}")), caption=caption,
                width=width, height=height, image_path=str(image_path),
            )


def iter_image_records(source, image_root=None):
    """Yield records from a JSONL file, one Parquet file, or a shard directory."""
    source = Path(source).expanduser().resolve()
    files = parquet_files(source)
    if files:
        readers = min(len(files), int(os.environ.get("SFT_PARQUET_READERS", "4")))
        if readers <= 0:
            raise ValueError("SFT_PARQUET_READERS must be positive.")
        if readers == 1 or len(files) == 1:
            for path in files:
                yield from _iter_parquet(path)
        else:
            # Preserve shard order and bound prefetched prompt data to `readers`
            # shards while overlapping GPT-Image's many small metadata reads.
            with ThreadPoolExecutor(max_workers=readers) as executor:
                paths = iter(files)
                pending = deque(executor.submit(_read_parquet_records, path)
                                for path in islice(paths, readers))
                for path in paths:
                    records = pending.popleft().result()
                    yield from records
                    pending.append(executor.submit(_read_parquet_records, path))
                while pending:
                    yield from pending.popleft().result()
        return
    if not source.is_file():
        raise FileNotFoundError(f"No JSONL manifest or GPT-Image Parquet shards found at {source}")
    yield from _iter_jsonl(source, image_root=image_root)


def open_record_image(record):
    """Decode one source image without applying a training transform."""
    if record.cache_path is not None:
        raise ValueError('A pixel-cache record contains transformed pixels, not a source image.')
    if record.image_path is not None:
        source = Image.open(record.image_path)
    else:
        row = _parquet_group(record.parquet_path, record.row_group)
        index = record.row_in_group
        caption_column = "prompt" if "prompt" in row.column_names else "caption"
        if (str(row["id"][index].as_py()) != record.identifier
                or row[caption_column][index].as_py() != record.caption):
            raise ValueError(f"Stale Parquet reference or caption for {record.identifier}; rebuild the index.")
        source = Image.open(BytesIO(row["image_bytes"][index].as_py()))
    try:
        return ImageOps.exif_transpose(source).convert("RGB")
    finally:
        source.close()
