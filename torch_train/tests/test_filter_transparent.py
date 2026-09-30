import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.filter_transparent import SCAN_NAME, filter_transparent, image_transparency


def encode(image, fmt, **kwargs):
    buffer = io.BytesIO()
    image.save(buffer, fmt, **kwargs)
    return buffer.getvalue()


def half_transparent():
    image = Image.new("RGBA", (10, 10), (255, 0, 0, 255))
    image.paste((0, 0, 0, 0), (0, 0, 10, 5))
    return image


def palette_transparent():
    image = Image.new("P", (10, 10), 0)
    image.putpalette([0, 0, 0, 255, 255, 255] + [0] * 762)
    image.paste(1, (0, 0, 10, 2))
    return image


IMAGES = {
    "a/jpeg": encode(Image.new("RGB", (10, 10), "white"), "JPEG"),
    "a/opaque_rgba": encode(Image.new("RGBA", (10, 10), (0, 0, 255, 255)), "PNG"),
    "b/half": encode(half_transparent(), "PNG"),
    "b/palette": encode(palette_transparent(), "PNG", transparency=0),  # 80% transparent
}


def write_dataset(directory, identifiers):
    data = Path(directory) / "data"
    data.mkdir()
    pq.write_table(pa.table(dict(id=identifiers, image_bytes=[IMAGES[i] for i in identifiers])),
                   data / "part-00000.parquet", row_group_size=2)
    path = Path(directory) / "i1_manifest.jsonl"
    with open(path, "w") as handle:
        for index, identifier in enumerate(identifiers):
            handle.write(json.dumps(dict(id=identifier, caption="é", width=10, height=10,
                                         parquet_path="data/part-00000.parquet", row_group=index // 2,
                                         row_in_group=index % 2), ensure_ascii=False) + "\n")
    return path


class FilterTransparentTest(unittest.TestCase):
    def test_transparency_measurement(self):
        self.assertEqual(image_transparency(IMAGES["a/jpeg"]), ("JPEG", None, 0.0))
        self.assertEqual(image_transparency(IMAGES["a/opaque_rgba"]), ("PNG", "RGBA", 0.0))
        self.assertAlmostEqual(image_transparency(IMAGES["b/half"])[2], 0.5)
        self.assertAlmostEqual(image_transparency(IMAGES["b/palette"])[2], 0.8)

    def test_keeps_lines_verbatim_and_reuses_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_dataset(directory, list(IMAGES))
            report = filter_transparent(source, workers=0, log=lambda _: None)
            output = Path(directory) / "i1_manifest.opaque.jsonl"
            original = source.read_bytes().splitlines(keepends=True)
            self.assertEqual(output.read_bytes(), original[0] + original[1])
            self.assertEqual((report["records"], report["kept"], report["dropped"]), (4, 2, 2))
            self.assertEqual(report["dropped_by_id_prefix"], {"b": 2})
            self.assertTrue((Path(directory) / SCAN_NAME).exists())
            self.assertFalse([p for p in os.listdir(directory) if ".tmp" in p])

            with self.assertRaises(FileExistsError):
                filter_transparent(source, workers=0, log=lambda _: None)
            # A looser threshold reuses the scan and keeps the half-transparent image.
            messages = []
            report = filter_transparent(source, max_transparent=0.6, workers=0, overwrite=True,
                                        log=messages.append)
            self.assertEqual(report["kept"], 3)
            self.assertTrue(any("reusing" in m for m in messages))

    def test_rejects_stale_scan_and_unsafe_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_dataset(directory, ["a/jpeg", "b/half"])
            pq.write_table(pa.table(dict(id=["a/jpeg"], format=["JPEG"], mode=[None],
                                         transparent_frac=[0.0])), Path(directory) / SCAN_NAME)
            with self.assertRaises(ValueError):
                filter_transparent(source, workers=0, log=lambda _: None)
            os.mkdir(Path(directory) / "other")
            for output in (source, Path(directory) / "other" / "x.jsonl"):
                with self.assertRaises(ValueError):
                    filter_transparent(source, output, workers=0, log=lambda _: None)
            report = filter_transparent(source, workers=2, rescan=True, log=lambda _: None)
            self.assertEqual((report["kept"], report["dropped"]), (1, 1))


if __name__ == "__main__":
    unittest.main()
