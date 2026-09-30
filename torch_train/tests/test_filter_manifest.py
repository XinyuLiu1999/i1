import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.captions import tokenize_captions
from datasets.filter_manifest import DEFAULT_TOKENIZER, filter_manifest


class WordTokenizer:
    """One token per word plus one special token, like add_special_tokens=True."""

    def __call__(self, captions, **kwargs):
        return {"input_ids": [[0] * (len(c.split()) + 1) for c in captions]}


def write_manifest(directory, captions):
    path = Path(directory) / "i1_manifest.jsonl"
    with open(path, "w") as handle:
        for index, caption in enumerate(captions):
            handle.write(json.dumps(dict(id=f"r{index}", caption=caption, width=64, height=64,
                                         parquet_path="data/a.parquet", row_group=0,
                                         row_in_group=index), ensure_ascii=False) + "\n")
        handle.write("\n")
    return path


class FilterManifestTest(unittest.TestCase):
    def test_keeps_lines_verbatim_and_reports_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_manifest(directory, ["a b c", "a b c d", "文字 é x"])
            report = filter_manifest(source, token_len=4, workers=0, tokenizer=WordTokenizer(), log=lambda _: None)
            output = Path(directory) / "i1_manifest.max4tok.jsonl"
            original = source.read_bytes().splitlines(keepends=True)
            self.assertEqual(output.read_bytes(), original[0] + original[2])
            self.assertEqual((report["records"], report["kept"], report["dropped"]), (3, 2, 1))
            self.assertEqual(report["dropped_records"], [dict(id="r1", tokens=5)])
            self.assertEqual(report["percentiles"]["max"], 5)
            saved = json.loads(Path(str(output) + ".report.json").read_text())
            self.assertEqual(saved["output_sha256"], report["output_sha256"])
            self.assertFalse([p for p in os.listdir(directory) if ".tmp" in p])

            with self.assertRaises(FileExistsError):
                filter_manifest(source, token_len=4, workers=0, tokenizer=WordTokenizer())

    def test_rejects_unsafe_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            source = write_manifest(directory, ["a"])
            os.mkdir(Path(directory) / "other")
            for output in (source, Path(directory) / "other" / "x.jsonl"):
                with self.assertRaises(ValueError):
                    filter_manifest(source, output, token_len=4, workers=0, tokenizer=WordTokenizer())
            with self.assertRaises(ValueError):
                filter_manifest(source, token_len=1, workers=0, tokenizer=WordTokenizer())
            self.assertFalse((Path(directory) / "i1_manifest.max1tok.jsonl").exists())

    @unittest.skipUnless(os.environ.get("HF_HUB_CACHE"), "needs the cached T5Gemma tokenizer")
    def test_real_tokenizer_workers_match_training_limit(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER)
        captions = ["word " * n for n in (1, 10, 11, 12, 30)]
        with tempfile.TemporaryDirectory() as directory:
            source = write_manifest(directory, captions)
            report = filter_manifest(source, token_len=12, workers=2, chunk_lines=2, log=lambda _: None)
            kept = [json.loads(line)["caption"] for line in Path(report["output"]).read_text().splitlines()]
        for caption in captions:
            fits = True
            try:
                tokenize_captions(tokenizer, [caption], 12)
            except ValueError:
                fits = False
            self.assertEqual(caption in kept, fits)
        self.assertTrue(0 < report["kept"] < len(captions))


if __name__ == "__main__":
    unittest.main()
