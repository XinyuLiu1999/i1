"""CPU regression coverage: no CUDA, model downloads, or OCR inference."""
import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import text_benchmarks as runner


class TextBenchmarksTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "checkpoint.pt-000010000"
        self.checkpoint.write_bytes(b"fake checkpoint; never loaded")
        self.output = self.root / "output"
        self.env = patch.dict(os.environ, {"OUTPUT_ROOT": str(self.output), "GPU_IDS": "2,5,7"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def args(self, *extra):
        return runner.parse_args(["--checkpoint", str(self.checkpoint), "--benchmark", "longtext",
                                  "--limit", "2", *extra])

    def execute(self, args):
        args.output_root.mkdir(exist_ok=True)
        with contextlib.redirect_stdout(io.StringIO()):
            runner.execute(args)

    def create_images(self, args):
        from PIL import Image
        for row in runner.prepare_data(args):
            Image.new("RGB", (args.resolution, args.resolution)).save(args.output_root / "images" / row["name"])

    def test_defaults_and_cvtg_scoring_supported(self):
        args = self.args()
        self.assertEqual(args.stage, "generate")
        self.assertEqual(args.prompt_variant, "original")
        self.assertIsNone(args.text_num_tokens)
        for stage in ("evaluate", "all"):
            cvtg = self.args("--benchmark", "cvtg", "--stage", stage)
            self.assertEqual(cvtg.benchmark, "cvtg-2k")
            self.assertEqual(cvtg.stage, stage)
        self.assertFalse(self.output.exists())

    def test_full_datasets_and_all_prompt_variants(self):
        for benchmark, count in (("longtext", 640), ("cvtg", 2000)):
            for variant in ("original", "simple_rewrite", "complex_rewrite"):
                rows = runner.prepare_data(self.args("--benchmark", benchmark, "--limit", "0",
                                                     "--prompt-variant", variant))
                self.assertEqual(len(rows), count)
                self.assertEqual(len({r['name'] for r in rows}), count)
                if benchmark == "longtext":
                    official = runner.read_jsonl(runner.HERE / "longtext/text_prompts.jsonl")
                    self.assertEqual([r['name'] for r in rows],
                                     [f"{r['prompt_id']}_{i}.png" for r in official for i in range(4)])
                    self.assertEqual(rows[0]['metadata'], official[0])
                else:
                    self.assertEqual(rows[-1]['name'], '01999.png')

    def test_partition_exact_coverage(self):
        for count in (1, 2, 160, 2000):
            for workers in (1, 3, 8):
                shards = list(runner.partitions(count, workers))
                self.assertEqual([i for a, b in shards for i in range(a, b)], list(range(count)))
                self.assertEqual(len(shards), min(count, workers))
                self.assertTrue(all(a < b for a, b in shards))

    def test_gpu_masks(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-abc,GPU-def"}):
            self.assertEqual(runner.gpu_ids(None, sys.executable), ["GPU-abc", "GPU-def"])
            self.assertEqual(runner.gpu_ids("3,7", sys.executable), ["3", "7"])
        for value in ("", "-1", "0,0", "0,", "0;echo bad"):
            with self.assertRaises(ValueError):
                runner.gpu_ids(value, sys.executable)

    def test_prepare_and_explicit_gpu_dry_run_never_load_models(self):
        with patch.object(runner.subprocess, "run", side_effect=AssertionError("subprocess")), \
                patch.object(runner.subprocess, "Popen", side_effect=AssertionError("model process")), \
                patch.object(runner.subprocess, "check_output", side_effect=AssertionError("GPU detection")):
            self.execute(self.args("--stage", "prepare"))
            self.execute(self.args("--stage", "all", "--dry-run"))
        samples = runner.read_jsonl(self.output / "inputs/samples.jsonl")
        self.assertEqual(len(samples), 8)
        self.assertEqual(len(runner.read_jsonl(self.output / "inputs/text_prompts.jsonl")), 2)

    def test_default_generation_shards_and_no_scoring(self):
        args = self.args("--resolution", "256")
        with patch.object(runner, "check_cuda"), patch.object(runner, "evaluate") as evaluate, \
                patch.object(runner, "run_jobs", side_effect=lambda *_: self.create_images(args)) as jobs:
            self.execute(args)
        evaluate.assert_not_called()
        calls = jobs.call_args.args[0]
        self.assertEqual([gpu for gpu, _ in calls], ["2", "5"])
        for i, (_, command) in enumerate(calls):
            for flag, expected in (("--start-idx", str(i*4)), ("--end-idx", str((i+1)*4)),
                                   ("--seed", str(42+i)), ("--rewrite-prompt", "false")):
                self.assertEqual(command[command.index(flag)+1], expected)
            self.assertNotIn("--prompt-set", command)  # Already repeated in the manifest.
            self.assertNotIn("--text-num-tokens", command)
        self.assertFalse((self.output / "eval_results").exists())

    def test_changed_checkpoint_settings_or_worker_count_rejected(self):
        self.execute(self.args("--dry-run"))
        for options in (("--seed", "43"), ("--limit", "1"), ("--prompt-variant", "simple_rewrite"),
                        ("--gpu-ids", "0")):
            with self.assertRaisesRegex(ValueError, "Run settings changed"):
                self.execute(self.args("--dry-run", *options))
        # GPU IDs may change as long as the same sample partitions are used.
        self.execute(self.args("--dry-run", "--gpu-ids", "0,1"))
        self.checkpoint.write_bytes(b"changed checkpoint")
        with self.assertRaisesRegex(ValueError, "Run settings changed"):
            self.execute(self.args("--stage", "prepare"))

    def test_missing_corrupt_and_extra_images_block_scoring(self):
        args = self.args("--stage", "evaluate", "--resolution", "256")
        self.execute(self.args("--stage", "prepare", "--resolution", "256"))
        with patch.object(runner, "evaluate") as evaluate:
            with self.assertRaisesRegex(ValueError, "missing"):
                self.execute(args)
            self.create_images(args)
            (self.output / "images/0_0.png").write_bytes(b"not PNG")
            with self.assertRaises(OSError):
                self.execute(args)
            self.create_images(args)
            (self.output / "images/unexpected.png").write_bytes(b"not PNG")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                self.execute(args)
            evaluate.assert_not_called()

    def test_failed_generation_never_scores(self):
        with patch.object(runner, "check_cuda"), \
                patch.object(runner, "run_jobs", side_effect=RuntimeError("worker failed")), \
                patch.object(runner, "evaluate") as evaluate:
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                self.execute(self.args("--stage", "all"))
            evaluate.assert_not_called()

    def judgments(self, args):
        return [{"image": str(args.output_root / "images" / row['name']),
                 "prompt": row['metadata']['prompt'], "ocr_gt": row['metadata']['text'],
                 "ocr_results": ' '.join(row['metadata']['text'])}
                for row in runner.prepare_data(args)]

    def test_incomplete_duplicate_and_wrong_ground_truth_rejected(self):
        args = self.args()
        samples = runner.prepare_data(args)
        records = self.judgments(args)
        cases = (records[:-1], records + records[:1],
                 [{**r, "ocr_gt": ["wrong ground truth"]} for r in records])
        for rows in cases:
            (self.root / "results_chunk0.jsonl").write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ValueError):
                runner.merge_results(self.root, samples, args.output_root / "images", 1)
        self.assertFalse((self.root / "results.jsonl").exists())

    def test_longtext_evaluate_merge_and_official_summary(self):
        args = self.args("--stage", "evaluate", "--resolution", "256")
        self.execute(self.args("--stage", "prepare", "--resolution", "256"))
        self.create_images(args)

        def fake_ocr(jobs, log_dir):
            gpu, command = jobs[0]
            self.assertEqual(gpu, "2,5,7")
            self.assertIn("--nproc_per_node=3", command)
            directory = Path(command[command.index("--output_dir")+1])
            for rank, (start, end) in enumerate(runner.partitions(8, 3)):
                rows = self.judgments(args)[start:end]
                (directory / f"results_chunk{rank}.jsonl").write_text(''.join(json.dumps(r)+'\n' for r in rows))

        def real_summary_without_progress_bar(command, **kwargs):
            # Execute the actual official scoring formula; only replace optional tqdm.
            module = types.ModuleType("tqdm")
            module.tqdm = lambda rows: rows
            with patch.dict(sys.modules, {"tqdm": module}), patch.object(sys, "argv", command[1:]):
                runpy.run_path(command[1], run_name="__main__")

        with patch.object(runner, "check_cuda"), patch.object(runner, "run_jobs", side_effect=fake_ocr), \
                patch.object(runner.subprocess, "run", side_effect=real_summary_without_progress_bar):
            self.execute(args)
            # A second score run starts from raw OCR, never the normalized previous result.
            self.execute(args)
        summary = json.loads((self.output / "eval_results/summary.json").read_text())
        self.assertEqual(summary['text_score'], 1.0)
        self.assertEqual(summary['image_count'], 8)
        self.assertEqual(summary['prompt_count'], 2)
        self.assertEqual((self.output / "eval_results/scores.txt").read_text(), "Text Score: 1.0000\n")
        self.assertEqual(len(runner.read_jsonl(self.output / "eval_results/results.jsonl")), 8)
        self.assertEqual(list(self.output.glob('.longtext-score-*')), [])

    def test_worker_failure_terminates_other_process_group(self):
        log_dir = self.root / "logs"
        log_dir.mkdir()
        with self.assertRaisesRegex(RuntimeError, "Worker failed"):
            runner.run_jobs([("0", [sys.executable, "-c", "raise SystemExit(7)"]),
                             ("1", [sys.executable, "-c", "import time; time.sleep(30)"])], log_dir)


if __name__ == "__main__":
    unittest.main()
