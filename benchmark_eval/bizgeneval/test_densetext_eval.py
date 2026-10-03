"""CPU tests: no model weights, CUDA initialization, or API calls."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import densetext_eval as runner


class DenseTextEvalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / "checkpoint.pt-000010000"
        self.checkpoint.write_bytes(b"fake checkpoint, never loaded")
        self.data = self.root / "data.jsonl"
        self.data.write_text("".join(json.dumps({
            "id": i, "domain": "slides", "dimension": "text", "prompt": f"Prompt {i}",
            "aspect_ratio": ratio, "questions": ["Readable?"], "eval_tag": "slides_text",
        }) + "\n" for i, ratio in enumerate(["16:9", "3:2", "2:3", "1:1", "9:16"])))
        self.biz = self.root / "BizGenEval"
        (self.biz / "evaluation").mkdir(parents=True)
        (self.biz / "evaluation/image_evaluation.py").touch()
        (self.biz / "config").mkdir()
        (self.biz / "config/evaluation_config.yaml").write_text("model: test\n")

    def args(self, *extra):
        return runner.parse_args([
            "--checkpoint", str(self.checkpoint), "--data-path", str(self.data),
            "--output-root", str(self.root / "output"), "--bizgeneval-root", str(self.biz),
            "--gpu-ids", "2,5,7", *extra])

    def execute(self, args):
        args.output_root.mkdir(exist_ok=True)
        runner.execute(args)

    def test_partition_coverage_and_no_idle_workers(self):
        for count in (1, 2, 5, 400):
            for workers in (1, 3, 8):
                shards = list(runner.partitions(count, workers))
                self.assertEqual([i for a, b in shards for i in range(a, b)], list(range(count)))
                self.assertEqual(len(shards), min(count, workers))
                self.assertLessEqual(max(b-a for a, b in shards) - min(b-a for a, b in shards), 1)

    def test_visible_gpu_masks_and_explicit_override(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "GPU-abcd,GPU-efgh"}):
            self.assertEqual(runner.gpu_ids(None), ["GPU-abcd", "GPU-efgh"])
            self.assertEqual(runner.gpu_ids("3,7"), ["3", "7"])
        for value in ("", "-1", "0,0", "0,", "0;echo bad"):
            with self.assertRaises(ValueError):
                runner.gpu_ids(value)

    def test_all_resolution_tiers_and_official_names(self):
        for resolution, square, anchor in ((1024, 1024, (832, 1248)),
                                            (1536, 1536, (1248, 1872)),
                                            (2048, 2048, (1664, 2496))):
            rows, names = runner.prepare_data(self.args("--resolution", str(resolution)))
            self.assertEqual(names, [f"slides_text_{i}.png" for i in range(5)])
            self.assertEqual((rows[1]["_i1_height"], rows[1]["_i1_width"]), anchor)
            self.assertEqual((rows[3]["_i1_height"], rows[3]["_i1_width"]), (square, square))
            for row in rows:
                self.assertEqual(row["_i1_height"] % 16, 0)
                self.assertLessEqual(row["_i1_height"] * row["_i1_width"], resolution ** 2)

    def test_changed_checkpoint_or_sampling_refuses_resume(self):
        args = self.args("--stage", "prepare")
        self.execute(args)
        self.execute(args)
        with self.assertRaisesRegex(ValueError, "Run settings changed"):
            self.execute(self.args("--stage", "prepare", "--seed", "43"))
        self.checkpoint.write_bytes(b"updated checkpoint")
        with self.assertRaisesRegex(ValueError, "Run settings changed"):
            self.execute(args)

    def test_gpu_count_pinned_and_evaluate_does_not_detect_cuda(self):
        with patch.object(runner, "run_workers") as workers, patch.object(runner.subprocess, "run") as run, \
                patch.object(runner, "check_cuda") as cuda:
            self.execute(self.args())
            # Default launch only generates and validates images.
            self.assertEqual(run.call_count, 1)
            self.assertIn(str(runner.HERE / "validate_images.py"), run.call_args.args[0])
            jobs = workers.call_args.args[0]
            self.assertEqual([gpu for gpu, _ in jobs], ["2", "5", "7"])
            cuda.assert_called_once_with(["2", "5", "7"])
            for index, (_, cmd) in enumerate(jobs):
                self.assertEqual(cmd[0], sys.executable)
                self.assertEqual(cmd[cmd.index("--seed") + 1], str(42 + index))
                self.assertEqual(cmd[cmd.index("--rewrite-prompt") + 1], "false")
            with self.assertRaisesRegex(ValueError, "Run settings changed"):
                self.execute(self.args("--stage", "generate", "--gpu-ids", "0"))
            run.reset_mock()
            with patch.object(runner, "gpu_ids", side_effect=AssertionError("CUDA accessed")):
                self.execute(self.args("--stage", "evaluate"))
            commands = [call.args[0] for call in run.call_args_list]
            self.assertEqual(len(commands), 4)
            self.assertIn("evaluation.image_evaluation", commands[1])
            self.assertIn("evaluation.summarize", commands[-1])

    def test_no_cuda_prevents_generation(self):
        with patch.object(runner, "check_cuda", side_effect=subprocess.CalledProcessError(1, "cuda check")), \
                patch.object(runner, "run_workers") as workers:
            with self.assertRaises(subprocess.CalledProcessError):
                self.execute(self.args("--stage", "generate"))
            workers.assert_not_called()

    def test_dry_run_does_not_start_workers_or_judge(self):
        with patch.object(runner, "check_cuda", side_effect=AssertionError("CUDA accessed")), \
                patch.object(runner, "run_workers") as workers, patch.object(runner.subprocess, "run") as run:
            self.execute(self.args("--dry-run", "--stage", "all"))
            workers.assert_not_called()
            run.assert_not_called()

    def test_partial_judgment_prevents_summary(self):
        with patch.object(runner.subprocess, "run") as run:
            run.side_effect = [None, None, subprocess.CalledProcessError(1, "validate_results")]
            with self.assertRaises(subprocess.CalledProcessError):
                self.execute(self.args("--stage", "evaluate"))
            self.assertEqual(run.call_count, 3)

    def test_actual_cpu_workers_use_disjoint_gpu_masks(self):
        jobs = [(gpu, [sys.executable, "-c",
                        "import os,pathlib,sys; pathlib.Path(sys.argv[1]).write_text(os.environ['CUDA_VISIBLE_DEVICES'])",
                        str(self.root / f"worker{i}.txt")]) for i, gpu in enumerate(("2", "5"))]
        runner.run_workers(jobs, self.root)
        self.assertEqual((self.root / "worker0.txt").read_text(), "2")
        self.assertEqual((self.root / "worker1.txt").read_text(), "5")

    def test_worker_failure_stops_other_workers(self):
        marker = self.root / "should_not_exist"
        jobs = [("0", [sys.executable, "-c", "raise SystemExit(7)"]),
                ("1", [sys.executable, "-c",
                       "import time,pathlib,sys; time.sleep(30); pathlib.Path(sys.argv[1]).touch()", str(marker)])]
        with self.assertRaisesRegex(RuntimeError, "Generation failed"):
            runner.run_workers(jobs, self.root)
        self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
