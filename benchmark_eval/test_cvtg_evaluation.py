"""CPU regression tests for CVTG orchestration and metric aggregation (no downloads)."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import cvtg_evaluation as cvtg
import text_benchmarks as runner

spec = importlib.util.spec_from_file_location("cvtg_backend", cvtg.CVTG / "unified_metrics_eval.py")
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


class CVTGTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "checkpoint.pt"
        self.checkpoint.write_bytes(b"not loaded")
        self.output = self.root / "run"
        self.args = runner.parse_args(["--checkpoint", str(self.checkpoint), "--benchmark", "cvtg",
                                       "--output-root", str(self.output), "--limit", "5",
                                       "--resolution", "256", "--gpu-ids", "2,5,7", "--stage", "prepare"])
        self.output.mkdir()
        with contextlib.redirect_stdout(io.StringIO()):
            runner.execute(self.args)
        self.samples = cvtg.load_samples(cvtg.CVTG / "prompts", self.output / "images",
                                         self.output / "inputs/samples.jsonl")

    def create_images(self):
        from PIL import Image
        for row in self.samples:
            Image.new("RGB", (256, 256)).save(row["image"])

    def records(self, samples):
        rows = []
        for index, sample in enumerate(samples):
            total = len(backend.UnifiedMetricsEvaluator.extract_words_from_prompt(sample["prompt"]))
            rows.append({**sample, "total_words": total, "correct_words": total if index % 2 else 0,
                         "ned_word_data": [0.25 if index % 2 else 0.75] * total,
                         "clipscore": 0.5 + index / 10, "vqascore": index / max(len(samples), 1),
                         "aesthetic_score": 4 + index / 10})
        return rows

    def write_chunks(self, records, shards):
        for rank, (start, end) in enumerate(shards):
            (self.root / f"results_chunk{rank}.jsonl").write_text(
                ''.join(json.dumps(r) + '\n' for r in records[start:end]))

    def test_weighted_summary_is_independent_of_worker_count(self):
        # Deliberately unequal area sizes AND word counts to catch averaging shard means.
        samples = [dict(self.samples[i], prompt="'one'" if i == 0 else "'two three four'",
                        area=2 if i < 2 else 3) for i in range(5)]
        records = self.records(samples)
        for workers in (1, 2, 3, 8):
            shards = list(runner.partitions(len(samples), workers))
            self.write_chunks(records, shards)
            rows, result = cvtg.merge_results(self.root, samples, shards)
            overall = result['overall_results']
            self.assertEqual(rows, records)
            self.assertEqual(overall['total_words'], 13)
            self.assertAlmostEqual(overall['word_accuracy'], 6 / 13)
            self.assertAlmostEqual(overall['ned'], (7 * .75 + 6 * .25) / 13)
            self.assertAlmostEqual(overall['clipscore'], .7)
            self.assertAlmostEqual(overall['vqascore'], .4)
            self.assertEqual([r['total_images'] for r in result['area_results']], [2, 3])

    def test_invalid_missing_duplicate_and_nonfinite_results_rejected(self):
        records = self.records(self.samples)
        cases = [records[:-1], records + records[:1],
                 [{**records[0], 'prompt': 'wrong'}, *records[1:]],
                 [{**records[0], 'total_words': 100}, *records[1:]],
                 [{**records[0], 'ned_word_data': []}, *records[1:]],
                 [{**records[0], 'vqascore': float('nan')}, *records[1:]],
                 [{**records[0], 'vqascore': 2}, *records[1:]],
                 [{**records[0], 'image': 'wrong.png'}, *records[1:]]]
        for rows in cases:
            with self.subTest(rows=rows[0]):
                (self.root / 'results_chunk0.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                with self.assertRaises(ValueError):
                    cvtg.merge_results(self.root, self.samples, [(0, len(self.samples))])

    def test_rewritten_generation_uses_original_evaluation_prompt(self):
        manifest = self.output / 'inputs/samples.jsonl'
        rows = runner.read_jsonl(manifest)
        rows[0]['prompt'] = 'Rewritten generation prompt'
        manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        samples = cvtg.load_samples(cvtg.CVTG / 'prompts', self.output / 'images', manifest)
        self.assertEqual(samples[0]['prompt'], rows[0]['metadata']['prompt'])
        rows[0]['metadata']['prompt'] = 'Wrong original'
        manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        with self.assertRaisesRegex(ValueError, 'metadata'):
            cvtg.load_samples(cvtg.CVTG / 'prompts', self.output / 'images', manifest)

    def test_legacy_layout_requires_full_coverage_and_rejects_duplicates(self):
        official = cvtg.official_prompts(cvtg.CVTG / 'prompts')
        legacy = self.root / 'legacy'
        for row in official.values():
            path = legacy / row['benchmark_type'] / str(row['area']) / (row['id'].rsplit('_', 1)[1] + '.png')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        samples = cvtg.load_samples(cvtg.CVTG / 'prompts', legacy)
        self.assertEqual(len(samples), 2000)
        first = Path(samples[0]['image'])
        first.with_suffix('.jpg').touch()
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            cvtg.load_samples(cvtg.CVTG / 'prompts', legacy)
        first.with_suffix('.jpg').unlink()
        first.unlink()
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            cvtg.load_samples(cvtg.CVTG / 'prompts', legacy)

    def test_explicit_gpu_dry_run_never_imports_models_or_checks_cuda(self):
        self.args.stage, self.args.dry_run = 'all', True
        with patch.object(cvtg, 'check_cuda', side_effect=AssertionError('CUDA probe')), \
                patch.object(cvtg, 'run_jobs', side_effect=AssertionError('worker')), \
                patch.object(cvtg, 'validate_samples', side_effect=AssertionError('image validation')), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            runner.execute(self.args)
        printed = output.getvalue()
        self.assertIn('CUDA_VISIBLE_DEVICES=2', printed)
        self.assertIn('--start 2 --end 4', printed)
        self.assertIn('--start 4 --end 5', printed)
        self.assertNotIn('torch.distributed.run', printed)
        self.assertFalse((self.output / 'eval_results/results.json').exists())

    def fake_workers(self, jobs, log_dir):
        self.assertEqual([gpu for gpu, _ in jobs], ['2', '5', '7'])
        records = self.records(self.samples)
        for _, command in jobs:
            self.assertEqual(command[0], self.args.evaluation_python)
            value = lambda flag: command[command.index(flag) + 1]
            self.assertEqual(value('--clip-batch-size'), '16')
            self.assertEqual(value('--device'), 'cuda')
            self.assertIn('--no_hf_mirror', command)
            start, end = int(value('--start')), int(value('--end'))
            Path(value('--output_file')).write_text(''.join(json.dumps(r)+'\n' for r in records[start:end]))

    def test_runner_evaluate_and_all_publish_scores_without_moving_images(self):
        self.create_images()
        self.args.stage = 'evaluate'
        with patch.object(cvtg, 'check_cuda'), patch.object(cvtg, 'run_jobs', side_effect=self.fake_workers), \
                contextlib.redirect_stdout(io.StringIO()):
            runner.execute(self.args)
            # Evaluation worker count can change independently of generation.
            self.args.stage = 'all'
            with patch.object(runner, 'check_cuda'), patch.object(runner, 'run_jobs'):
                runner.execute(self.args)
        result = json.loads((self.output / 'eval_results/results.json').read_text())
        summary = json.loads((self.output / 'eval_results/results_summary.json').read_text())
        self.assertEqual(result['overall_results']['total_images'], 5)
        self.assertEqual(summary['checkpoint'], str(self.checkpoint.resolve()))
        self.assertEqual(len(runner.read_jsonl(self.output / 'eval_results/results.jsonl')), 5)
        self.assertEqual(len(list((self.output / 'images').glob('*.png'))), 5)
        self.assertEqual(list((self.output / 'eval_results').glob('.cvtg-score-*')), [])

    def test_failure_preserves_previous_successful_results(self):
        self.create_images()
        self.args.stage = 'evaluate'
        with patch.object(cvtg, 'check_cuda'), patch.object(cvtg, 'run_jobs', side_effect=self.fake_workers), \
                contextlib.redirect_stdout(io.StringIO()):
            runner.execute(self.args)
        outputs = list((self.output / 'eval_results').glob('*.json*'))
        previous = {p: p.read_bytes() for p in outputs}
        for failure in ('crash', 'incomplete'):
            def failing_jobs(jobs, logs):
                if failure == 'crash':
                    raise RuntimeError('model loading failed')
                self.fake_workers(jobs, logs)
                command = jobs[0][1]
                Path(command[command.index('--output_file')+1]).write_text('')
            with patch.object(cvtg, 'check_cuda'), patch.object(cvtg, 'run_jobs', side_effect=failing_jobs), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises((RuntimeError, ValueError)):
                runner.execute(self.args)
            self.assertEqual({p: p.read_bytes() for p in outputs}, previous)

    def test_real_worker_processes_use_isolated_gpus_and_shared_cache(self):
        self.create_images()
        original_command = cvtg.worker_command
        bootstrap = r"""
import json, os, runpy, sys, types
from pathlib import Path
class FakeEvaluator:
    def __init__(self, device, cache_dir):
        assert device == 'cuda'
        assert os.environ['CUDA_VISIBLE_DEVICES'] in ('2', '5', '7')
        assert os.environ['HF_HUB_CACHE'] == str(cache_dir)
        assert 'HF_ENDPOINT' not in os.environ
    def score_samples(self, samples, batch_size):
        import re
        assert batch_size == 2
        for sample in samples:
            total = sum(len(s.split()) for s in re.findall(r"'(.*?)'", sample['prompt']))
            yield {**sample, 'total_words': total, 'correct_words': total,
                   'ned_word_data': [1.] * total, 'clipscore': 1., 'vqascore': .5, 'aesthetic_score': 5.}
module = types.ModuleType('unified_metrics_eval')
module.UnifiedMetricsEvaluator = FakeEvaluator
sys.modules['unified_metrics_eval'] = module
sys.argv = sys.argv[1:]
sys.path.insert(0, str(Path(sys.argv[0]).resolve().parent))
runpy.run_path(sys.argv[0], run_name='__main__')
"""
        def fake_backend_command(*args, **kwargs):
            command = original_command(*args, **kwargs)
            return [command[0], '-c', bootstrap, *command[1:]]
        output = self.root / 'process_results/scores.json'
        with patch.object(cvtg, 'check_cuda'), patch.object(cvtg, 'worker_command', side_effect=fake_backend_command), \
                contextlib.redirect_stdout(io.StringIO()):
            cvtg.evaluate_dataset(self.samples, output, ['2', '5', '7'], batch_size=2,
                                  cache_dir=self.root / 'cache')
        result = json.loads(output.read_text())
        self.assertEqual(result['overall_results']['word_accuracy'], 1.)
        self.assertEqual(result['overall_results']['total_images'], 5)
        self.assertEqual(len(list((output.parent / 'logs').glob('worker_*.log'))), 3)

    def test_concurrent_evaluation_is_rejected(self):
        import fcntl
        output = self.root / 'scores.json'
        with output.with_suffix('.lock').open('w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, 'Another evaluation'):
                cvtg.evaluate_dataset(self.samples, output, ['2'], dry_run=True)

    def test_cli_validation_and_environment_selection(self):
        for benchmark, variable in (('longtext', 'LONGTEXT_PYTHON'), ('cvtg', 'CVTG_PYTHON')):
            with patch.dict(os.environ, {variable: '/evaluation/python'}):
                args = runner.parse_args(['--checkpoint', str(self.checkpoint), '--benchmark', benchmark])
                self.assertEqual(args.evaluation_python, '/evaluation/python')
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cvtg.parse_args(['--result_dir', 'images', '--output_file', 'scores.json', '--clip-batch-size', '0'])
        with patch.dict(os.environ, {}, clear=False):
            cvtg.configure_cache(self.root / 'cache', True)
            self.assertEqual(os.environ['HF_ENDPOINT'], 'https://hf-mirror.com')
            cvtg.configure_cache(self.root / 'cache', False)
            self.assertNotIn('HF_ENDPOINT', os.environ)
            self.assertEqual(os.environ['TRANSFORMERS_CACHE'], str((self.root / 'cache').resolve()))

    def test_clip_batches_are_bounded_and_inference_errors_propagate(self):
        evaluator = backend.UnifiedMetricsEvaluator.__new__(backend.UnifiedMetricsEvaluator)
        sizes = []
        def clip(paths, texts):
            sizes.append(len(paths))
            return [.5] * len(paths)
        with patch.object(evaluator, 'compute_clip_score_batch', side_effect=clip), \
                patch.object(evaluator, 'compute_ocr_metrics', return_value=(1, 1, [1.])), \
                patch.object(evaluator, 'compute_vqa_score', return_value=.7), \
                patch.object(evaluator, 'compute_aesthetic_score', return_value=5.):
            rows = list(evaluator.score_samples(self.samples, batch_size=2))
            self.assertEqual(sizes, [2, 2, 1])
            self.assertEqual(len(rows), 5)
            with patch.object(evaluator, 'compute_vqa_score', side_effect=RuntimeError('CUDA out of memory')):
                with self.assertRaisesRegex(RuntimeError, 'out of memory'):
                    list(evaluator.score_samples(self.samples, batch_size=2))


if __name__ == '__main__':
    unittest.main()
