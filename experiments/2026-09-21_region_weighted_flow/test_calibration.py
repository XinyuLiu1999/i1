"""Offline gradient geometry, actual FSDP2, and initialization-only runner checks."""
from contextlib import ExitStack
import dataclasses
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "torch_train"))

import torch
from torch.utils.checkpoint import checkpoint
from calibration import (RegionCalibration, add_gradients, build_report, geometry,
                         gradient_products, measure_gradients, regional_share, solve_coefficient)
from config import get_config
from models.dit import DualStreamDiTConfig, i1DiT
from region_loss import region_flow_loss
from test_region_flow import fixture
from training import main as trainer


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Conv2d(2, 4, 1)
        self.last = torch.nn.Conv2d(4, 2, 1)
        self.first.bias.requires_grad_(False)

    def forward(self, x):
        return checkpoint(self.last, torch.tanh(self.first(x)), use_reentrant=False)


def measure(model, x, target, mask, sharded=False, accumulation=1):
    global_grads = regional_grads = None
    for a, b, c in zip(x.chunk(accumulation), target.chunk(accumulation), mask.chunk(accumulation)):
        _, metrics = region_flow_loss(model(a), b, c)
        gg, gr = measure_gradients(model, metrics['flow_loss'] / accumulation,
                                  metrics['region_flow_loss'] / accumulation, sharded=sharded)
        global_grads = add_gradients(global_grads, gg)
        regional_grads = add_gradients(regional_grads, gr)
    return gradient_products(global_grads, regional_grads, x.device, distributed=sharded)


def fsdp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist = torch.distributed
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard
        torch.manual_seed(7)
        reference = TinyModel()
        model = TinyModel()
        model.load_state_dict(reference.state_dict())
        # Match the trainer's two-dimensional mesh, including size-one replica axis.
        mesh = init_device_mesh('cpu', (1, 2), mesh_dim_names=('data', 'fsdp'))
        fully_shard(model.first, mesh=mesh)
        fully_shard(model.last, mesh=mesh)
        fully_shard(model, mesh=mesh)
        for accumulation in (1, 2):
            x, target = torch.randn(8, 2, 3, 5), torch.randn(8, 2, 3, 5)
            mask = torch.rand(8, 3, 5)
            mask[:4] = 0  # One rank has no regional supervision at all.
            expected = measure(reference, x, target, mask, accumulation=accumulation)
            sl = slice(rank * 4, (rank + 1) * 4)
            actual = measure(model, x[sl], target[sl], mask[sl], True, accumulation)
            torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected), rtol=2e-5, atol=1e-7)
            assert all(p.grad is None for p in model.parameters())
    finally:
        dist.destroy_process_group()


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_solver_alignment_and_scale(self):
        for cosine in (-1., -.9, 0., .7, 1.):
            for r in (1e-8, .1, 10.):
                for target in (.3, .4, .5):
                    coefficient = solve_coefficient(4., r*r, cosine*2*r, target)
                    self.assertGreater(coefficient, 0)
                    self.assertAlmostEqual(regional_share(4., r*r, cosine*2*r, coefficient), target)
        for values in ((1., 0., 0., .4), (0., 1., 0., .4), (1., 1., 0., 1.),
                       (float('nan'), 1., 0., .4)):
            with self.assertRaises(ValueError):
                solve_coefficient(*values)

    def test_shared_graph_gradients_and_accumulation(self):
        torch.manual_seed(3)
        model = TinyModel()
        x, target = torch.randn(4, 2, 3, 5), torch.randn(4, 2, 3, 5)
        mask = torch.rand(4, 3, 5)
        mask[1] = 0
        expected = measure(model, x, target, mask)
        for accumulation in (1, 2, 4):
            actual = measure(model, x, target, mask, accumulation=accumulation)
            torch.testing.assert_close(torch.tensor(actual), torch.tensor(expected))
        # Compare direct gradients of the combined objective to G²+2λD+λ²R².
        loss, _ = region_flow_loss(model(x), target, mask, 2.5)
        loss.backward()
        squared = sum(p.grad.square().sum().item() for p in model.parameters() if p.grad is not None)
        self.assertAlmostEqual(squared, expected[0] + 5*expected[2] + 6.25*expected[1], places=6)

    def test_empty_masks_reports_and_nonfinite(self):
        rows = [dict(bucket='8x8', **geometry(4., 1., .5)),
                dict(bucket='8x8', **geometry(9., 0., 0.))]
        report = build_report(rows)
        self.assertEqual(report['all_batches']['G2'], 13.)
        self.assertEqual(report['nonempty_region_batches']['batches'], 1)
        self.assertEqual(rows[1]['shares']['lambda_40'], 0.)
        self.assertAlmostEqual(report['all_batches']['shares']['lambda_40']['aggregate'], .4)
        json.dumps(report, allow_nan=False)
        with self.assertRaisesRegex(ValueError, 'nonzero'):
            build_report([rows[1]])
        with self.assertRaises(FloatingPointError):
            gradient_products([torch.tensor(float('nan'))], [None], torch.device('cpu'))

    def test_fsdp_two_passes_match_full_batch(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(fsdp_worker, args=(str(Path(directory) / 'rendezvous'),), nprocs=2, join=True)

    def test_actual_runner_no_updates_no_resume_no_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_config, tokenizer = fixture(root)
            config = get_config()
            config.input.update(data_config)
            config.image_size, config.token_len = 64, 16
            config.fsdp_axis_size = 1
            config.perceptual_weight = 0
            config.amp = False
            config.use_grad_ckpt = True
            small = DualStreamDiTConfig(input_size=8, image_resolution=64, in_channels=32,
                                       hidden_size=64, depth=3, num_heads=4, text_embed_dim=24,
                                       text_num_tokens=16, drop_text_prob=0., use_grad_ckpt=True)
            initial = i1DiT(small)
            initial.init_weights()
            torch.save({'model': initial.state_dict()}, root / 'initial.pt')
            model = i1DiT(small)
            bundle = SimpleNamespace(tokenizer=tokenizer, text_encoder=None, text_token_len=16, hidden_dim=24)
            workdir = root / 'calibration'
            argv = ['calibrate.py', '--config', 'unused', '--manifest', data_config['manifest'],
                    '--init_from', str(root / 'initial.pt'), '--workdir', str(workdir),
                    '--calibration-batches', '2']

            def encode_images(vae, images):
                rgb = torch.nn.functional.interpolate(images.permute(0, 3, 1, 2), scale_factor=1/8)
                return rgb.repeat(1, 11, 1, 1)[:, :32]

            with ExitStack() as stack:
                stack.enter_context(patch.dict('os.environ', {'GPU_TASK_COMPLETION_CONFIG': '/missing/secret'}))
                stack.enter_context(patch.object(sys, 'argv', argv))
                for name, value in {
                    'load_config': lambda _: config,
                    'TextEncoder': lambda *a, **kw: bundle,
                    'load_vae': lambda *a, **kw: None,
                    'encode_images_to_latents': encode_images,
                    'scale_latents': lambda x, c: x,
                    'encode_text_encoder': lambda enc, ids, mask: torch.ones(*ids.shape, 24),
                    'build_dit_model': lambda *a: model,
                    'checkpoint_config': lambda *a: dataclasses.asdict(small),
                }.items():
                    stack.enter_context(patch.object(trainer, name, value))
                no_optimizer = stack.enter_context(patch.object(trainer.optim_lib, 'Adam', side_effect=AssertionError))
                no_ema = stack.enter_context(patch.object(trainer.optim_lib, 'EMA', side_effect=AssertionError))
                no_notify = stack.enter_context(patch.object(trainer, 'notify_task_completion_with_retries', side_effect=AssertionError))
                trainer.main(extension=RegionCalibration())
                report = json.loads((workdir / 'calibration.json').read_text())
                self.assertEqual(report['all_batches']['batches'], 2)
                self.assertEqual(report['source_counts'], {'danqing': 4})
                self.assertEqual(report['unsampled_buckets'], [])
                self.assertEqual(report['unsampled_sources'], [])
                self.assertEqual(len(report['batch_measurements'][0]['timesteps']), 2)
                self.assertAlmostEqual(report['all_batches']['shares']['lambda_40']['aggregate'], .4)
                self.assertFalse((workdir / 'checkpoint.pt').exists())
                self.assertEqual(len((workdir / 'batches.jsonl').read_text().splitlines()), 2)
                for key, value in initial.state_dict().items():
                    torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                no_optimizer.assert_not_called()
                no_ema.assert_not_called()
                no_notify.assert_not_called()
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    trainer.main(extension=RegionCalibration())
                config.resume = str(root / 'initial.pt')
                with self.assertRaisesRegex(ValueError, 'not resume'):
                    RegionCalibration().configure(config, SimpleNamespace(
                        calibration_batches=2, workdir=str(root / 'other'), resume=None))


if __name__ == '__main__':
    unittest.main()
