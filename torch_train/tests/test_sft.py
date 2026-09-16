import dataclasses
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "torch_train"))
from models.dit import DualStreamDiTConfig, i1DiT
from datasets.bucketed import BucketedImages, BucketBatchSampler, build_bucket_iterator
from datasets.captions import tokenize_captions
from training.checkpoint import load_model_weights, load_train_states, resolve_resume_path, save_checkpoint
from training.parallel import DistInfo, load_tp_batch
from training.optim import Adam, EMA
from diffusion.rectified_flow import RectifiedFlowConfig, prepare_rectified_flow_inputs

spec = importlib.util.spec_from_file_location("i1_inference", ROOT / "torch_inference/generate.py")
inference = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = inference
spec.loader.exec_module(inference)


def tiny_config(**kwargs):
    config = dict(input_size=16, image_resolution=128, in_channels=32, hidden_size=64,
                  depth=3, num_heads=4, text_embed_dim=24, text_num_tokens=8, drop_text_prob=0.0)
    config.update(kwargs)
    return DualStreamDiTConfig(**config)


class FakeTokenizer:
    def __call__(self, captions, padding=False, truncation=False, add_special_tokens=True,
                 max_length=None, **kwargs):
        ids = [[1] * (len(c) + 1) for c in captions]
        if padding == "max_length":
            mask = [[1] * min(len(x), max_length) + [0] * max(0, max_length - len(x)) for x in ids]
            ids = [x[:max_length] + [0] * max(0, max_length - len(x)) for x in ids]
            return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}
        return {"input_ids": ids}


def tp_worker(rank, rendezvous):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    info = DistInfo(rank=rank, world_size=2, model_size=2, tp_group=dist.group.WORLD, device=torch.device("cpu"))
    specs = [("image", (2, 16, 16, 3), torch.float32), ("attention_mask", (2, 8), torch.long)]
    batches = [{"image": torch.full((2, h, w, 3), float(h)), "attention_mask": torch.ones(2, 8, dtype=torch.long)}
               for h, w in [(16, 16), (16, 32), (32, 16)]]
    iterator = iter(batches) if rank == 0 else None
    for expected in batches:
        result = load_tp_batch(iterator, info, specs, torch.device("cpu"))
        for key in expected:
            torch.testing.assert_close(result[key], expected[key])
    dist.destroy_process_group()


class GeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_rectangular_and_larger_forward_backward(self):
        model = i1DiT(tiny_config())
        keys = set(model.state_dict())
        caption, mask = torch.randn(1, 8, 24), torch.ones(1, 8, dtype=torch.bool)
        for h, w in [(16, 16), (8, 32), (32, 8), (32, 32)]:
            x = torch.randn(1, 32, h, w)
            output = model(x, torch.tensor([0.5]), caption, mask, train=True)
            self.assertEqual(output.shape, x.shape)
            output.square().mean().backward()
            self.assertTrue(torch.isfinite(model.x_embedder.proj.weight.grad).all())
            model.zero_grad(set_to_none=True)
        self.assertEqual(set(model.state_dict()), keys)
        with self.assertRaises(ValueError):
            model(torch.randn(1, 32, 15, 16), torch.ones(1), caption, mask)

    def test_dynamic_square_positions_match_existing_resolution_initialization(self):
        small = i1DiT(tiny_config())
        large = i1DiT(tiny_config(input_size=32, image_resolution=256))
        pos, rows, cols, cos, sin = small.grid_geometry(16, 16, torch.device("cpu"))
        torch.testing.assert_close(pos, large.pos_embed, rtol=0, atol=0)
        torch.testing.assert_close(rows, large.image_row_ids)
        torch.testing.assert_close(cols, large.image_col_ids)
        for a, b in zip(cos, large.rope_embedder.cos_tables):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for a, b in zip(sin, large.rope_embedder.sin_tables):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_training_inference_parity_and_cache_shape(self):
        cfg = tiny_config()
        train = i1DiT(cfg).eval()
        infer = inference.i1DiT(**dataclasses.asdict(cfg)).eval()
        infer.load_state_dict(train.state_dict(), strict=True)
        caption = torch.randn(1, 8, 24)
        mask = torch.tensor([[True] * 5 + [False] * 3])
        for shape in [(16, 16), (8, 32), (32, 8), (32, 32)]:
            x = torch.randn(1, 32, *shape)
            with torch.no_grad():
                torch.testing.assert_close(train(x, torch.ones(1), caption, mask),
                                           infer(x, torch.ones(1), caption, mask), rtol=1e-5, atol=1e-5)
        cache = infer.prepare_forward_cache(caption, mask, 64, (4, 16))
        with self.assertRaises(ValueError):
            infer(torch.randn(1, 32, 32, 8), torch.ones(1), caption, mask, cache)

    def test_long_caption_init_cfg_and_backward(self):
        old = i1DiT(tiny_config())
        long = i1DiT(tiny_config(text_num_tokens=1024, drop_text_prob=1.0))
        load_model_weights(long, {"model": old.state_dict()}, DistInfo(), initialize=True)
        torch.testing.assert_close(long.text_encoder_adapter.learnable_null_caption[:, :8],
                                   old.text_encoder_adapter.learnable_null_caption)
        x = torch.randn(1, 32, 8, 12)
        caption, mask = torch.randn(1, 1024, 24), torch.ones(1, 1024, dtype=torch.bool)
        out = long(x, torch.ones(1), caption, mask, train=True)
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(long.text_encoder_adapter.learnable_null_caption.grad).all())
        self.assertGreater(long.text_encoder_adapter.learnable_null_caption.grad.abs().sum().item(), 0)
        infer = inference.i1DiT(**dataclasses.asdict(long.config))
        infer.load_state_dict(long.state_dict())
        cfg_text, cfg_mask = inference.prepare_cfg_conditioning(infer, caption, mask)
        self.assertEqual(cfg_text.shape, (2, 1024, 24))
        with torch.no_grad():
            result = infer(x.repeat(2, 1, 1, 1), torch.ones(2), cfg_text, cfg_mask)
        self.assertEqual(result.shape, (2, 32, 8, 12))

    def test_compiled_blocks_with_activation_checkpointing(self):
        model = i1DiT(tiny_config(use_grad_ckpt=True))
        for block in [*model.in_blocks, model.mid_block, *model.out_blocks]:
            block.compile(backend="aot_eager", dynamic=True)
        for h, w in [(8, 12), (12, 16)]:
            output = model(torch.randn(1, 32, h, w), torch.ones(1), torch.randn(1, 8, 24),
                           torch.ones(1, 8, dtype=torch.bool), train=True)
            output.square().mean().backward()
            self.assertTrue(torch.isfinite(model.x_embedder.proj.weight.grad).all())
            model.zero_grad(set_to_none=True)

    def test_training_checkpoint_resume_and_ema_initialization(self):
        cfg = tiny_config()
        model = i1DiT(cfg)
        model.pos_embed.requires_grad_(False)
        optimizer, ema = Adam(model.named_parameters(), lr=1e-4), EMA(model, decay_rate=0.9)
        for h, w in [(8, 12), (12, 8)]:
            xt, target, t = prepare_rectified_flow_inputs(torch.randn(1, 32, h, w), RectifiedFlowConfig())
            prediction = model(xt, t, torch.randn(1, 8, 24), torch.ones(1, 8, dtype=torch.bool), train=True)
            (prediction - target).square().mean().backward()
            optimizer.step()
            model.zero_grad(set_to_none=True)
            ema.update(model)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "checkpoint.pt")
            save_checkpoint(path, model, ema, optimizer, 2, dataclasses.asdict(cfg), DistInfo())
            saved = torch.load(path, weights_only=True)
            resumed = i1DiT(cfg)
            resumed.pos_embed.requires_grad_(False)
            load_model_weights(resumed, saved, DistInfo())
            opt2, ema2 = Adam(resumed.named_parameters(), lr=1e-4), EMA(resumed)
            self.assertEqual(load_train_states(ema2, opt2, saved, DistInfo()), 2)
            self.assertEqual(opt2.count, optimizer.count)
            for name in optimizer.mu:
                torch.testing.assert_close(opt2.mu[name], optimizer.mu[name])
            fresh = i1DiT(cfg)
            load_model_weights(fresh, saved, DistInfo(), initialize=True)
            torch.testing.assert_close(fresh.x_embedder.proj.weight, ema.shadow["x_embedder.proj.weight"])
            loaded_inference = inference.build_model(torch.device("cpu"), path, dtype=torch.float32)
            x, caption = torch.randn(1, 32, 8, 12), torch.randn(1, 8, 24)
            with torch.no_grad():
                torch.testing.assert_close(fresh(x, torch.ones(1), caption),
                                           loaded_inference(x, torch.ones(1), caption), rtol=1e-5, atol=1e-5)


class DataTests(unittest.TestCase):
    def test_rank_shapes_disjoint_samples_and_resume(self):
        groups = [list(range(100)), list(range(100, 200)), []]
        samplers = [list(BucketBatchSampler(groups, 8, rank, 2, 12, seed=4)) for rank in range(2)]
        for a, b in zip(*samplers):
            self.assertEqual(a[0] // 100, b[0] // 100)
            self.assertEqual(len(set(a + b)), 8)
        resumed = list(BucketBatchSampler(groups, 8, 0, 2, 12, start_step=5, seed=4))
        self.assertEqual(resumed, samplers[0][5:])

    def test_image_preservation_and_iterator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = Image.new("RGB", (96, 32), (255, 0, 0))
            image.save(root / "image.png")
            (root / "train.jsonl").write_text(json.dumps(dict(image_path="image.png", caption="abc", width=96, height=32)) + "\n")
            config = dict(manifest=str(root / "train.jsonl"), buckets=[(32, 64)], batch_size=2, num_workers=0)
            dataset = BucketedImages(config)
            pixels, _ = dataset[0]
            self.assertEqual(pixels.shape, (32, 64, 3))
            self.assertTrue(torch.equal(pixels[0], torch.ones(64, 3)))
            self.assertTrue(torch.equal(pixels[16, 0], torch.tensor([1., -1., -1.])))
            batch = next(build_bucket_iterator(dataset, config, FakeTokenizer(), 8, DistInfo(device=torch.device("cpu")), 1, 0, 0))
            self.assertEqual(batch["image"].shape, (2, 32, 64, 3))
            self.assertEqual(batch["attention_mask"].sum().item(), 8)

    def test_caption_overflow_and_checkpoint_errors(self):
        with self.assertRaisesRegex(ValueError, "exceeding"):
            tokenize_captions(FakeTokenizer(), ["a" * 10], 8)
        tokens = tokenize_captions(FakeTokenizer(), ["a" * 10], 8, "truncate")
        self.assertEqual(tokens["input_ids"].shape, (1, 8))
        with self.assertRaisesRegex(ValueError, "--init_from"):
            load_train_states(None, None, {"model": {}}, DistInfo())
        with self.assertRaises(FileNotFoundError):
            resolve_resume_path(None, {"resume": "/nonexistent-i1-checkpoint"})

    def test_tensor_parallel_variable_shape_broadcast(self):
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(tp_worker, args=(str(Path(directory) / "rendezvous"),), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
