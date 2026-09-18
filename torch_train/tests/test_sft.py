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
from datasets.bucketed import BucketedImages, BucketBatchSampler, bucket_steps_per_epoch, build_bucket_iterator
from datasets.data_sources import iter_image_records, open_record_image
from datasets.validate_images import _validate_shard
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

    def test_dynamic_inference_context_preserves_native_output(self):
        config = dataclasses.asdict(tiny_config())
        source = inference.i1DiT(**config).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "checkpoint.pt")
            torch.save({"config": config, "model": source.state_dict()}, path)
            native = inference.build_model(torch.device("cpu"), path, dtype=torch.float32)
            extended = inference.build_model(
                torch.device("cpu"), path, dtype=torch.float32, text_num_tokens=16
            )

        tokenizer = FakeTokenizer()
        self.assertEqual(
            inference.select_text_context_length(tokenizer, ["short"], extended, True), 8
        )
        self.assertEqual(
            inference.select_text_context_length(tokenizer, ["long enough"], extended, True), 16
        )
        self.assertEqual(
            inference.select_text_context_length(tokenizer, ["short"], extended, False), 16
        )

        x = torch.randn(2, 32, 8, 8)
        t = torch.ones(2)
        caption = torch.randn(1, 8, 24)
        mask = torch.tensor([[True] * 6 + [False] * 2])
        native_cfg = inference.prepare_cfg_conditioning(native, caption, mask)
        extended_cfg = inference.prepare_cfg_conditioning(extended, caption, mask)
        with torch.no_grad():
            native_output = native(x, t, *native_cfg)
            extended_output = extended(x, t, *extended_cfg)
        torch.testing.assert_close(native_output, extended_output, rtol=0, atol=0)

        long_caption = torch.randn(1, 16, 24)
        long_mask = torch.ones(1, 16, dtype=torch.bool)
        long_cfg = inference.prepare_cfg_conditioning(extended, long_caption, long_mask)
        with torch.no_grad():
            self.assertEqual(extended(x, t, *long_cfg).shape, x.shape)

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
    def test_1024_sft_defaults_to_one_epoch(self):
        from configs.sft_1024 import get_config
        config = get_config()
        self.assertEqual(config.num_epochs, 1)
        self.assertIsNone(config.total_steps)

    def test_gpt_image_parquet_source(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is not installed")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            from io import BytesIO
            buffer = BytesIO()
            Image.new("RGB", (96, 64), (12, 34, 56)).save(buffer, format="PNG")
            table = pa.table(dict(id=["sample"], prompt=["caption"], size=["96x64"],
                                  image_bytes=[buffer.getvalue()]))
            path = root / "sample.parquet"
            pq.write_table(table, path, row_group_size=1)
            records = list(iter_image_records(path))
            self.assertEqual((records[0].width, records[0].height), (96, 64))
            self.assertEqual(open_record_image(records[0]).size, (96, 64))
            validation = _validate_shard(path, 10)
            self.assertEqual((validation["checked"], validation["decoded"],
                              validation["decode_error_count"],
                              validation["dimension_mismatch_count"]), (1, 1, 0, 0))
            for shard in range(2, 6):
                pq.write_table(table, root / f"sample_{shard}.parquet", row_group_size=1)
            self.assertEqual(len(list(iter_image_records(root))), 5)
            config = dict(manifest=str(path), buckets=[(32, 48)], batch_size=1, num_workers=0)
            dataset = BucketedImages(config)
            pixels, caption = dataset[0]
            self.assertEqual((pixels.shape, caption), (torch.Size([32, 48, 3]), "caption"))

    def test_rank_shapes_disjoint_samples_and_resume(self):
        groups = [list(range(100)), list(range(100, 200)), []]
        samplers = [list(BucketBatchSampler(groups, 8, rank, 2, 12, seed=4)) for rank in range(2)]
        for a, b in zip(*samplers):
            self.assertEqual(a[0] // 100, b[0] // 100)
            self.assertEqual(len(set(a + b)), 8)
        resumed = list(BucketBatchSampler(groups, 8, 0, 2, 12, start_step=5, seed=4))
        self.assertEqual(resumed, samplers[0][5:])

    def test_epoch_sampler_exhausts_buckets_and_pads_only_tails(self):
        groups = [list(range(10)), list(range(100, 105)), []]
        sampler = BucketBatchSampler(groups, 4, 0, 1, 10, seed=7)
        batches = list(sampler)
        self.assertEqual(bucket_steps_per_epoch(groups, 4), 5)
        self.assertEqual(sampler.steps_per_epoch, 5)
        for epoch in range(2):
            epoch_batches = batches[epoch * 5:(epoch + 1) * 5]
            self.assertTrue(all(all(index < 100 for index in batch) or
                                all(index >= 100 for index in batch)
                                for batch in epoch_batches))
            seen = {index for batch in epoch_batches for index in batch}
            self.assertTrue(set(range(10)).issubset(seen))
            self.assertTrue(set(range(100, 105)).issubset(seen))
            self.assertEqual(sum(map(len, epoch_batches)), 20)

        resumed = list(BucketBatchSampler(groups, 4, 0, 1, 10, start_step=5, seed=7))
        self.assertEqual(resumed, batches[5:])

    def test_distributed_sampler_small_buckets_and_every_resume_offset(self):
        groups = [[], list(range(3)), list(range(10, 18)),
                  list(range(20, 29)), list(range(40, 57))]
        steps_per_epoch = bucket_steps_per_epoch(groups, 8)
        total_steps = steps_per_epoch * 3
        for seed in (0, 7, 42):
            full = list(BucketBatchSampler(groups, 8, 0, 1, total_steps, seed=seed))
            for epoch in range(3):
                batches = full[epoch * steps_per_epoch:(epoch + 1) * steps_per_epoch]
                for group in groups[1:]:
                    matching = [batch for batch in batches if batch[0] in group]
                    self.assertEqual(len(matching), (len(group) + 7) // 8)
                    self.assertEqual({item for batch in matching for item in batch}, set(group))
                    for batch in matching:
                        self.assertEqual(len(batch), 8)
                        self.assertTrue(set(batch).issubset(group))
                        if len(group) >= 8:
                            self.assertEqual(len(set(batch)), 8)
            for world in (1, 2, 4, 8):
                ranks = [list(BucketBatchSampler(groups, 8, rank, world, total_steps, seed=seed))
                         for rank in range(world)]
                combined = [[item for rank in ranks for item in rank[step]] for step in range(total_steps)]
                self.assertEqual(combined, full)
                for rank in range(world):
                    for start in range(total_steps + 1):
                        resumed = BucketBatchSampler(groups, 8, rank, world, total_steps,
                                                     start_step=start, seed=seed)
                        self.assertEqual(len(resumed), total_steps - start)
                        self.assertEqual(list(resumed), ranks[rank][start:])

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
