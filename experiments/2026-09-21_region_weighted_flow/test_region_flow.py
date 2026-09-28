"""Offline CPU integration checks; no pretrained downloads or real shutdown calls."""
from contextlib import ExitStack
from io import BytesIO
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

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from config import get_config
from datasets.precompute_captioned import precompute_captioned
from models.dit import DualStreamDiTConfig, i1DiT
from precompute import build_regions
from region_data import RegionImages, build_iterator, mapped_masks
from region_loss import region_flow_loss
from region_masks import make_mask
from runtime import RegionFlow
from training import main as trainer
from training.parallel import DistInfo, load_tp_batch


def fixture(root, workers=1):
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1, "chart": 2}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]")
    tokenizer.save_pretrained(root / "tokenizer")
    image = BytesIO()
    Image.new("RGB", (80, 64), (40, 80, 120)).save(image, format="PNG")
    ocr = json.dumps([dict(text="Title", score=.99, box=[8, 8, 40, 24])])
    rows = []
    for uid, caption, raw, width in [
        ("text", "chart title", ocr, 80),
        ("missing", "chart axes", None, 80),
        ("mismatch", "chart legend", ocr, 81),
        ("photograph", "A PHOTOGRAPH, with text", ocr, 80),
        ("tenth", "one two three four five six seven eight nine photograph", ocr, 80),
        ("eleventh", "one two three four five six seven eight nine ten photograph", ocr, 80),
    ]:
        rows.append(dict(id=uid, caption=caption, ocr_raw_output=raw, source_width=width,
                         source_height=64, source_dataset="danqing", image_bytes=image.getvalue(),
                         declared_width=80, declared_height=64, caption_status="ok", image_decode_error=None))
    source = root / "captioned/data/danqing"
    source.mkdir(parents=True)
    for start in range(0, len(rows), 2):
        pq.write_table(pa.Table.from_pylist(rows[start:start+2]),
                       source / f"part-{start//2:05d}.parquet", row_group_size=2)
    config = dict(type="bucketed", buckets=[(64, 80)], min_image_area=0, min_image_side=0,
                  allow_upscale=False, resize_mode="pad", image_root="", caption_overflow="error",
                  batch_size=2, num_workers=0)
    cache = root / "cache"
    precompute_captioned(root / "captioned", cache / "pixels", config, workers=workers,
                         token_limit=16, tokenizer_name=str(root / "tokenizer"), records_per_part=2,
                         max_shard_bytes=1024**2)
    build_regions(cache, config, workers=workers)
    config["manifest"] = str(cache / "cache.jsonl")
    return config, tokenizer


def tp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist = torch.distributed
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    info = DistInfo(rank=rank, world_size=2, model_size=2, tp_group=dist.group.WORLD,
                    device=torch.device("cpu"))
    specs = [("region_mask", (2, 8, 8), torch.float32)]
    expected = [dict(region_mask=torch.arange(2*h*w).reshape(2, h, w).float())
                for h, w in [(8, 8), (8, 10), (10, 8)]]
    iterator = iter(expected) if rank == 0 else None
    for batch in expected:
        result = load_tp_batch(iterator, info, specs, info.device)
        torch.testing.assert_close(result["region_mask"], batch["region_mask"])
    dist.destroy_process_group()


class RegionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def tearDown(self):
        mapped_masks.cache_clear()

    def test_defaults_match_recorded_1024_run(self):
        with patch.dict("os.environ", {}, clear=True):
            config = get_config()
        self.assertEqual((config.region_weight, config.perceptual_weight), (0., 0.))
        self.assertEqual((config.image_size, config.token_len, len(config.input.buckets)), (1024, 1024, 45))
        self.assertEqual((config.input.batch_size, config.fsdp_axis_size,
                          config.tensor_parallel_size, config.grad_accum_steps), (32, 8, 1, 1))
        self.assertEqual((config.num_epochs, config.total_steps), (1, None))
        self.assertEqual((config.lr, config.ema_decay_rate), (1e-5, .9995))
        self.assertEqual((config.log_training_steps, config.ckpt_steps, config.keep_ckpt_steps), (50, 1000, 2500))
        self.assertEqual(config.transport.train_timestep_shift, .3)
        self.assertTrue(config.amp and config.use_grad_ckpt)
        self.assertFalse(config.compile)

    def test_masks_use_height_weights_max_overlap_and_exact_fit_pad(self):
        region = dict(text="x", score=.99, box=[0, 0, 80, 40])
        mask, _ = make_mask([region, region], 80, 40, (64, 80), reference_height=40)
        expected = np.zeros((64, 80), dtype=np.float32)
        expected[12:52] = 1  # Letterbox top/bottom are excluded, including polygon edges.
        np.testing.assert_array_equal(mask, expected.reshape(8, 8, 10, 8).mean((1, 3)))
        single, _ = make_mask([region], 80, 40, (64, 80), reference_height=40)
        np.testing.assert_array_equal(mask, single)
        columns = dict(res=dict(rec_texts=["x"], rec_scores=[.99], rec_boxes=[[0, 0, 80, 40]]))
        np.testing.assert_array_equal(
            mask, make_mask(json.dumps(columns), 80, 40, (64, 80), reference_height=40)[0])

        short = dict(text="short", score=.99, box=[0, 0, 32, 8])
        tall = dict(text="tall", score=.99, box=[48, 0, 80, 32])
        weighted, stats = make_mask([short, tall], 80, 64, (64, 80), reference_height=8)
        short_mask = make_mask([short], 80, 64, (64, 80), reference_height=100)[0]
        tall_mask = make_mask([tall], 80, 64, (64, 80), reference_height=100)[0]
        np.testing.assert_array_equal(weighted, np.maximum(short_mask, tall_mask * .25))
        self.assertEqual((stats["regions_used"], stats["regions_downweighted"]), (2, 1))
        self.assertEqual(stats["region_weight_sum"], 1.25)
        for raw in [None, "broken", {"res": None}, {"rec_texts": ["x"], "rec_scores": None},
                    [dict(region, score=.1)], [dict(region, box=[-1, 0, 80, 40])],
                    [dict(region, poly=[[0, 0], [1, 1], [2, 2]])]]:
            self.assertFalse(make_mask(raw, 80, 40, (64, 80))[0].any())

    def test_loss_gradient_and_microbatch_equivalence(self):
        prediction = torch.tensor([[[[1., 2.], [3., 4.]]], [[[2., 3.], [4., 5.]]]], requires_grad=True)
        mask = torch.tensor([[[1., 0.], [0., 0.]], [[0., 0.], [0., 0.]]])
        target = torch.zeros_like(prediction)
        loss, metrics = region_flow_loss(prediction, target, mask, weight=2)
        torch.testing.assert_close(loss, prediction.square().mean() + .25)
        gradient, = torch.autograd.grad(loss, prediction)
        expected = 2*prediction.detach()/8
        expected[0, 0, 0, 0] += .5
        torch.testing.assert_close(gradient, expected)
        self.assertEqual(metrics["region_image_fraction"].item(), .5)
        micro = sum(region_flow_loss(prediction[i:i+1], target[i:i+1], mask[i:i+1], 2)[0]/2
                    for i in range(2))
        torch.testing.assert_close(torch.autograd.grad(micro, prediction)[0], gradient)
        for weight, m in [(0, mask), (2, torch.zeros_like(mask))]:
            torch.testing.assert_close(region_flow_loss(prediction, target, m, weight)[0],
                                       torch.nn.functional.mse_loss(prediction, target))
        small = torch.full_like(mask, .01)
        torch.testing.assert_close(region_flow_loss(prediction, target, small)[1]["region_flow_loss"],
                                   prediction.square().mean() * .0001)

    def test_loss_squares_soft_region_weights(self):
        # Compare our NCHW interface against a packed token/channel reference,
        # with multiple channels, rectangular geometry, and nonzero targets.
        prediction = torch.arange(48, dtype=torch.float32).reshape(2, 3, 2, 4) / 10
        prediction.requires_grad_()
        target = prediction.detach().flip(-1) + .3
        mask = torch.tensor([[[1., .5, 0., .25], [0., 1., .75, 0.]],
                             [[0., 0., 0., 0.], [0., 0., 0., 0.]]])
        pred_tokens = prediction.flatten(2).transpose(1, 2)
        target_tokens = target.flatten(2).transpose(1, 2)
        token_mask = mask.flatten(1)[:, :, None]
        reference_region = ((pred_tokens - target_tokens).square() * token_mask.square()).mean()
        reference = torch.nn.functional.mse_loss(pred_tokens, target_tokens) + reference_region
        actual, metrics = region_flow_loss(prediction, target, mask)
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(metrics["region_flow_loss"], reference_region)
        torch.testing.assert_close(torch.autograd.grad(actual, prediction)[0],
                                   torch.autograd.grad(reference, prediction)[0])

    def test_cpu_pipeline_resume_checksums_and_batches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, tokenizer = fixture(root, workers=2)
            dataset = RegionImages(config)
            summary = dataset.summary
            self.assertEqual((summary["count"], summary["masked_images"], summary["rejected"]), (6, 4, 0))
            self.assertEqual(summary["rejected_counts"], {})
            self.assertIsNone(summary["settings"]["caption_filter"])
            self.assertEqual(summary["mask_stats"]["ocr_dimension_mismatch_images"], 1)
            by_id = {record[0].identifier: i for i, record in enumerate(dataset.records)}
            self.assertGreater(dataset[by_id["text"]][2].sum(), 0)
            self.assertEqual(dataset[by_id["mismatch"]][2].sum(), 0)
            for identifier in ("photograph", "tenth", "eleventh"):
                self.assertIn(identifier, by_id)
                self.assertGreater(dataset[by_id[identifier]][2].sum(), 0)
            iterator = build_iterator(dataset, config, tokenizer, 16, DistInfo(device=torch.device("cpu")), 2, 0, 0)
            batch = next(iterator)
            self.assertEqual(batch["image"].shape, (2, 64, 80, 3))
            self.assertEqual(batch["region_mask"].shape, (2, 8, 10))
            self.assertEqual(batch["input_ids"].shape, (2, 16))
            rebuilt = build_regions(root / "cache", config, workers=1)
            self.assertEqual(rebuilt, summary)
            with self.assertRaisesRegex(ValueError, "settings changed"):
                build_regions(root / "cache", config, workers=1, min_confidence=.9)
            payload = root / "cache" / summary["mask_files"][0]["path"]
            with payload.open("r+b") as handle:
                handle.write(b"\xff\xff")
            with self.assertRaisesRegex(ValueError, "checksum"):
                RegionImages(config)
            with self.assertRaisesRegex(ValueError, "checksum"):
                build_regions(root / "cache", config, workers=1)
            self.assertFalse((root / "cache/summary.json").exists())

    def test_tp_broadcast_rectangular_masks(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(tp_worker, args=(str(Path(directory)/"rendezvous"),), nprocs=2, join=True)

    def test_actual_training_loop_resume_and_success_only_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_config, tokenizer = fixture(root)
            config = get_config()
            config.region_weight = 1.
            config.perceptual_weight = 0  # This integration covers the no-perceptual control.
            config.input.update(data_config)
            config.image_size = 64
            config.token_len = 16
            config.fsdp_axis_size = 1
            config.wandb.log_wandb = False
            config.amp = False
            config.log_training_steps = 1
            config.use_grad_ckpt = False
            config.mu_dtype = "float32"
            small = DualStreamDiTConfig(input_size=8, image_resolution=64, in_channels=32,
                                       hidden_size=64, depth=3, num_heads=4, text_embed_dim=24,
                                       text_num_tokens=16, drop_text_prob=0.)
            initial = i1DiT(small)
            initial.init_weights()
            torch.save({"model": initial.state_dict()}, root / "initial.pt")
            credentials = root / "completion.json"
            credentials.write_text(json.dumps(dict(name="test", password="dummy", vmids=["test-only"])))
            credentials.chmod(0o600)
            workdir = root / "train"
            bundle = SimpleNamespace(tokenizer=tokenizer, text_encoder=None, text_token_len=16, hidden_dim=24)

            def encode_images(vae, images):
                rgb = torch.nn.functional.interpolate(images.permute(0, 3, 1, 2), scale_factor=1/8)
                return rgb.repeat(1, 11, 1, 1)[:, :32]

            def finished(*args, **kwargs):
                self.assertEqual(args[0], dict(name="test", password="dummy", vmids=["current-training-vm-0"]))
                checkpoint = torch.load(workdir / "checkpoint.pt", weights_only=False)
                self.assertGreaterEqual(checkpoint["train_state"]["step"], 2)
                return 200

            def run(steps, extension=None, extra=()):
                argv = ["train.py", "--config", "unused", "--manifest", data_config["manifest"],
                        "--init_from", str(root / "initial.pt"), "--workdir", str(workdir),
                        "--total_steps", str(steps), "--completion-config", str(credentials), *extra]
                with patch.object(sys, "argv", argv):
                    trainer.main(extension=extension or RegionFlow())

            with ExitStack() as stack:
                stack.enter_context(patch("training.completion.socket.gethostname", return_value="current-training-vm-0"))
                for name, value in {
                    "load_config": lambda _: config,
                    "TextEncoder": lambda *a, **kw: bundle,
                    "load_vae": lambda *a, **kw: None,
                    "encode_images_to_latents": encode_images,
                    "scale_latents": lambda x, c: x,
                    "encode_text_encoder": lambda enc, ids, mask: torch.ones(*ids.shape, 24),
                    "build_dit_model": lambda *a: i1DiT(small),
                    "checkpoint_config": lambda *a: dataclasses.asdict(small),
                }.items():
                    stack.enter_context(patch.object(trainer, name, value))
                notify = stack.enter_context(patch.object(trainer, "notify_task_completion_with_retries", side_effect=finished))
                run(2)
                notify.assert_called_once()
                checkpoint = torch.load(workdir / "checkpoint.pt", weights_only=False)
                self.assertEqual(checkpoint["config"]["training_objective"]["weight"], 1.)
                objective = checkpoint["config"]["training_objective"]
                self.assertEqual(objective["name"], "region-weighted-flow-v5")
                self.assertEqual(objective["perceptual"], dict(weight=0.))
                old_objective = dict(objective, name="region-weighted-flow-v1", min_area=1.,
                                     normalization="per-image-union-area-then-all-image-mean")
                with self.assertRaisesRegex(ValueError, "Resume objective"):
                    RegionFlow.validate_resume(
                        {"config": {"training_objective": old_objective}}, objective)
                self.assertTrue(any(not torch.equal(initial.state_dict()[k], v)
                                    for k, v in checkpoint["model"].items()))
                run(3)
                self.assertEqual(notify.call_count, 2)
                config.region_weight = 2.
                with self.assertRaisesRegex(ValueError, "Resume objective"):
                    run(4)
                self.assertEqual(notify.call_count, 2)
                config.region_weight = 1.
                bad = RegionFlow()
                bad.loss = lambda *a, **kw: (torch.tensor(float("nan")), {})
                with self.assertRaises(FloatingPointError):
                    run(4, bad)
                self.assertEqual(notify.call_count, 2)
                with self.assertRaisesRegex(ValueError, "checkpoint saving"):
                    run(4, extra=("--no_save",))
                self.assertEqual(notify.call_count, 2)


if __name__ == "__main__":
    unittest.main()
