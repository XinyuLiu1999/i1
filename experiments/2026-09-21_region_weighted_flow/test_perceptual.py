"""Perceptual gradients, reference arithmetic, frozen state, and trainer integration."""
from contextlib import ExitStack
import dataclasses
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "torch_train")]

import torch
from torch import nn
import torch.nn.functional as F

from config import get_config
from perceptual_loss import RegionPerceptualLoss, reconstruct_clean_latents, load_odm, encoder_source
from region_loss import region_flow_loss
from runtime import RegionFlow
from test_region_flow import fixture
from models.dit import DualStreamDiTConfig, i1DiT
from training import main as trainer
from vae.vae import reverse_scale_latents


class TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(32, 3, 1, bias=False)
        nn.init.constant_(self.proj.weight, .03)

    def decode(self, latents, return_dict=False):
        return (F.interpolate(self.proj(latents), scale_factor=8, mode="nearest"),)


class TinyFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        nn.init.constant_(self.conv.weight, .1)
        nn.init.zeros_(self.conv.bias)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, image):
        feature = self.bn(self.conv(image))
        return [F.avg_pool2d(feature, scale) for scale in (4, 8, 16, 32)], None


class PerceptualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def inputs(self):
        generator = torch.Generator().manual_seed(31)
        clean = torch.randn(2, 32, 8, 10, generator=generator, requires_grad=True)
        noise = torch.randn(2, 32, 8, 10, generator=generator)
        target = (clean.detach() - noise).requires_grad_()
        prediction = (target.detach() + .2).requires_grad_()
        images = torch.randn(2, 64, 80, 3, generator=generator, requires_grad=True)
        mask = torch.zeros(2, 8, 10)
        mask[0, 1:5, 2:7] = .5  # second image is unmasked; keep it in the denominator
        return prediction, target, clean, images, mask

    def test_reconstruction_matches_fluxtext_sign_and_perfect_prediction(self):
        prediction, target, clean, images, mask = self.inputs()
        torch.testing.assert_close(reconstruct_clean_latents(target, target, clean), clean)
        noise = clean.detach() - target.detach()
        # Their velocity is noise-data, ours is data-noise.
        torch.testing.assert_close(reconstruct_clean_latents(prediction, target, clean),
                                   noise - (-prediction))

    def test_reference_loss_gradients_checkpointing_and_empty_masks(self):
        prediction, target, clean, images, mask = self.inputs()
        encoder, vae = TinyFeatures(), TinyVAE()
        module = RegionPerceptualLoss(encoder, vae, checkpointing=False)
        bn_before = encoder.bn.running_mean.clone()
        actual = module(prediction, target, clean, images, mask)
        decoded = vae.decode(reverse_scale_latents(clean.detach() + prediction - target.detach(), "flux2"))[0]
        pf = encoder(decoded)[0]
        with torch.no_grad():
            tf = encoder(images.detach().permute(0, 3, 1, 2))[0]
        reference = sum(((p - t).square()
                         * F.interpolate(mask[:, None], size=p.shape[-2:], mode="area").square()).mean()
                        for p, t in zip(pf, tf))
        torch.testing.assert_close(actual, reference)
        grad, = torch.autograd.grad(actual, prediction, retain_graph=True)
        torch.testing.assert_close(grad, torch.autograd.grad(reference, prediction)[0])
        self.assertGreater(grad[0].abs().sum().item(), 0)
        self.assertEqual(grad[1].abs().sum().item(), 0)
        actual.backward()
        self.assertIsNone(clean.grad)
        self.assertIsNone(target.grad)
        self.assertIsNone(images.grad)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in module.parameters()))
        torch.testing.assert_close(encoder.bn.running_mean, bn_before)

        module.checkpointing = True
        checkpointed = module(prediction, target, clean, images, mask)
        torch.testing.assert_close(checkpointed, actual)
        torch.testing.assert_close(torch.autograd.grad(checkpointed, prediction)[0], grad)
        # Full-batch averaging must agree with gradient accumulation.
        micros = sum(module(prediction[i:i+1], target[i:i+1], clean[i:i+1],
                            images[i:i+1], mask[i:i+1]) / 2 for i in range(2))
        torch.testing.assert_close(micros, actual)
        torch.testing.assert_close(torch.autograd.grad(micros, prediction)[0], grad)
        with patch.object(vae, "decode", side_effect=AssertionError("empty masks must skip decode")):
            empty = module(prediction, target, clean, images, torch.zeros_like(mask))
            self.assertEqual(empty.item(), 0)
            torch.testing.assert_close(torch.autograd.grad(empty, prediction)[0], torch.zeros_like(prediction))

    def test_strict_odm_checkpoint_loading(self):
        config = get_config()
        source = encoder_source(config.fluxtext_root)
        spec = importlib.util.spec_from_file_location("test_odm_reference", source)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        model = reference.ResNet([3, 4, 6, 3], 512, 32, input_resolution=512, width=64)
        state = {"module.visual." + k: v for k, v in model.state_dict().items()}
        with patch("perceptual_loss.torch.load", return_value={"state_dict": state}):
            loaded = load_odm("unused", config.fluxtext_root, "cpu")
        self.assertFalse(loaded.training)
        self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
        torch.testing.assert_close(loaded.conv1.weight, model.conv1.weight)
        del state["module.visual.conv1.weight"]
        with patch("perceptual_loss.torch.load", return_value={"state_dict": state}):
            with self.assertRaisesRegex(RuntimeError, "Missing key"):
                load_odm("unused", config.fluxtext_root, "cpu")

    def test_missing_checkpoint_fails_and_disabled_term_needs_no_dependencies(self):
        config = get_config()
        config.perceptual_checkpoint = "/does-not-exist/odm.pt"
        config.perceptual_weight = 20
        with self.assertRaisesRegex(FileNotFoundError, "Missing perceptual dependency"):
            RegionFlow().configure(config, None)
        config.perceptual_weight = 0
        flow = RegionFlow()
        flow.configure(config, None)
        flow.setup(config, None, "cpu")
        prediction, target, clean, images, mask = self.inputs()
        actual, metrics = flow.loss(prediction, target, {"region_mask": mask}, slice(None),
                                    latents=clean, images=images)
        torch.testing.assert_close(actual, region_flow_loss(prediction, target, mask, flow.weight)[0])
        self.assertEqual(metrics["perceptual_loss"].item(), 0)

    def test_actual_training_with_perceptual_and_resume_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_config, tokenizer = fixture(root)
            config = get_config()
            config.input.update(data_config)
            config.image_size, config.token_len, config.fsdp_axis_size = 64, 16, 1
            config.wandb.log_wandb = False
            config.amp = config.use_grad_ckpt = False
            config.mu_dtype = "float32"
            config.log_training_steps = 1
            config.perceptual_weight = 20
            config.perceptual_checkpoint = str(root / "odm.pt")
            Path(config.perceptual_checkpoint).write_bytes(b"stub checkpoint for mocked loader")
            small = DualStreamDiTConfig(input_size=8, image_resolution=64, in_channels=32,
                                       hidden_size=64, depth=3, num_heads=4, text_embed_dim=24,
                                       text_num_tokens=16, drop_text_prob=0.)
            initial = i1DiT(small)
            initial.init_weights()
            torch.save({"model": initial.state_dict()}, root / "initial.pt")
            encoder, vae = TinyFeatures(), TinyVAE()
            bundle = SimpleNamespace(tokenizer=tokenizer, text_encoder=None, text_token_len=16, hidden_dim=24)

            def encode_images(vae, images):
                rgb = F.interpolate(images.permute(0, 3, 1, 2), scale_factor=1/8)
                return rgb.repeat(1, 11, 1, 1)[:, :32]

            def run(steps):
                flow = RegionFlow()
                argv = ["train.py", "--config", "unused", "--manifest", data_config["manifest"],
                        "--init_from", str(root / "initial.pt"), "--workdir", str(root / "train"),
                        "--total_steps", str(steps), "--completion-config", ""]
                with patch.object(sys, "argv", argv):
                    trainer.main(extension=flow)
                return flow

            with ExitStack() as stack:
                for name, value in {
                    "load_config": lambda _: config, "TextEncoder": lambda *a, **kw: bundle,
                    "load_vae": lambda *a, **kw: vae, "encode_images_to_latents": encode_images,
                    "scale_latents": lambda x, c: x,
                    "encode_text_encoder": lambda enc, ids, mask: torch.ones(*ids.shape, 24),
                    "build_dit_model": lambda *a: i1DiT(small),
                    "checkpoint_config": lambda *a: dataclasses.asdict(small),
                }.items():
                    stack.enter_context(patch.object(trainer, name, value))
                stack.enter_context(patch("runtime.load_odm", return_value=encoder))
                flow = run(2)
                self.assertIsNotNone(flow.perceptual)
                saved = torch.load(root / "train/checkpoint.pt", weights_only=False)
                self.assertEqual(saved["config"]["training_objective"]["perceptual"]["weight"], 20)
                self.assertTrue(any(not torch.equal(initial.state_dict()[k], v)
                                    for k, v in saved["model"].items()))
                self.assertTrue(all(p.grad is None for p in vae.parameters()))
                self.assertTrue(all(p.grad is None for p in encoder.parameters()))
                run(3)
                config.perceptual_weight = 1
                with self.assertRaisesRegex(ValueError, "Resume objective"):
                    run(4)
                config.perceptual_weight = 20
                Path(config.perceptual_checkpoint).write_bytes(b"different weights")
                with self.assertRaisesRegex(ValueError, "Resume objective"):
                    run(4)


if __name__ == "__main__":
    unittest.main()
