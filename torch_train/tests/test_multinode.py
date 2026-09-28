"""Launcher validation and a real two-agent CPU rendezvous regression test."""
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "run_multinode.sh"


def launch_env(**overrides):
    env = os.environ.copy()
    for key in ("NNODES", "NODE_RANK", "RANK", "WORLD_SIZE", "LOCAL_RANK",
                "GPUS_PER_NODE", "MASTER_ADDR", "MASTER_PORT", "FSDP_SIZE", "TP_SIZE",
                "GLOBAL_BATCH_SIZE", "GRAD_ACCUM", "SFT_INIT", "SFT_RESUME",
                "SFT_CONFIG", "SFT_MANIFEST", "SFT_WORKDIR", "CHECK_BACKEND"):
        env.pop(key, None)
    env.update(TRAIN_PYTHON=sys.executable, WORLD_SIZE="2", RANK="1",
               MASTER_ADDR="127.0.0.1", GPUS_PER_NODE="4",
               SFT_INIT="/shared/model.pt", SFT_MANIFEST="/shared/data/cache.jsonl",
               SFT_WORKDIR="/shared/output")
    env.update(overrides)
    return env


class LauncherTests(unittest.TestCase):
    def run_dry(self, *args, **env):
        return subprocess.run(["bash", str(LAUNCHER), "--dry-run", *args],
                              env=launch_env(**env), capture_output=True, text=True, timeout=10)

    def command(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)
        line = next(line for line in result.stdout.splitlines() if line.startswith("Command:"))
        return shlex.split(line.removeprefix("Command:"))

    def test_platform_node_ranks_and_global_sharding(self):
        command = self.command(self.run_dry("--total_steps", "3"))
        for flag, expected in (("--nnodes", "2"), ("--node_rank", "1"),
                               ("--nproc_per_node", "4"), ("--fsdp", "8"),
                               ("--tp", "1"), ("--completion-config", ""),
                               ("--total_steps", "3")):
            self.assertEqual(command[command.index(flag) + 1], expected)
        self.assertNotIn("--standalone", command)
        self.assertIn("--no_compile", command)

    def test_one_gpu_per_node_and_paths_with_spaces(self):
        result = self.run_dry(WORLD_SIZE="4", RANK="3", GPUS_PER_NODE="1",
                              SFT_WORKDIR="/shared/output with spaces")
        command = self.command(result)
        self.assertEqual(command[command.index("--fsdp") + 1], "4")
        self.assertEqual(command[command.index("--workdir") + 1], "/shared/output with spaces")
        self.assertIn("microbatch_per_gpu=2", result.stdout)

    def test_hybrid_sharding_and_explicit_node_overrides(self):
        command = self.command(self.run_dry(NNODES="2", NODE_RANK="0", WORLD_SIZE="99",
                                            RANK="98", FSDP_SIZE="2", TP_SIZE="2"))
        self.assertEqual(command[command.index("--nnodes") + 1], "2")
        self.assertEqual(command[command.index("--node_rank") + 1], "0")
        self.assertEqual(command[command.index("--fsdp") + 1], "2")

    def test_invalid_topology_and_batch_fail_before_launch(self):
        for values in ({"RANK": "2"}, {"WORLD_SIZE": "0"}, {"MASTER_ADDR": ""},
                       {"GPUS_PER_NODE": "0"}, {"GPUS_PER_NODE": "08"},
                       {"FSDP_SIZE": "3"}, {"TP_SIZE": "3"},
                       {"GLOBAL_BATCH_SIZE": "16"}, {"GRAD_ACCUM": "0"},
                       {"MASTER_PORT": "65536"}):
            with self.subTest(values=values):
                self.assertNotEqual(self.run_dry(**values).returncode, 0)

    def test_reject_duplicate_owned_arguments(self):
        self.assertNotEqual(self.run_dry("--fsdp", "1").returncode, 0)
        self.assertNotEqual(self.run_dry("--completion-config=credentials.json").returncode, 0)

    def test_resume_is_exclusive_with_initialization(self):
        self.assertNotEqual(self.run_dry(SFT_RESUME="/shared/resume.pt").returncode, 0)
        command = self.command(self.run_dry(SFT_INIT="", SFT_RESUME="/shared/resume.pt"))
        self.assertIn("--resume", command)
        self.assertNotIn("--init_from", command)

    def test_existing_workdir_requires_explicit_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.pt"
            checkpoint.touch()
            result = subprocess.run(["bash", str(LAUNCHER)], capture_output=True, text=True,
                                    env=launch_env(SFT_WORKDIR=tmp, SFT_INIT=str(checkpoint),
                                                   SFT_MANIFEST=str(checkpoint)), timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Workdir contains checkpoint.pt", result.stderr)

    def test_communication_check_does_not_require_training_files(self):
        command = self.command(self.run_dry("--check-communication", SFT_INIT="",
                                            SFT_MANIFEST="", SFT_WORKDIR=""))
        self.assertIn("training.distributed_smoke", command)
        self.assertNotIn("training.main", command)

    def test_two_node_agents_four_cpu_workers(self):
        # Two torchrun agents exercise exactly the platform rank translation.
        # Gloo on localhost verifies launcher/rendezvous, not GPU/NCCL transport.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = str(sock.getsockname()[1])
        with tempfile.TemporaryDirectory() as tmp:
            processes = []
            try:
                for rank in range(2):
                    env = launch_env(RANK=str(rank), GPUS_PER_NODE="2", MASTER_PORT=port,
                                     CHECK_BACKEND="gloo", SFT_WORKDIR=tmp)
                    processes.append(subprocess.Popen(
                        ["bash", str(LAUNCHER), "--check-communication", "--timeout", "30"],
                        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
                outputs = []
                for process in processes:
                    output, _ = process.communicate(timeout=60)
                    outputs.append(output)
                    self.assertEqual(process.returncode, 0, output)
                combined = "\n".join(outputs)
                for rank in range(4):
                    self.assertIn(f"rank={rank}/4", combined)
                for rank in range(2):
                    self.assertTrue((Path(tmp) / "logs" / f"node_{rank}.log").is_file())
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                for process in processes:
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()


if __name__ == "__main__":
    unittest.main()
