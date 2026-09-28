import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.error import URLError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.completion import (
    current_vm_id,
    load_completion_config,
    notify_task_completion,
    notify_task_completion_with_retries,
)


class CompletionTests(unittest.TestCase):
    def test_transient_notification_failure_recovers(self):
        with patch("training.completion.notify_task_completion",
                   side_effect=[URLError("unavailable"), 200]) as notify, \
                patch("training.completion.time.sleep"):
            status = notify_task_completion_with_retries({}, attempts=3)
        self.assertEqual(status, 200)
        self.assertEqual(notify.call_count, 2)

    def test_notification_failure_is_reported_after_retry_limit(self):
        with patch("training.completion.notify_task_completion",
                   side_effect=TimeoutError("timed out")) as notify, \
                patch("training.completion.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "failed after 3 attempts"):
                notify_task_completion_with_retries({}, attempts=3)
        self.assertEqual(notify.call_count, 3)

    def test_private_completion_config_and_notification_payload(self):
        expected = {"name": "trainer", "password": "secret", "vmids": ["gpu-0"]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "completion.json"
            path.write_text(json.dumps(dict(expected, vmids=["other-vm", "stale-vm"])), encoding="utf-8")
            path.chmod(0o600)
            with patch("training.completion.socket.gethostname", return_value="gpu-0.cluster.local"):
                config = load_completion_config(path)

            received = {}

            class Handler(BaseHTTPRequestHandler):
                def do_POST(self):
                    length = int(self.headers["Content-Length"])
                    received["path"] = self.path
                    received["payload"] = json.loads(self.rfile.read(length))
                    self.send_response(200)
                    self.end_headers()

                def log_message(self, *_):
                    pass

            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.handle_request, daemon=True)
            thread.start()
            try:
                status = notify_task_completion(
                    config,
                    url=f"http://127.0.0.1:{server.server_port}/api/task_finished",
                )
                thread.join(timeout=5)
            finally:
                server.server_close()

        self.assertEqual(status, 200)
        self.assertEqual(received, {"path": "/api/task_finished", "payload": expected})

    def test_credentials_only_json_and_ignored_hostname_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "completion.json"
            path.write_text(json.dumps(dict(name="trainer", password="secret")))
            path.chmod(0o600)
            with patch("training.completion.socket.gethostname", return_value="own-vm-0"), \
                    patch.dict("os.environ", {"HOSTNAME": "other-vm-0"}):
                self.assertEqual(load_completion_config(path),
                                 dict(name="trainer", password="secret", vmids=["own-vm-0"]))

    def test_invalid_hostname_fails(self):
        for hostname in ["", "localhost", "localhost.localdomain", "bad host", "-bad"]:
            with self.subTest(hostname=hostname), \
                    patch("training.completion.socket.gethostname", return_value=hostname):
                with self.assertRaisesRegex(ValueError, "current VM ID"):
                    current_vm_id()

    def test_completion_config_rejects_broad_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "completion.json"
            path.write_text(
                json.dumps({"name": "trainer", "password": "secret", "vmids": ["gpu-0"]}),
                encoding="utf-8",
            )
            path.chmod(0o644)
            with self.assertRaises(PermissionError):
                load_completion_config(path)


if __name__ == "__main__":
    unittest.main()
