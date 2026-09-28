"""Notify the GPU task manager after a training job completes successfully."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import socket
import stat
import time
from urllib import error, request


DEFAULT_COMPLETION_URL = (
    "http://k8svmgr-main.devops.svc.cluster.local:8000/api/task_finished"
)


def current_vm_id() -> str:
    """This VM platform uses the machine's short hostname as its task-manager ID."""
    vmid = socket.gethostname().split(".", 1)[0]
    if (not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", vmid)
            or vmid.lower() in {"localhost", "localhost6"}):
        raise ValueError("Cannot determine the current VM ID from the machine hostname")
    return vmid


def load_completion_config(path: str | os.PathLike[str]) -> dict:
    """Load private credentials and target only this VM; ignore stored vmids."""
    path = Path(path).expanduser().resolve()
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        raise PermissionError(
            f"completion config must not be group/world accessible: {path} has mode {mode:04o}; "
            f"run chmod 600 {path}"
        )

    value = json.loads(path.read_text(encoding="utf-8"))
    name = value.get("name")
    password = value.get("password")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{path}: name must be a nonempty string")
    if not isinstance(password, str) or not password:
        raise ValueError(f"{path}: password must be a nonempty string")
    return {"name": name.strip(), "password": password, "vmids": [current_vm_id()]}


def notify_task_completion(config: dict, url: str = DEFAULT_COMPLETION_URL,
                           timeout: float = 15) -> int:
    """POST a task-finished notification and return the HTTP status code."""
    payload = json.dumps(config).encode("utf-8")
    req = request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        response.read()
        return response.status


def notify_task_completion_with_retries(config: dict, url: str = DEFAULT_COMPLETION_URL,
                                        timeout: float = 15, attempts: int = 3,
                                        log_fn=None) -> int:
    """Notify the task manager, retrying transient request failures."""
    if attempts <= 0 or timeout <= 0:
        raise ValueError("completion notification attempts and timeout must be positive")
    for attempt in range(1, attempts + 1):
        try:
            return notify_task_completion(config, url=url, timeout=timeout)
        except (error.URLError, TimeoutError) as exc:
            if attempt == attempts:
                raise RuntimeError(
                    f"training completed, but completion notification failed after {attempts} "
                    f"attempts: {type(exc).__name__}: {exc}"
                ) from exc
            delay = min(2 ** (attempt - 1), 5)
            if log_fn is not None:
                log_fn(
                    f"completion notification attempt {attempt}/{attempts} failed; "
                    f"retrying in {delay}s"
                )
            time.sleep(delay)

    raise AssertionError("unreachable")
