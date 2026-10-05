"""Test-only Harbor environment that runs a trial's commands as host subprocesses (no Docker), plus a scripted
OpenAI-compatible HTTP server. Together they let `harbor run` execute a real job end to end on a laptop:

    harbor run -p TASKS -e tests.eval.host_env:HostEnvironment -a honest_agents.eval.harbor_agent:HonestPolicyAgent ...

Container-absolute paths Harbor uses (/tests, /logs, /solution, /app) are rewritten into a private root per trial.
The `user` argument is ignored (no setuid on a laptop). Never use this for anything but tests.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Self

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities

_ABS = re.compile(r"(?<![\w./-])/(tests|logs|solution|app)(?=/|\b)")


class HostEnvironment(BaseEnvironment):
    @staticmethod
    def type() -> str:
        return "host-test"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities()

    def _validate_definition(self) -> None:
        return None

    @property
    def root(self) -> Path:
        return Path(self.trial_paths.trial_dir) / "sandbox_root"

    def _map(self, path: str | Path) -> Path:
        p = str(path)
        return Path(_ABS.sub(lambda m: f"{self.root}/{m.group(1)}", p)) if p.startswith("/") else self.root / p

    @property
    def _path_env(self) -> dict[str, str]:
        # read by tests.sandbox_fakes.PASS_IF_HELLO, whose /logs/verifier is inside the script (not rewritable in `command`)
        return {
            "HA_VERIFIER_DIR": str(self.root / "logs" / "verifier"),
            "HA_TESTS_DIR": str(self.root / "tests"),
        }

    async def start(self, force_build: bool) -> None:
        for d in ("app", "logs/agent", "logs/verifier", "logs/artifacts"):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    async def stop(self, delete: bool) -> None:
        return None

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None) -> ExecResult:
        cmd = _ABS.sub(lambda m: f"{self.root}/{m.group(1)}", command)
        proc = await asyncio.create_subprocess_exec(
            "/bin/bash",
            "-c",
            cmd,
            cwd=str(self._map(cwd)) if cwd else str(self.root / "app"),
            env={**os.environ, **self._path_env, **(self._merge_env(env) or {})},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout_sec)
        except TimeoutError:
            proc.kill()
            raise RuntimeError(f"Command timed out after {timeout_sec} seconds") from None
        return ExecResult(stdout=out.decode(), stderr=err.decode(), return_code=proc.returncode)

    async def upload_file(self, source_path, target_path) -> None:
        dst = self._map(target_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dst)

    async def upload_dir(self, source_dir, target_dir) -> None:
        shutil.copytree(source_dir, self._map(target_dir), dirs_exist_ok=True)

    async def download_file(self, source_path, target_path) -> None:
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self._map(source_path), target_path)

    async def download_dir(self, source_dir, target_dir) -> None:
        src = self._map(source_dir)
        if src.exists():
            shutil.copytree(src, target_dir, dirs_exist_ok=True)


class ScriptedChatServer:
    """Minimal `/v1/chat/completions` server: `policy(request_json, call_index) -> message dict` (OpenAI wire format)."""

    def __init__(self, policy):
        self.policy = policy
        self.requests: list[dict[str, Any]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                i = len(outer.requests)
                outer.requests.append(body)
                msg = outer.policy(body, i)
                payload = {
                    "id": f"c{i}",
                    "object": "chat.completion",
                    "created": 0,
                    "model": body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", **msg},
                            "finish_reason": "tool_calls" if msg.get("tool_calls") else "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100 + 10 * i,
                        "completion_tokens": 5,
                        "total_tokens": 105 + 10 * i,
                    },
                }
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def __enter__(self) -> Self:
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()


def tool_msg(name: str, args: dict[str, Any], call_id: str = "c") -> dict[str, Any]:
    return {
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }
