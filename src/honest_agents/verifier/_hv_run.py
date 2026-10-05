"""The in-sandbox half of the hermetic pytest verifier. Standard library only, Python >= 3.9.

The host uploads this file with a one-shot config and runs `python3 -I -S -B run.py config.json` under `env -i`. pytest
comes from the vendored bundle, pinned to the front of `sys.path`; `-S` means no `site`, so no `sitecustomize`, `.pth`
file or `usercustomize` the agent planted can run (SPEC_V3 section 3.2, GR-3). pytest's own stdout and stderr go to a
log file; the result is written to the real stdout between two nonce markers, which the host captures from the exec
stream. The host also imports `tree_digest` from here, so both sides hash trees identically.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

BEGIN = "<<HV-BEGIN {nonce}>>"
END = "<<HV-END {nonce}>>"
LOG_TAIL_CHARS = 4000
_SKIP_DIRS = ("__pycache__", ".pytest_cache")


def tree_digest(root: str) -> str:
    """sha256 over the relative path and content of every file (and symlink target) under `root`, ignoring caches."""
    h = hashlib.sha256()
    entries = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS)
        for name in sorted(filenames):
            entries.append(os.path.join(dirpath, name))
    for path in entries:
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        h.update(rel.encode("utf-8", "surrogateescape") + b"\0")
        if os.path.islink(path):
            h.update(b"L" + os.readlink(path).encode("utf-8", "surrogateescape"))
        else:
            try:
                with open(path, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
            except OSError as e:  # unreadable file: make the digest differ rather than crash the verifier
                h.update(b"E" + str(e.errno).encode())
        h.update(b"\0")
    return h.hexdigest()


def undumpable() -> bool:
    """`prctl(PR_SET_DUMPABLE, 0)` on Linux: a process that is not dumpable cannot be read through `/proc/<pid>/mem`, and
    its `/proc/<pid>/fd/*` cannot be opened, by another process without CAP_SYS_PTRACE (which Docker drops by default),
    root included. A script the tests run therefore cannot recover the nonce from this process or write a forged pair into
    the exec pipe it holds. Best effort: False where there is no `prctl` (macOS) or no `ctypes`."""
    try:
        import ctypes

        return ctypes.CDLL(None, use_errno=True).prctl(4, 0, 0, 0, 0) == 0  # PR_SET_DUMPABLE = 4
    except Exception:  # noqa: BLE001 - hardening only; the digests and the marker rules still apply
        return False


class _Collector:
    """Per-test outcomes from pytest's own reports, so nothing on the filesystem is trusted for the result."""

    def __init__(self) -> None:
        self.outcomes: dict[str, str] = {}
        self.collect_errors = 0

    def pytest_collectreport(self, report) -> None:
        if report.failed:
            self.collect_errors += 1

    def pytest_runtest_logreport(self, report) -> None:
        nid = report.nodeid
        xfail = hasattr(report, "wasxfail")
        if report.when == "call":
            if report.passed:
                value = "xpassed" if xfail else "passed"
            elif report.skipped:
                value = "xfailed" if xfail else "skipped"
            else:
                value = "failed"
        elif report.failed:
            value = "failed"  # setup or teardown error
        elif report.skipped:
            value = "skipped"
        else:
            return
        if self.outcomes.get(nid) != "failed":
            self.outcomes[nid] = value


def main(config_path: str, tests: str, bundle: str, ini: str, log: str) -> int:
    """`config_path` holds only the nonce; the paths come from argv, where the host's shell has already expanded them."""
    hardened = undumpable()  # first, before the nonce is read and long before anything of the tests' own can run
    with open(config_path) as f:
        cfg = json.load(f)
    cfg.update({"tests": tests, "bundle": bundle, "ini": ini, "log": log})
    try:
        os.unlink(config_path)  # the nonce lives on disk only until this process has read it
    except OSError:
        pass
    sys.dont_write_bytecode = True
    sys.path.insert(0, cfg["bundle"])
    real_stdout = os.dup(1)
    bundle_digest = tree_digest(cfg["bundle"])
    tests_before = tree_digest(cfg["tests"])

    log_fd = os.open(cfg["log"], os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o666)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    import pytest

    collector = _Collector()
    args = [
        "-c",
        cfg["ini"],
        "--rootdir",
        cfg["tests"],
        "--confcutdir",
        cfg["tests"],
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=",
        "--color=no",
        "--tb=short",
        "-q",
        cfg["tests"],
    ]
    exit_code = int(pytest.main(args, plugins=[collector]))
    sys.stdout.flush()
    sys.stderr.flush()
    tests_after = tree_digest(cfg["tests"])
    bundle_after = tree_digest(cfg["bundle"])  # modules pytest imports lazily would otherwise be a window
    try:
        with open(cfg["log"], "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - LOG_TAIL_CHARS))
            log_tail = f.read().decode("utf-8", "replace")
    except OSError:
        log_tail = ""
    payload = {
        "nonce": cfg["nonce"],
        "exit_code": exit_code,
        "outcomes": collector.outcomes,
        "collect_errors": collector.collect_errors,
        "tests_before": tests_before,
        "tests_after": tests_after,
        "bundle": bundle_digest,
        "bundle_after": bundle_after,
        "hardened": hardened,
        "pytest": getattr(pytest, "__version__", "?"),
        "python": "{}.{}.{}".format(*sys.version_info[:3]),
        "log_tail": log_tail,
    }
    text = "\n{}\n{}\n{}\n".format(
        BEGIN.format(nonce=cfg["nonce"]), json.dumps(payload), END.format(nonce=cfg["nonce"])
    )
    os.write(real_stdout, text.encode("utf-8"))
    return 0


if __name__ == "__main__":
    code = main(*sys.argv[1:6])
    sys.stdout.flush()
    os._exit(code)  # skip atexit handlers the code under test may have registered
