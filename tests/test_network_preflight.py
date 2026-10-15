"""Network-policy preflight (iteration 15 G3 incident): a cached False from harbor's probe must be retried."""

from __future__ import annotations

import pytest

from honest_agents.env import preflight_network_policy

DockerEnvironment = pytest.importorskip("harbor.environments.docker.docker").DockerEnvironment


class FakeProbe:
    def __init__(self, results):
        self.results, self.calls, self.cleared = list(results), 0, 0

    def __call__(self):
        self.calls += 1
        return self.results.pop(0)

    def cache_clear(self):
        self.cleared += 1


def patch(monkeypatch, results):
    probe = FakeProbe(results)
    monkeypatch.setattr(DockerEnvironment, "_egress_control_kernel_support", probe)
    return probe


def test_fails_twice_then_succeeds(monkeypatch):
    probe = patch(monkeypatch, [False, False, True])
    assert preflight_network_policy(sleep=lambda s: None) is True
    assert probe.calls == 3 and probe.cleared == 2


def test_always_fails_raises(monkeypatch):
    probe = patch(monkeypatch, [False, False, False])
    with pytest.raises(RuntimeError, match="cannot enforce network policies"):
        preflight_network_policy(sleep=lambda s: None)
    assert probe.calls == 3 and probe.cleared == 3  # never leave the cached False behind


def test_skipped_when_off_or_not_docker(monkeypatch):
    probe = patch(monkeypatch, [False])
    assert preflight_network_policy(enforce_network_policy=False) is False
    assert preflight_network_policy(environment_type="modal") is False
    assert probe.calls == 0


def test_skipped_without_the_method(monkeypatch):
    monkeypatch.delattr(DockerEnvironment, "_egress_control_kernel_support")
    assert preflight_network_policy() is False
