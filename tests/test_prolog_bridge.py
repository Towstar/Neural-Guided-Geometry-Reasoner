from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from geometry_reasoner.prolog_bridge import PrologBridgeError, query_state


def test_query_state_returns_json_candidates_in_deterministic_order() -> None:
    state = query_state(
        [{"pred": "midpoint", "args": ["D", "B", "C"]}],
        {"pred": "collinear", "args": ["B", "D", "C"]},
    )

    assert state.goal_reached is False
    assert [candidate.rule for candidate in state.candidates] == [
        "midpoint_def_collinear",
        "midpoint_def_equal_segments",
    ]


def test_query_state_reports_missing_swipl() -> None:
    with pytest.raises(PrologBridgeError, match="was not found"):
        query_state(
            [],
            {"pred": "collinear", "args": ["A", "B", "C"]},
            executable="definitely-not-a-real-swipl-command",
        )


def test_query_state_rejects_malformed_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "geometry_reasoner.prolog_bridge.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout="not json", stderr="", returncode=0
        ),
    )
    with pytest.raises(PrologBridgeError, match="malformed JSON"):
        query_state([], {"pred": "collinear", "args": ["A", "B", "C"]})


def test_query_state_reports_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    def time_out(*args: object, **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd="swipl", timeout=10)

    monkeypatch.setattr("geometry_reasoner.prolog_bridge.subprocess.run", time_out)
    with pytest.raises(PrologBridgeError, match="timed out"):
        query_state([], {"pred": "collinear", "args": ["A", "B", "C"]})


def test_query_state_reports_nonzero_bridge_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "geometry_reasoner.prolog_bridge.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout='{"ok": false, "error": "bridge rejected request"}',
            stderr="internal details",
            returncode=1,
        ),
    )
    with pytest.raises(PrologBridgeError, match="bridge rejected request"):
        query_state([], {"pred": "collinear", "args": ["A", "B", "C"]})
