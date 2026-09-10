from __future__ import annotations

import json

from demo import cli
from geometry_reasoner.reasoner import ProofResult
from geometry_reasoner.schema import FormalizationResult


def _formalization(status: str = "ok") -> FormalizationResult:
    if status == "unsupported":
        return FormalizationResult(
            status="unsupported",
            problem=None,
            unsupported_reason="Unsupported construction",
        )
    return FormalizationResult.model_validate(
        {
            "status": "ok",
            "problem": {
                "statement": "D is the midpoint of AB. Prove A, D, B are collinear.",
                "givens": [{"pred": "midpoint", "args": ["D", "A", "B"]}],
                "goal": {"pred": "collinear", "args": ["A", "D", "B"]},
            },
            "unsupported_reason": None,
        }
    )


def test_cli_json_success(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "formalize_text", lambda text: _formalization())
    monkeypatch.setattr(
        cli,
        "solve",
        lambda problem, max_steps: ProofResult(
            status="solved",
            solved=True,
            problem_id=problem["id"],
            steps=[],
            final_facts=problem["givens"],
        ),
    )

    exit_code = cli.main(["--text", "A problem", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["formalization"]["status"] == "ok"
    assert payload["proof"]["solved"] is True


def test_cli_unsupported_input_has_exit_three(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli, "formalize_text", lambda text: _formalization("unsupported")
    )
    exit_code = cli.main(["--text", "A problem"])
    assert exit_code == 3
    assert "Unsupported construction" in capsys.readouterr().err
