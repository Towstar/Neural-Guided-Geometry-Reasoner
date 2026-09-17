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


class StubFormalizer:
    def __init__(self, result: FormalizationResult) -> None:
        self.result = result

    def formalize_text(self, text: str) -> FormalizationResult:
        return self.result

    def formalize_image(self, path: object) -> FormalizationResult:
        return self.result


def test_cli_json_success(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "create_formalizer",
        lambda config: StubFormalizer(_formalization()),
    )
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
        cli,
        "create_formalizer",
        lambda config: StubFormalizer(_formalization("unsupported")),
    )
    exit_code = cli.main(["--text", "A problem"])
    assert exit_code == 3
    assert "Unsupported construction" in capsys.readouterr().err


def test_cli_uses_provider_and_model_to_create_formalizer(monkeypatch, capsys) -> None:
    captured = []
    monkeypatch.setattr(
        cli,
        "create_formalizer",
        lambda config: captured.append(config) or StubFormalizer(_formalization()),
    )
    monkeypatch.setattr(
        cli,
        "solve",
        lambda problem, max_steps: ProofResult(
            status="exhausted",
            solved=False,
            problem_id=problem["id"],
            steps=[],
            final_facts=problem["givens"],
        ),
    )

    exit_code = cli.main(
        [
            "--text",
            "A problem",
            "--provider",
            "ollama",
            "--model",
            "geometry-local",
            "--base-url",
            "http://localhost:11434",
        ]
    )

    assert exit_code == 4
    assert len(captured) == 1
    assert captured[0].provider == "ollama"
    assert captured[0].model == "geometry-local"
    assert captured[0].base_url == "http://localhost:11434"
    assert capsys.readouterr().err == ""
