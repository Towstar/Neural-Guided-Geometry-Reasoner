from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from .formalizer import (
    Formalizer,
    FormalizerConfig,
    FormalizationError,
    create_formalizer,
    materialize_problem,
)
from .loader import load_problems
from .prolog_bridge import PrologBridgeError
from .reasoner import solve


ROOT = Path(__file__).resolve().parents[1]


def evaluate(
    dataset_path: Path,
    image_manifest_path: Path,
    output_path: Path,
    *,
    limit: int | None = None,
    formalizer: Formalizer | None = None,
) -> dict[str, Any]:
    problems = load_problems(dataset_path)
    selected_ids = sorted(problems)[:limit]
    runner = formalizer or create_formalizer(FormalizerConfig())
    cases: list[dict[str, Any]] = []

    for problem_id in selected_ids:
        gold = problems[problem_id]
        statement = gold.get("statement")
        if not isinstance(statement, str) or not statement.strip():
            cases.append(_failed_case(problem_id, "text", "missing statement"))
            continue
        cases.append(
            _evaluate_case(
                case_id=problem_id,
                modality="text",
                gold=gold,
                source_fingerprint=statement.encode("utf-8"),
                formalize=lambda statement=statement: runner.formalize_text(statement),
                formalizer=runner,
            )
        )

    manifest = _load_image_manifest(image_manifest_path)
    for image_case in manifest:
        problem_id = image_case["problem_id"]
        if problem_id not in problems:
            cases.append(_failed_case(problem_id, "image", "unknown problem id"))
            continue
        image_path = ROOT / image_case["path"]
        cases.append(
            _evaluate_case(
                case_id=image_case["id"],
                modality="image",
                gold=problems[problem_id],
                source_fingerprint=image_path.read_bytes(),
                formalize=lambda image_path=image_path: runner.formalize_image(image_path),
                formalizer=runner,
            )
        )

    formalized_count = sum(case["formalization_success"] for case in cases)
    solved_count = sum(case["verified_correct"] for case in cases)
    attempted_steps = [
        case["proof_steps"]
        for case in cases
        if case["formalization_success"]
    ]
    text_cases = [case for case in cases if case["modality"] == "text"]
    image_cases = [case for case in cases if case["modality"] == "image"]
    text_target = math.ceil(len(text_cases) * 0.8)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": runner.provider,
        "model": runner.model,
        "metrics": {
            "formalization_success_rate": _rate(formalized_count, len(cases)),
            "proof_completion_rate": _rate(solved_count, len(cases)),
            "mean_steps_explored": mean(attempted_steps) if attempted_steps else 0.0,
        },
        "mvp_passed": (
            sum(case["verified_correct"] for case in text_cases) >= text_target
            and all(case["verified_correct"] for case in image_cases)
        ),
        "cases": cases,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _evaluate_case(
    *,
    case_id: str,
    modality: str,
    gold: dict[str, Any],
    source_fingerprint: bytes,
    formalize: Any,
    formalizer: Formalizer,
) -> dict[str, Any]:
    case: dict[str, Any] = {
        "id": case_id,
        "modality": modality,
        "formalization_success": False,
        "exact_match": False,
        "proof_solved": False,
        "verified_correct": False,
        "proof_steps": 0,
    }
    try:
        result = formalize()
        metadata = formalizer.last_call_metadata
        if metadata is not None:
            case["api"] = asdict(metadata)
        if result.status != "ok":
            case["error"] = result.unsupported_reason or "unsupported"
            return case

        case["formalization_success"] = True
        problem = materialize_problem(result, source_fingerprint)
        case["exact_match"] = (
            problem["givens"] == gold["givens"] and problem["goal"] == gold["goal"]
        )
        proof = solve(problem)
        case["proof_solved"] = proof.solved
        case["proof_steps"] = len(proof.steps)
        case["proof_status"] = proof.status
        case["verified_correct"] = bool(case["exact_match"] and proof.solved)
    except (FormalizationError, PrologBridgeError, OSError, ValueError) as error:
        case["error"] = str(error)
    return case


def _failed_case(case_id: str, modality: str, error: str) -> dict[str, Any]:
    return {
        "id": case_id,
        "modality": modality,
        "formalization_success": False,
        "exact_match": False,
        "proof_solved": False,
        "verified_correct": False,
        "proof_steps": 0,
        "error": error,
    }


def _load_image_manifest(path: Path) -> list[dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("image manifest must be a JSON list")
    required = {"id", "problem_id", "path"}
    result: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError("each image manifest item must contain id, problem_id, path")
        if not all(isinstance(item[key], str) for key in required):
            raise ValueError("image manifest values must be strings")
        result.append(item)
    return result


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate LLM geometry formalization.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "problems.json")
    parser.add_argument(
        "--image-manifest",
        type=Path,
        default=ROOT / "data" / "image_manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "eval.json",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--provider",
        choices=("openai", "ollama"),
        default="openai",
        help="LLM provider for formalization (default: openai)",
    )
    parser.add_argument("--model", help="Model name or Ollama model tag")
    parser.add_argument("--base-url", help="Provider endpoint")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    report = evaluate(
        args.dataset,
        args.image_manifest,
        args.output,
        limit=args.limit,
        formalizer=create_formalizer(
            FormalizerConfig(
                provider=args.provider,
                model=args.model,
                base_url=args.base_url,
            )
        ),
    )
    print(json.dumps({"metrics": report["metrics"], "mvp_passed": report["mvp_passed"]}, indent=2))
    print(f"Full report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
