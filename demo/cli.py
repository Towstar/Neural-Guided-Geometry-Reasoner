from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from geometry_reasoner.formalizer import (
    FormalizerConfig,
    FormalizationError,
    create_formalizer,
    materialize_problem,
)
from geometry_reasoner.prolog_bridge import PrologBridgeError
from geometry_reasoner.reasoner import ProofResult, solve
from geometry_reasoner.schema import FormalizationResult


EXIT_INPUT = 2
EXIT_FORMALIZATION = 3
EXIT_UNSOLVED = 4
EXIT_PROLOG = 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Formalize a geometry question and verify it with Prolog."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("-t", "--text", type=str, help="Geometry question written as text")
    source.add_argument("-i", "--image", type=Path, help="Local image of a geometry question")
    parser.add_argument(
        "-p",
        "--provider",
        choices=("openai", "ollama"),
        default="openai",
        help="LLM provider for formalization (default: openai)",
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Model name or Ollama model tag; uses the provider default when omitted",
    )
    parser.add_argument(
        "--base-url",
        help="Provider endpoint; defaults to the provider's configured endpoint",
    )
    parser.add_argument(
        "-M",
        "--max-steps",
        type=_nonnegative_int,
        default=32,
        help="Maximum derived facts before proof search stops (default: 32)",
    )
    parser.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="Emit one machine-readable JSON object",
    )
    return parser


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        formalizer = create_formalizer(
            FormalizerConfig(
                provider=args.provider,
                model=args.model,
                base_url=args.base_url,
            )
        )
        if args.text is not None:
            source_bytes = args.text.strip().encode("utf-8")
            formalization = formalizer.formalize_text(args.text)
        else:  # args.image is not None
            source_bytes = args.image.read_bytes() if args.image.exists() else b""
            formalization = formalizer.formalize_image(args.image)
    except (OSError, FormalizationError) as error:
        _emit_error(str(error), args.json)
        return EXIT_FORMALIZATION
    if formalization.status == "unsupported":
        _emit_error(
            formalization.unsupported_reason or "Unsupported geometry question",
            args.json,
        )
        return EXIT_FORMALIZATION

    try:
        problem = materialize_problem(formalization, source_bytes)
    except ValueError as error:
        _emit_error(str(error), args.json)
        return EXIT_INPUT

    try:
        proof = solve(problem, max_steps=args.max_steps)
    except PrologBridgeError as error:
        _emit_error(str(error), args.json)
        return EXIT_PROLOG

    _emit_success(formalization, problem, proof, args.json)
    return 0 if proof.solved else EXIT_UNSOLVED


def _emit_success(
    formalization: FormalizationResult,
    problem: dict,
    proof: ProofResult,
    json_output: bool,
) -> None:
    if json_output:
        print(
            json.dumps(
                {
                    "formalization": {
                        "status": formalization.status,
                        "problem": problem,
                        "unsupported_reason": None,
                    },
                    "proof": proof.model_dump(exclude_none=True),
                },
                indent=2,
            )
        )
        return

    print(f"Statement: {problem['statement']}")
    print("\nCanonical geometry JSON:")
    print(json.dumps(problem, indent=2))
    print(f"\nProof status: {proof.status}")
    if not proof.steps:
        print("No derivation steps were required.")
        return
    for index, step in enumerate(proof.steps, start=1):
        derived = step.derived.model_dump() if step.derived is not None else None
        print(f"{index}. {step.rule}: {json.dumps(derived)}")


def _emit_error(message: str, json_output: bool) -> None:
    if json_output:
        print(json.dumps({"error": message}))
    else:
        print(f"Error: {message}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
