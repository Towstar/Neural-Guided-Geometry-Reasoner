from __future__ import annotations

import os
from pathlib import Path

import pytest

from geometry_reasoner.formalizer import formalize_image, formalize_text, materialize_problem
from geometry_reasoner.reasoner import solve


pytestmark = [
    pytest.mark.live_api,
    pytest.mark.skipif(
        os.getenv("RUN_OPENAI_LIVE_TESTS") != "1",
        reason="set RUN_OPENAI_LIVE_TESTS=1 to make billable API requests",
    ),
]

ROOT = Path(__file__).resolve().parents[1]


def test_live_text_to_verified_proof() -> None:
    statement = "D is the midpoint of AB. Prove that A, D, and B are collinear."
    formalization = formalize_text(statement)
    problem = materialize_problem(formalization, statement.encode("utf-8"))
    assert solve(problem).solved is True


def test_live_image_to_verified_proof() -> None:
    image_path = ROOT / "data" / "images" / "geom_0002.png"
    formalization = formalize_image(image_path)
    problem = materialize_problem(formalization, image_path.read_bytes())
    assert solve(problem).solved is True
