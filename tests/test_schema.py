from __future__ import annotations

import pytest
from pydantic import ValidationError

from geometry_reasoner.schema import FactModel, FormalizationResult


@pytest.mark.parametrize(
    ("predicate", "args"),
    [
        ("triangle", ["A", "B", "C"]),
        ("collinear", ["A", "B", "C"]),
        ("between", ["A", "B", "C"]),
        ("midpoint", ["M", "A", "B"]),
        ("on_perp_bisector", ["P", "A", "B"]),
        ("is_isosceles", ["A", "B", "C"]),
        ("angle_bisector", ["P", "A", "B", "C"]),
        ("equal_length", [["segment", "A", "B"], ["segment", "C", "D"]]),
        ("parallel", [["line", "A", "B"], ["line", "C", "D"]]),
        ("perpendicular", [["line", "A", "B"], ["line", "C", "D"]]),
        (
            "equal_angle",
            [["angle", "A", "B", "C"], ["angle", "D", "E", "F"]],
        ),
    ],
)
def test_supported_fact_signatures(predicate: str, args: list) -> None:
    fact = FactModel(pred=predicate, args=args)
    assert fact.pred == predicate


@pytest.mark.parametrize(
    "payload",
    [
        {"pred": "unknown", "args": ["A"]},
        {"pred": "midpoint", "args": ["M", "A"]},
        {"pred": "equal_length", "args": [["line", "A", "B"], ["line", "C", "D"]]},
        {"pred": "parallel", "args": [["segment", "A", "B"], ["line", "C", "D"]]},
        {"pred": "triangle", "args": ["A", "B", "C!"]},
    ],
)
def test_invalid_fact_signatures_are_rejected(payload: dict) -> None:
    with pytest.raises(ValidationError):
        FactModel.model_validate(payload)


def test_formalization_status_payloads_are_consistent() -> None:
    with pytest.raises(ValidationError):
        FormalizationResult(
            status="ok",
            problem=None,
            unsupported_reason=None,
        )

    unsupported = FormalizationResult(
        status="unsupported",
        problem=None,
        unsupported_reason="The question requires circles.",
    )
    assert unsupported.status == "unsupported"
