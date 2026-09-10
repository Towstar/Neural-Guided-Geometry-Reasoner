from __future__ import annotations

from geometry_reasoner.reasoner import solve


def _problem(givens: list[dict], goal: dict, problem_id: str = "test") -> dict:
    points: set[str] = set()
    for fact in [*givens, goal]:
        for argument in fact["args"]:
            if isinstance(argument, str):
                points.add(argument)
            else:
                points.update(argument[1:])
    return {
        "id": problem_id,
        "entities": {"points": sorted(points)},
        "givens": givens,
        "goal": goal,
    }


def test_already_solved_goal_requires_no_steps() -> None:
    fact = {"pred": "collinear", "args": ["C", "A", "B"]}
    result = solve(_problem([fact], {"pred": "collinear", "args": ["A", "B", "C"]}))

    assert result.status == "already_solved"
    assert result.solved is True
    assert result.steps == []


def test_deterministic_candidate_ordering() -> None:
    result = solve(
        _problem(
            [{"pred": "midpoint", "args": ["D", "B", "C"]}],
            {
                "pred": "equal_length",
                "args": [["segment", "B", "D"], ["segment", "D", "C"]],
            },
        )
    )

    assert result.status == "solved"
    assert [step.rule for step in result.steps] == [
        "midpoint_def_collinear",
        "midpoint_def_equal_segments",
    ]


def test_genuine_two_rule_proof() -> None:
    result = solve(
        _problem(
            [
                {"pred": "between", "args": ["A", "D", "B"]},
                {
                    "pred": "equal_length",
                    "args": [["segment", "A", "D"], ["segment", "B", "D"]],
                },
            ],
            {"pred": "midpoint", "args": ["D", "A", "B"]},
        )
    )

    assert result.status == "solved"
    assert [step.rule for step in result.steps] == [
        "between_implies_collinear",
        "midpoint_converse",
    ]


def test_exhausted_when_no_rule_applies() -> None:
    result = solve(
        _problem(
            [{"pred": "triangle", "args": ["A", "B", "C"]}],
            {"pred": "midpoint", "args": ["D", "A", "B"]},
        )
    )
    assert result.status == "exhausted"
    assert result.solved is False


def test_step_limit_stops_search() -> None:
    result = solve(
        _problem(
            [{"pred": "midpoint", "args": ["D", "B", "C"]}],
            {
                "pred": "equal_length",
                "args": [["segment", "B", "D"], ["segment", "D", "C"]],
            },
        ),
        max_steps=1,
    )
    assert result.status == "step_limit"
    assert result.solved is False
    assert len(result.steps) == 1
