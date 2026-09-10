from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from .canonicalize import (
    canonicalize_fact,
    canonicalize_problem,
    deduplicate_facts,
    sort_facts,
)
from .prolog_bridge import query_state
from .schema import FactModel, ProblemModel, ProofStepModel

#----------------------------------------------------------
# Controller for the deterministic proof search process, which uses the Prolog bridge to query the state of the proof.
#----------------------------------------------------------

ProofStatus = Literal["already_solved", "solved", "exhausted", "step_limit"]


class ProofResult(BaseModel):
    """The result of a proof search, including the status, whether the problem was solved, and the steps taken."""
    model_config = ConfigDict(extra="forbid")

    status: ProofStatus
    solved: bool
    problem_id: str
    steps: list[ProofStepModel]
    final_facts: list[FactModel]


def solve(problem: dict[str, Any] | ProblemModel, max_steps: int = 32) -> ProofResult:
    """Run deterministic, bounded forward proof search through SWI-Prolog."""
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    validated = (
        problem if isinstance(problem, ProblemModel) else ProblemModel.model_validate(problem)
    )
    canonical = canonicalize_problem(validated.model_dump())
    facts = sort_facts(deduplicate_facts(list(canonical["givens"])))
    goal = canonicalize_fact(canonical["goal"])
    steps: list[ProofStepModel] = []

    state = query_state(facts, goal)
    if state.goal_reached:
        return _result("already_solved", canonical["id"], steps, facts)

    for _ in range(max_steps):
        if not state.candidates:
            return _result("exhausted", canonical["id"], steps, facts)

        selected = state.candidates[0]
        facts = sort_facts(deduplicate_facts([*facts, selected.derived]))
        steps.append(
            ProofStepModel(rule=selected.rule, derived=selected.derived, facts_used=None)
        )

        if goal in facts:
            return _result("solved", canonical["id"], steps, facts)

        state = query_state(facts, goal)
        if state.goal_reached:
            return _result("solved", canonical["id"], steps, facts)

    return _result("step_limit", canonical["id"], steps, facts)


def _result(
    status: ProofStatus,
    problem_id: str,
    steps: list[ProofStepModel],
    facts: list[dict[str, Any]],
) -> ProofResult:
    return ProofResult(
        status=status,
        solved=status in {"already_solved", "solved"},
        problem_id=problem_id,
        steps=steps,
        final_facts=[FactModel.model_validate(fact) for fact in facts],
    )
