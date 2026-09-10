from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .canonicalize import Fact, Problem, canonicalize_fact
from .schema import FactModel

#----------------------------------------------------------
# Connects Python to Prolog via a JSON bridge
#----------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
JSON_BRIDGE = ROOT / "prolog" / "json_bridge.pl"
ATOM_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class PrologBridgeError(RuntimeError):
    """Raised when the SWI-Prolog process cannot return a valid state response."""


@dataclass(frozen=True)
class Candidate:
    """A candidate fact derived by a Prolog rule from the given facts."""
    rule: str
    derived: Fact


@dataclass(frozen=True)
class PrologState:
    """The state of a Prolog query, including whether the goal was reached and any candidate facts derived."""
    goal_reached: bool
    candidates: tuple[Candidate, ...]

#region: Legacy Prolog Bridge Functions (Before I found out Prolog has a JSON library...)

def atomize_name(value: str) -> str:
    """Convert a string to a valid Prolog atom by lowercasing and validating it."""
    if not isinstance(value, str) or not ATOM_PATTERN.fullmatch(value.strip()):
        raise ValueError(
            "Prolog names must start with a letter and contain only letters, "
            "numbers, and underscores"
        )
    return value.strip().lower()


def term_to_prolog(term: Any) -> str:
    """Convert a canonicalized JSON term into a Prolog term string."""
    if isinstance(term, str):
        return atomize_name(term)
    if isinstance(term, (int, float)):
        return str(term)
    if isinstance(term, list):
        if not term:
            raise ValueError("Cannot convert an empty list to a Prolog term")
        head = atomize_name(term[0])
        args = [term_to_prolog(arg) for arg in term[1:]]
        return f"{head}({', '.join(args)})"
    raise ValueError(f"Unsupported term type: {type(term)}")


def fact_to_prolog(fact: Fact, terminate: bool = True) -> str:
    validated = FactModel.model_validate(fact)
    text = term_to_prolog([validated.pred, *validated.args])
    return f"{text}." if terminate else text


def problem_to_prolog_facts(problem: Problem) -> list[str]:
    givens = problem.get("givens", [])
    if not isinstance(givens, list):
        raise ValueError("Problem 'givens' must be a list of facts")
    return [fact_to_prolog(fact) for fact in givens]


def goal_to_prolog(goal: Fact, terminate: bool = True) -> str:
    return fact_to_prolog(goal, terminate=terminate)


def problem_to_prolog_goal(problem: Problem) -> str:
    goal = problem.get("goal")
    if not isinstance(goal, dict):
        raise ValueError("Problem must contain a 'goal' object")
    return goal_to_prolog(goal)


def problem_to_prolog_program(
    problem: Problem,
    include_goal_comments: bool = True,
) -> str:
    fact_lines = problem_to_prolog_facts(problem)
    goal_line = problem_to_prolog_goal(problem)
    lines = fact_lines[:]
    if include_goal_comments:
        lines.extend(["", "% goal:", f"% {goal_line}"])
    return "\n".join(lines)

#endregion


def query_state(
    facts: list[Fact],
    goal: Fact,
    *,
    executable: str = "swipl",
    timeout_seconds: float = 10.0,
) -> PrologState:
    """Query the Prolog bridge with a set of facts and a goal, returning the state of the query."""
    payload = {
        "facts": [FactModel.model_validate(fact).model_dump() for fact in facts],
        "goal": FactModel.model_validate(goal).model_dump(),
    }

    try:
        # Starts a SWI-Prolog process that runs the JSON bridge script, passing the facts and goal as JSON input.
        completed = subprocess.run(
            [
                executable,
                "-q", # quiet mode
                "-s",
                str(JSON_BRIDGE), # load the bridge module
                "-g",
                "json_bridge:main", # run main/0
                "-t",
                "halt", # exit upon execution completion
            ],
            cwd=ROOT,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError as error:
        raise PrologBridgeError(
            "SWI-Prolog executable 'swipl' was not found on PATH"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise PrologBridgeError("SWI-Prolog query timed out") from error

    response = _parse_response(completed.stdout) # parses the JSON output from Prolog into a Python dictionary
    
    if completed.returncode != 0 or response.get("ok") is not True: # if theres a known failure from the Prolog end
        message = response.get("error")
        if not isinstance(message, str) or not message:
            message = "SWI-Prolog bridge failed"
        raise PrologBridgeError(message[:500])

    goal_reached = response.get("goal_reached")
    raw_candidates = response.get("candidates")
    if not isinstance(goal_reached, bool) or not isinstance(raw_candidates, list):
        raise PrologBridgeError("SWI-Prolog returned an invalid state response")

    candidates: list[Candidate] = []
    for raw in raw_candidates:
        # require rule to be a dictionary
        if not isinstance(raw, dict):
            raise PrologBridgeError("SWI-Prolog returned an invalid candidate")
        
        rule = raw.get("rule")
        # require rule to be a non-empty string
        if not isinstance(rule, str) or not rule:
            raise PrologBridgeError("SWI-Prolog returned a candidate without a rule")
        try:
            derived = FactModel.model_validate(raw.get("derived")).model_dump()
        except ValidationError as error:
            raise PrologBridgeError(
                "SWI-Prolog returned an invalid derived fact"
            ) from error
        # canonicalize the candidate and append it to the candidates list
        candidates.append(Candidate(rule=rule, derived=canonicalize_fact(derived)))

    #return immutable snapshot of the Prolog state with goal_reached and candidates
    return PrologState(goal_reached=goal_reached, candidates=tuple(candidates))


def _parse_response(stdout: str) -> dict[str, Any]:
    """Parse the JSON response from the Prolog bridge, raising an error if it is malformed."""
    try:
        response = json.loads(stdout.strip())
    except json.JSONDecodeError as error:
        raise PrologBridgeError("SWI-Prolog returned malformed JSON") from error
    if not isinstance(response, dict):
        raise PrologBridgeError("SWI-Prolog returned a non-object JSON response")
    return response
