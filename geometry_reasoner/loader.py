from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from .canonicalize import canonicalize_problem
from .schema import ProblemFileModel, ProblemModel

Problem = dict[str, Any]

def load_problems(path: str | Path) -> dict[str, Problem]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Problem file not found: {path}")
    raw_text = path.read_text(encoding="utf-8")
    parsed_problems = parse_problem_file(raw_text)
    problems_by_id: dict[str, Problem] = {}
    for problem_model in parsed_problems:
        raw_problem = problem_model.model_dump()
        normalized_problem = canonicalize_problem(raw_problem)
        problem_id = normalized_problem["id"]
        
        if problem_id in problems_by_id:
            raise ValueError(f"Duplicate problem id found: {problem_id}")
        problems_by_id[problem_id] = normalized_problem
        
    return problems_by_id

def load_problem(path: str | Path, problem_id: str) -> Problem:
    problems = load_problems(path)
    return get_problem_by_id(problems, problem_id)

def parse_problem_file(raw_text: str) -> list[ProblemModel]:
    list_adapter = TypeAdapter(list[ProblemModel])

    list_error = None
    try:
        return list_adapter.validate_json(raw_text)
    except ValidationError as e:
        list_error = e

    wrapped_error = None
    try:
        wrapped = ProblemFileModel.model_validate_json(raw_text)
        return wrapped.problems
    except ValidationError as e:
        wrapped_error = e

    raise ValueError(
        "Invalid problems JSON.\n\n"
        f"Top-level list parse failed:\n{list_error}\n\n"
        f"Wrapped object parse failed:\n{wrapped_error}"
    )

def get_problem_by_id(problems: dict[str, Problem], problem_id: str) -> Problem:
    try:
        return problems[problem_id]
    except KeyError as e:
        available = ", ".join(sorted(problems.keys()))
        raise ValueError(f"Problem with id '{problem_id}' not found. Available ids: {available}") from e
    
def list_problem_ids(problems: dict[str, Problem]) -> list[str]:
    return sorted(problems.keys())