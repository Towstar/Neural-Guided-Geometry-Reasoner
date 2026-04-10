from typing import Any

from geometry_reasoner.canonicalize import Problem
from geometry_reasoner.canonicalize import Fact

def atomize_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Expected a non-empty string name for atomization, got {type(value)}")
    return value.strip().lower().replace(" ", "_").replace("-", "_")

def term_to_prolog(term: Any) -> str:
    """Convert a canonicalized term into a Prolog term string.
        For example, ["segment", "A", "B"] would become "segment(a, b)". The first element is treated as the functor, and the rest are arguments.
    """
    if isinstance(term, str):
        return atomize_name(term)
    if isinstance(term, (int, float)):
        return atomize_name(str(term))
    if isinstance(term, list):
        if not term:
            raise ValueError("Cannot convert an empty list to a Prolog term")
        head = term[0]
        if not isinstance(head, str) or not head.strip():
            raise ValueError(f"First element of term list must be a non-empty string to serve as the functor, got: {type(head)}")
        functor = atomize_name(head)
        args = [term_to_prolog(arg) for arg in term[1:]]
        return f"{functor}({', '.join(args)})"
    raise ValueError(f"Unsupported term type: {type(term)}. Expected a string or a list.")

def fact_to_prolog(fact: Fact, terminate: bool = True) -> str:
    "Convert a fact dict into a Prolog fact string."
    if not isinstance(fact, dict):
        raise ValueError(f"Fact must be a dict, got: {type(fact)}")
    pred = fact.get("pred")
    args = fact.get("args", [])
    if not isinstance(pred, str) or not pred.strip():
        raise ValueError(f"Fact 'pred' must be a string, got: {type(pred)}")
    if not isinstance(args, list):
        raise ValueError(f"Fact 'args' must be a list, got: {type(args)}")
    
    functor = atomize_name(pred)
    rendered_args = [term_to_prolog(arg) for arg in args]
    text = f"{functor}({', '.join(rendered_args)})"
    if terminate:
        text += "."
    return text

def problem_to_prolog_facts(problem: Problem) -> list[str]:
    """
    Convert a conicalized problem's facts into a list of Prolog fact strings.
    Only givens are included, not the goal.
    """
    if not isinstance(problem, dict):
        raise ValueError("Incorrect type for problem; expected a Problem dict")
    
    givens = problem.get("givens", [])
    if not isinstance(givens, list):
        raise ValueError("Problem 'givens' must be a list of facts")
    return [fact_to_prolog(fact) for fact in givens]

def goal_to_prolog(goal: Fact, terminate: bool = True) -> str:
    return fact_to_prolog(goal, terminate=terminate)

def problem_to_prolog_goal(problem: Problem) -> str:
    """
    Convert the problem goal into a Prolog goal string.
    """
    goal = problem.get("goal")
    if not isinstance(goal, dict):
        raise ValueError("Problem must contain a 'goal' object")

    return goal_to_prolog(goal)


def problem_to_prolog_program(problem: Problem, include_goal_comments: bool = True) -> str:
    """Render a complete Prolog program for the given problem, including facts and an optional comment with the goal."""
    fact_lines = problem_to_prolog_facts(problem)
    goal_line = problem_to_prolog_goal(problem)
    lines = fact_lines[:]
    if include_goal_comments:
        lines.append("")
        lines.append("% goal: ")
        lines.append(f"% {goal_line}")
    return "\n".join(lines)