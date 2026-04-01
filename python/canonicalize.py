from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, List, Tuple

JSONValue = Any
Fact = Dict[str, Any]
Problem = Dict[str, Any]

def cannonicalize_fact(fact: Fact) -> Fact:
    pass

def canonicalize_goal(goal: Fact) -> Fact:
    return cannonicalize_fact(goal)

def canonicalize_problem(problem: Problem) -> Problem:
    pass