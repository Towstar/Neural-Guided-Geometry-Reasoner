from __future__ import annotations
from copy import deepcopy
from typing import Any, Dict, List, Tuple

JSONValue = Any
Fact = Dict[str, Any]
Problem = Dict[str, Any]

def canonicalize_term(term: JSONValue) -> JSONValue:
    """
    Canonicalize structured geometry terms.

    Examples:
      ["segment", "B", "A"] -> ["segment", "A", "B"]
      ["line", "D", "C"]    -> ["line", "C", "D"]

    Ordered objects like angles are preserved:
      ["angle", "A", "B", "C"] stays as-is
    """
    if not isinstance(term, list):
        return term
    if len(term) == 0:
        return term
    head = term[0]

    if head in {"segment", "line"}:
        if len(term) != 3:
            raise ValueError(f"Invalid {head} term: {term}")
        p1, p2 = term[1], term[2]
        if not (is_point_name(p1) and is_point_name(p2)):
            raise ValueError(f"Invalid endpoints in {head} term: {term}")
        a, b = sort_points(p1, p2)
        return [head, a, b]

    if head == "angle":
        if len(term) != 4:
            raise ValueError(f"Invalid angle term: {term}")
        a, b, c = term[1], term[2], term[3]
        if not (is_point_name(a) and is_point_name(b) and is_point_name(c)):
            raise ValueError(f"Invalid points in angle term: {term}")
        return ["angle", a, b, c]
    # Recursive fallback for future compound terms
    return [canonicalize_term(x) for x in term]

def canonicalize_fact(fact: Fact) -> Fact:
    """Canonicalize a geometry fact by normalizing its predicate and arguments."""
    if not isinstance(fact, dict):
        raise ValueError(f"Fact must be a dictionary, got: {type(fact)}")
    pred = fact.get("pred")
    args = fact.get("args")
    if not isinstance(pred, str) or not pred.strip():
        raise ValueError(f"Fact 'pred' must be a non-empty string, got: {pred}")
    if not isinstance(args, list):
        raise ValueError(f"Fact 'args' must be a list, got: {type(args)}")
    args = [canonicalize_term(arg) for arg in args]
    if pred == "collinear":
        if len(args) != 3 or not all(is_point_name(x) for x in args):
            raise ValueError(f"Invalid collinear fact: {fact}")
        args = sorted(args)
    elif pred == "between":
        if len(args) != 3 or not all(is_point_name(x) for x in args):
            raise ValueError(f"Invalid between fact: {fact}")
    elif pred == "midpoint":
        if len(args) != 3 or not all(is_point_name(x) for x in args):
            raise ValueError(f"Invalid midpoint fact: {fact}")
        m, a, b = args
        a, b = sort_points(a, b)
        args = [m, a, b]
    elif pred == "equal_length":
        if len(args) != 2 or not all(is_segment_term(x) for x in args):
            raise ValueError(f"Invalid equal_length fact: {fact}")
        s1, s2 = sort_terms_lexicographically(args[0], args[1])
        args = [s1, s2]
    elif pred in {"parallel", "perpendicular"}:
        if len(args) != 2 or not all(is_line_term(x) for x in args):
            raise ValueError(f"Invalid {pred} fact: {fact}")
        l1, l2 = sort_terms_lexicographically(args[0], args[1])
        args = [l1, l2]
    elif pred == "equal_angle":
        if len(args) != 2 or not all(is_angle_term(x) for x in args):
            raise ValueError(f"Invalid equal_angle fact: {fact}")
        a1, a2 = sort_terms_lexicographically(args[0], args[1])
        args = [a1, a2]
    return {"pred": pred, "args": args}

def canonicalize_goal(goal: Fact) -> Fact:
    return canonicalize_fact(goal)

def canonicalize_problem(problem: Problem) -> Problem:
    normalized = deepcopy(problem) 
    normalized["entities"]["points"] = sorted(set(normalized["entities"]["points"])) 
    normalized["entities"]["points"] = sorted(set(normalized["entities"]["points"]))
    
    if "lines" in normalized["entities"]:
        normalized["entities"]["lines"] = sorted(set(normalized["entities"]["lines"]))
    if "circles" in normalized["entities"]:
        normalized["entities"]["circles"] = sorted(set(normalized["entities"]["circles"]))
        
    canon_givens = [canonicalize_fact(f) for f in normalized["givens"]] 
    normalized["givens"] = sort_facts(deduplicate_facts(canon_givens)) 
    normalized["goal"] = canonicalize_goal(normalized["goal"]) 
    if normalized.get("proof_trace") is not None: 
        for step in normalized["proof_trace"]: 
            if step.get("facts_used") is not None: 
                used = [canonicalize_fact(f) for f in step["facts_used"]] 
                step["facts_used"] = sort_facts(deduplicate_facts(used)) 
            if step.get("derived") is not None: 
                step["derived"] = canonicalize_fact(step["derived"]) 
    return normalized

def is_point_name(value: Any) -> bool: 
    return isinstance(value, str) and len(value.strip()) > 0
def sort_points(a: str, b: str) -> Tuple[str, str]: 
    return tuple(sorted((a, b)))
def sort_terms_lexicographically(term1: Any, term2: Any) -> Tuple[Any, Any]: 
    return tuple(sorted((term1, term2), key=repr))
def is_line_term(term: Any) -> bool: 
    return isinstance(term, list) and len(term) == 3 and term[0] == "line"
def is_segment_term(term: Any) -> bool: 
    return isinstance(term, list) and len(term) == 3 and term[0] == "segment"
def is_angle_term(term: Any) -> bool: 
    return isinstance(term, list) and len(term) == 4 and term[0] == "angle"
def fact_key(fact: Fact) -> str: 
    return repr(fact)
def sort_facts(facts: List[Fact]) -> List[Fact]: 
    return sorted(facts, key=fact_key)
def deduplicate_facts(facts: List[Fact]) -> List[Fact]:
    seen = set()
    result: List[Fact] = []
    for fact in facts:
        key = fact_key(fact)
        if key not in seen:
            seen.add(key)
            result.append(fact)
    return result