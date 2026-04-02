import pytest
from geometry_reasoner.canonicalize import canonicalize_term, canonicalize_fact, canonicalize_problem


def test_canonicalize_term():
    assert canonicalize_term("A") == "A"
    assert canonicalize_term(["segment", "B", "A"]) == ["segment", "A", "B"]
    assert canonicalize_term(["angle", "C", "B", "A"]) == ["angle", "C", "B", "A"]

    with pytest.raises(ValueError):
        canonicalize_term(["segment", "A"])

    with pytest.raises(ValueError):
        canonicalize_term(["segment", "A", 123])


def test_canonicalize_fact():
    fact = {"pred": "collinear", "args": ["C", "A", "B"]}
    expected = {"pred": "collinear", "args": ["A", "B", "C"]}
    assert canonicalize_fact(fact) == expected

    fact = {"pred": "between", "args": ["B", "A", "C"]}
    expected = {"pred": "between", "args": ["B", "A", "C"]}
    assert canonicalize_fact(fact) == expected

    fact = {"pred": "midpoint", "args": ["M", "B", "A"]}
    expected = {"pred": "midpoint", "args": ["M", "A", "B"]}
    assert canonicalize_fact(fact) == expected

    with pytest.raises(ValueError):
        canonicalize_fact({"pred": "", "args": []})

    with pytest.raises(ValueError):
        canonicalize_fact({"pred": "collinear", "args": ["A", 123, "C"]})


def test_canonicalize_problem():
    

    raw_problem = {
        "id": "prob1",
        "entities": {"points": ["C", "A", "B", "A"]},
        "givens": [
            {"pred": "collinear", "args": ["C", "A", "B"]},
            {"pred": "between", "args": ["B", "A", "C"]},
            {"pred": "collinear", "args": ["A", "B", "C"]},
        ],
        "goal": {"pred": "midpoint", "args": ["M", "B", "A"]},
    }

    result = canonicalize_problem(raw_problem)

    assert result["id"] == "prob1"
    assert result["entities"]["points"] == ["A", "B", "C"]
    assert result["goal"] == {"pred": "midpoint", "args": ["M", "A", "B"]}

    assert {"pred": "collinear", "args": ["A", "B", "C"]} in result["givens"]
    assert {"pred": "between", "args": ["B", "A", "C"]} in result["givens"]
    assert len(result["givens"]) == 2