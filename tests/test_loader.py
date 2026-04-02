import json
import pytest
from geometry_reasoner.loader import load_problems, parse_problem_file, get_problem_by_id, list_problem_ids


def _make_minimal_problem(problem_id: str = "p1") -> dict:
    return {
        "id": problem_id,
        "entities": {"points": ["B", "A"]},
        "givens": [{"pred": "collinear", "args": ["A", "B", "C"]}],
        "goal": {"pred": "between", "args": ["A", "B", "C"]},
    }


def test_parse_problem_file_as_list_can_parse():
    raw_text = json.dumps([_make_minimal_problem()])
    problems = parse_problem_file(raw_text)
    assert len(problems) == 1
    assert problems[0].id == "p1"


def test_parse_problem_file_as_wrapped_object_can_parse():
    raw_text = json.dumps({"problems": [_make_minimal_problem("p2")]})
    problems = parse_problem_file(raw_text)
    assert len(problems) == 1
    assert problems[0].id == "p2"


def test_parse_problem_file_invalid_json_raises_value_error():
    with pytest.raises(ValueError, match="Invalid problems JSON"):
        parse_problem_file("{ invalid json }")


def test_load_problems_missing_file_raises_file_not_found_error(tmp_path):
    missing_file = tmp_path / "not_here.json"
    with pytest.raises(FileNotFoundError):
        load_problems(missing_file)


def test_load_problems_duplicate_id_raises_value_error(tmp_path):
    problems = [_make_minimal_problem("dup"), _make_minimal_problem("dup")]
    file_path = tmp_path / "problems.json"
    file_path.write_text(json.dumps(problems), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate problem id"):
        load_problems(file_path)


def test_load_problems_canonicalizes_id_and_returns_dict(tmp_path):
    problems = [_make_minimal_problem("p5")]
    file_path = tmp_path / "problems.json"
    file_path.write_text(json.dumps(problems), encoding="utf-8")
    loaded = load_problems(file_path)
    assert list(loaded) == ["p5"]
    problem = loaded["p5"]
    assert problem["id"] == "p5"
    assert problem["entities"]["points"] == ["A", "B"]


def test_get_problem_by_id_success(tmp_path):
    payload = {_make_minimal_problem("p3")["id"]: _make_minimal_problem("p3")}  # type: ignore
    problem = get_problem_by_id(payload, "p3")
    assert problem["id"] == "p3"


def test_get_problem_by_id_not_found_raises_value_error():
    problems = {"p1": _make_minimal_problem("p1")}
    with pytest.raises(ValueError, match="Problem with id 'p2' not found"):
        get_problem_by_id(problems, "p2")


def test_list_problem_ids_returns_sorted_ids():
    problems = {"z": {}, "a": {}}
    assert list_problem_ids(problems) == ["a", "z"]

