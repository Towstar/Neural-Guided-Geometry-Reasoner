from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
QUERY = ROOT / "prolog" / "query.pl"


def run_swipl(goal: str) -> str:
    result = subprocess.run(
        ["swipl", "-q", "-s", str(QUERY), "-g", goal],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def test_goal_reached_canonicalizes_input_facts_and_goal():
    output = run_swipl(
        "query:goal_reached([collinear(a,d,b)], collinear(a,b,d)), "
        "writeln(reached), halt."
    )

    assert output == "reached"


def test_all_candidates_returns_unique_canonical_midpoint_derivations():
    output = run_swipl(
        "query:all_candidates([midpoint(d,b,c)], C), writeln(C), halt."
    )

    assert output == (
        "[candidate(midpoint_def_collinear,collinear(b,c,d)),"
        "candidate(midpoint_def_equal_segments,"
        "equal_length(segment(b,d),segment(c,d)))]"
    )


def test_all_candidates_derives_midpoint_converse_from_raw_ordering():
    output = run_swipl(
        "query:all_candidates("
        "[equal_length(segment(a,d),segment(d,b)), collinear(a,d,b)], "
        "C), writeln(C), halt."
    )

    assert output == "[candidate(midpoint_converse,midpoint(d,a,b))]"


def test_all_candidates_derives_equal_length_transitivity():
    output = run_swipl(
        "query:all_candidates("
        "[equal_length(segment(a,b),segment(b,c)), "
        "equal_length(segment(b,c),segment(c,d))], "
        "C), writeln(C), halt."
    )

    assert output == (
        "[candidate(equal_length_transitivity,"
        "equal_length(segment(a,b),segment(c,d)))]"
    )
