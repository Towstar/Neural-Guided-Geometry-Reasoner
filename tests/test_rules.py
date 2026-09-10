from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / "prolog" / "rules.pl"


def run_swipl(goal: str) -> str:
    result = subprocess.run(
        ["swipl", "-q", "-s", str(RULES), "-g", goal],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.strip()


def test_canonical_fact_returns_single_canonical_collinear_fact():
    output = run_swipl(
        "findall(C, rules:canonical_fact(collinear(a,d,b), C), Cs), "
        "writeln(Cs), halt."
    )

    assert output == "[collinear(a,b,d)]"


def test_canonical_fact_returns_single_canonical_midpoint_fact():
    output = run_swipl(
        "findall(C, rules:canonical_fact(midpoint(d,c,b), C), Cs), "
        "writeln(Cs), halt."
    )

    assert output == "[midpoint(d,b,c)]"


def test_midpoint_converse_matches_canonicalized_facts():
    output = run_swipl(
        "once(rules:derive("
        "[equal_length(segment(a,d),segment(b,d)), collinear(a,b,d)], "
        "Derived, midpoint_converse)), "
        "writeln(Derived), halt."
    )

    assert output == "midpoint(d,a,b)"
