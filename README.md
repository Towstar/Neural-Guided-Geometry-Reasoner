# Neural Guided Geometry Reasoner

A neuro-symbolic geometry system that combines **Prolog-based geometric reasoning** with **Python-based candidate ranking** to guide proof search.

The project is designed around a simple idea: symbolic rules guarantee correctness, while a learned or heuristic ranker helps decide which valid inference step to try next. The result is a small but interpretable reasoning engine for structured Euclidean geometry problems.

---

## Description

This project explores a hybrid approach to automated geometry reasoning.

- **Prolog** is used to represent geometric facts and inference rules
- **Python** is used to load problems, manage proof state, extract features, and rank candidate inference steps
- A lightweight **CLI** and optional **demo app** are used to inspect and visualize the reasoning process

The main goal is not just to prove geometry statements, but to do so in a way that is:

- interpretable
- modular
- mathematically structured
- extensible toward learned guidance

---

## Current Scope

The MVP focuses on **synthetic Euclidean geometry problems** in a structured symbolic format.

Initial supported concepts include:

- triangles
- collinearity
- midpoint
- equal lengths
- equal angles
- parallel lines
- angle bisectors
- perpendicular bisectors
- isosceles triangle reasoning

The system currently assumes problems are provided in a structured JSON format rather than natural language.

---

## Prerequisites

Before setting up this project, ensure you have the following installed:

- **Python 3.10+**
- **SWI-Prolog**
- **pip** for Python package installation

Optional:

- **Streamlit** if using the demo app as a small web interface
- **pytest** for running tests

---

## Project Structure

```text
/
├── Data/                # Geometry problems and proof/ranking traces
│   ├── problems.json
│   └── traces.json
├── Demo/                # CLI and optional interactive demo
│   ├── app.py
│   └── cli.py
├── Prolog/              # Prolog rules and query entry points
│   ├── facts.pl
│   ├── query.pl
│   └── rules.pl
├── Python/              # Python loaders, reasoning loop, features, and ranker
│   ├── features.py
│   ├── ranker.py
│   └── train.py
├── README.md
└── ...
```
