# Neural-Guided Geometry Reasoner

A research-oriented geometry prover that combines LLM formalization with deterministic
SWI-Prolog verification.

The current CLI accepts a geometry statement or local image, asks an OpenAI model for
schema-constrained geometry JSON, canonicalizes the facts, and runs a bounded proof loop.
The model may translate the problem; Prolog decides which mathematical steps are valid.

```text
text or image
    -> structured geometry JSON
    -> Pydantic validation and canonicalization
    -> bounded proof search
    -> SWI-Prolog verification
    -> verified proof trace
```

## Prerequisites

- Python 3.12
- `uv` for Python and dependency management
- SWI-Prolog 10+, with `swipl` on `PATH`
- An OpenAI API key for hosted formalization

Check the important executables:

```powershell
uv --version
swipl --version
```

The current tested SWI-Prolog version is 10.0.0. The project currently uses
`library(http/json)` for JSON communication.

## Quick start

From the repository root in PowerShell:

```powershell
uv sync
```

`uv sync` creates and maintains the project virtual environment using the pinned Python 3.12
version. Activation is not required; run project commands through `uv run`.

Copy the environment template:

```powershell
Copy-Item .env.example .env
```

Then put your real key in `.env`:

```dotenv
OPENAI_API_KEY=replace_with_your_key
OPENAI_MODEL=gpt-5.6-luna
```

The legacy name `OPEN_AI_KEY` is also accepted. `.env` is ignored by Git.

Run a text problem:

```powershell
uv run python -m demo.cli --text "D is the midpoint of AB. Prove that A, D, and B are collinear."
```

Run a local image:

```powershell
uv run python -m demo.cli --image path\to\problem.png
```

Request machine-readable output:

```powershell
uv run python -m demo.cli --text "B lies between A and C. Prove A, B, and C are collinear." --json
```

Use `--max-steps N` to change the default 32-step proof bound.

## What works today

- Strict schemas for eleven supported geometry predicates.
- Text and single-image formalization using `gpt-5.6-luna` by default.
- Explicit unsupported-input results rather than invented facts.
- Canonical fact ordering and duplicate removal.
- JSON-only Python/SWI-Prolog communication.
- Deterministic bounded proof search.
- Human-readable and JSON CLI output.
- Offline tests with real SWI-Prolog integration.
- Opt-in text and image API smoke tests.
- Reproducible evaluation over ten text and two image cases.

The initial hosted-model baseline produced valid schema output for 12/12 cases and exact,
Prolog-verified results for 9/10 text cases and 2/2 image cases. API-backed results may
vary.

## Supported geometry

- `triangle`
- `collinear`
- `between`
- `midpoint`
- `equal_length`
- `parallel`
- `perpendicular`
- `equal_angle`
- `angle_bisector`
- `on_perp_bisector`
- `is_isosceles`

## Testing

Run the deterministic offline suite:

```powershell
uv run --group dev python -m pytest -q -m "not live_api"
```

These tests require a real `swipl` installation but no OpenAI key.

Run the two billable live smoke tests only when wanted:

```powershell
$env:RUN_OPENAI_LIVE_TESTS="1"
uv run --group dev python -m pytest tests\test_live_api.py -q
```

## Evaluation

Run the hosted-model evaluator:

```powershell
uv run python -m geometry_reasoner.evaluate
```

It reports:

- Formalization success rate.
- Exact, Prolog-verified proof-completion rate.
- Mean proof steps explored.
- Per-case model, latency, and token metadata.

The detailed report is written to ignored `artifacts/eval.json`. Gold JSON and proof
traces are used only for scoring after formalization; they are not placed in the prompt.

## Privacy

- Hosted text/image inputs are sent to the configured OpenAI model.
- Requests use `store=False`.
- Images are encoded in memory; Base64 payloads are not saved to evaluation artifacts.
- API keys are read from environment variables and are not logged or written to output.
- Local `.env`, evaluation artifacts, and downloaded model files must remain untracked.

## CLI exit codes

- `0`: goal verified or already satisfied
- `2`: invalid input or configuration
- `3`: API or unsupported-input failure
- `4`: incomplete proof
- `5`: SWI-Prolog infrastructure failure

## Project structure

```text
geometry_reasoner/
    schema.py           validated geometry data models
    formalizer.py       hosted LLM adapter
    canonicalize.py     deterministic fact normalization
    loader.py           dataset loading
    prolog_bridge.py    JSON subprocess boundary
    reasoner.py         bounded Python proof loop
    evaluate.py         text/image evaluation

prolog/
    rules.pl            canonicalization and geometry rules
    query.pl            goal checks and candidate generation
    json_bridge.pl      JSON stdin/stdout bridge

demo/cli.py             end-to-end command-line interface
data/problems.json      baseline geometry fixtures
tests/                  offline, integration, and opt-in live tests
```

## Planned finish line

The remaining scope is intentionally limited:

1. Abstract formalization so provider/model are runtime CLI choices.
2. Add one local text formalizer through `llama.cpp`, then run one measured fine-tune.
3. Make proof traces premise-complete.
4. Keep one persistent SWI-Prolog process per problem.
5. Add bounded Prolog-native recursive search.
6. Train and persist one XGBoost candidate ranker.
7. Compare Prolog-only, first-candidate, and XGBoost-guided search.
8. Record the experiments in `RESEARCH_NOTES.md`, document the result, and stop.

Reinforcement learning, a web UI, retrieval, distributed training, and production
serving are outside the completion scope.

Dependencies are declared in `pyproject.toml` and fully resolved in the committed `uv.lock`.
Use `uv sync` to create a development environment, `uv run` to execute project commands,
`uv add <package>` to add a runtime dependency, and `uv add --group dev <package>` for a
development-only dependency. Commit both `pyproject.toml` and `uv.lock` whenever dependencies
change.
