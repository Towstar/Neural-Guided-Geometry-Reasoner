# Neural-Guided Geometry Reasoner

A single, staged capstone project that combines a small language model, retrieval, neural proof-search guidance, and Prolog verification for Euclidean geometry.

The project deliberately absorbs the planned “modern LLM architecture learning lab” into the geometry reasoner. The transformer implementation is not a second project or a separate portfolio deliverable; it is the local model used to study and support the geometry system. This makes the overall scope larger while reusing the same tokenizer, training loop, evaluation harness, data pipeline, and experiment infrastructure.

## Project thesis

The language model may parse, retrieve, rank, and explain. It may not certify a mathematical step.

```text
natural-language problem
        ↓
small local decoder-only Transformer
        ↓
schema-validated canonical geometry JSON
        ↓
Python proof state + retrieved rules/examples
        ↓
neural or heuristic candidate ranking
        ↓
Prolog verification
        ↓
verified proof trace
        ↓
trace-grounded explanation
```

The central research question is:

> Can a small learned model make symbolic geometry proof search more efficient or accessible without weakening the symbolic verifier?

## Unified scope

The project has two connected tracks inside one repository:

1. **Model-building track:** implement and train a small decoder-only Transformer from random initialization to understand tokenization, attention, RoPE, RMSNorm, GQA, SwiGLU, optimization, checkpointing, sampling, and KV caching.
2. **Reasoning track:** use the resulting local-model and data infrastructure for natural-language formalization, retrieval-guided proof search, neural candidate ranking, and explanation of verified proofs.

The model-building track is intentionally educational rather than production-oriented. The goal is a functioning, inspectable model and reproducible experiments, not a competitive foundation model.

## Milestones

### Milestone 1 — Deterministic symbolic core

```text
problem JSON
→ Pydantic validation
→ canonical facts
→ Prolog rules
→ Python proof loop
→ deterministic candidate selection
→ verified multistep ProofResult
```

The proof loop must support proof-state transitions, duplicate prevention, goal checks, bounded termination, and clear Prolog error handling before any model is used to guide it.

### Milestone 2 — Learning lab inside the reasoner

Build a small decoder-only Transformer and its training harness. The minimum educational architecture is:

- BPE tokenizer and chat/message formatting
- pre-normalized decoder blocks
- RMSNorm, RoPE, causal attention, and GQA
- SwiGLU feed-forward layers
- AdamW, learning-rate scheduling, gradient clipping, and checkpoint resume
- sampling and KV-cache inference

An optional sparse MoE feed-forward block is a stretch experiment, evaluated with routing statistics and a dense-vs-MoE comparison. Distributed training, custom CUDA kernels, and production serving are explicitly outside this milestone.

### Milestone 3 — Natural language → geometry DSL baseline

Add an open-weight/local model baseline that converts a geometry statement into canonical JSON. Measure JSON validity, schema validity, predicate validity, argument accuracy, goal accuracy, and downstream proof success before fine-tuning.

### Milestone 4 — Fine-tuned formalization

Use LoRA/QLoRA or a similarly lightweight adapter method to train the local model on:

```text
natural-language geometry problem → canonical structured JSON
```

Keep held-out problems and source/proof traces isolated from training data. Compare the base model with the adapted model using the same evaluation harness.

### Milestone 5 — Retrieval-guided proof search

Retrieve only evidence relevant to the current proof state and goal:

- formal theorem and rule descriptions
- predicate documentation
- verified example proofs
- analogous proof states

Every retrieved item must retain a source identifier and provenance. Start with a simple lexical or dense retriever; the value of retrieval must be tested with an ablation rather than assumed.

### Milestone 6 — Learned guidance and explanation

Train or evaluate a ranker over Prolog-generated candidates. Begin with deterministic heuristics, then compare embedding similarity, a learned classifier/cross-encoder, and only if justified a generative ranker. Prolog remains the final verifier.

Once a proof is complete, generate an educational explanation from the verified trace only. Unverified model text must never be inserted into the proof trace.

## Evaluation matrix

The primary comparison is:

| System | Local model | Fine-tuned | Retrieval | Neural guidance | Prolog verification |
|---|---:|---:|---:|---:|---:|
| Deterministic symbolic baseline | No | No | No | No | Yes |
| Local structured-output baseline | Yes | No | No | No | Yes |
| Fine-tuned formalizer | Yes | Yes | No | No | Yes |
| Fine-tuned + retrieval | Yes | Yes | Yes | No | Yes |
| Full neural-guided system | Yes | Yes | Yes | Yes | Yes |

Track structured parsing accuracy, invalid-predicate rate, verified proof-completion rate, proof steps explored, candidate count, search branching factor, latency, memory use, and reproducibility metadata such as seeds, configurations, checkpoints, and test manifests.

## Geometry domain

The initial domain is synthetic Euclidean geometry in a structured symbolic format. Initial concepts include triangles, collinearity, midpoints, equal lengths, equal angles, parallel lines, angle bisectors, perpendicular bisectors, and isosceles-triangle reasoning. Natural-language input is introduced only after the structured proof path is stable.

## Scope boundary

This is one integrated research-engineering project, not a production platform. It does not include distributed training, multi-node inference, custom high-performance kernels, unrestricted autonomous agents, a production vector-database cluster, browser automation, or a general theorem prover.

The completion target is a convincing end-to-end demonstration with objective ablations—not every possible model feature. The symbolic core, formalization baseline, one lightweight fine-tuning experiment, one provenance-aware retrieval experiment, and one proof-guidance comparison are the required outcome. MoE, multimodal input, frontier-model comparison, and advanced serving are optional extensions.

---

## Prerequisites

Before setting up this project, install:

- **Python 3.10+**
- **SWI-Prolog**, with `swipl` available on your `PATH`
- **pip** for Python package installation

## Quick start

Create and activate a virtual environment, then install the project dependencies:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Confirm that SWI-Prolog is available:

```bash
swipl --version
```

## Testing

Run the complete suite from the repository root:

```bash
python -m pytest -q
```

The Prolog integration tests invoke `swipl` directly against `prolog/rules.pl` and `prolog/query.pl`. Python alone is therefore not sufficient to run the full suite.

To run one test module while developing a rule or query, use:

```bash
python -m pytest tests/test_query.py -q
```

## Continuous-integration contract

The first CI workflow should run on Ubuntu with Python 3.12 and `swi-prolog-nox` installed, then execute:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
swipl --version
python -m pytest -q
```

This verifies the real Python-to-Prolog boundary rather than a mocked substitute. Configure it to run for pull requests and pushes to `main` once `.github/workflows/ci.yml` is added.

Optional tools:

- **Streamlit** if using the demo app as a small web interface

---

## Project Structure

```text
/
├── configs/              # Experiment and runtime configuration
├── data/                 # Geometry problems and proof/ranking traces
│   ├── problems.json
│   └── traces.json
├── demo/                 # CLI and optional interactive demo
│   ├── app.py
│   └── cli.py
├── geometry_reasoner/    # Python schema, canonicalization, and reasoning code
│   ├── canonicalize.py
│   ├── loader.py
│   └── prolog_bridge.py
├── prolog/               # Prolog rules and query entry points
│   ├── query.pl
│   └── rules.pl
├── tests/                # Unit and Python-to-Prolog integration tests
├── requirements.txt
├── README.md
└── ...
```

## Current status and next step

The schema, loader, canonicalization layer, Prolog bridge, and initial rule/test fixtures exist. The proof loop, feature extractor, ranker, and training modules are still being built.

The immediate next milestone is to implement and test the deterministic Python proof loop. Do not add fine-tuning or retrieval until a structured problem can produce a verified multistep `ProofResult` or a bounded, explainable failure.
