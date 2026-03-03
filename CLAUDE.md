# CLAUDE.md — Project Guide for AI Coding Assistants

## Project Overview

**nrel.routee.powertrain** is a Python package for predicting vehicle energy consumption over road network links. It ships pre-trained mesoscopic vehicle energy models (ICE, HEV, BEV, PHEV, heavy-duty) and supports training custom models from drive-cycle data.

- **Maintainer**: National Laboratory of the Rockies 
- **License**: BSD 3-Clause
- **Python**: >=3.10, <3.13
- **Build system**: Hatchling (`hatch build`)
- **Package manager**: Pixi (preferred) or pip
- **Core deps**: pandas, numpy, onnx, onnxruntime

## Quick Commands

| Action            | Command                                                 |
| ----------------- | ------------------------------------------------------- |
| Install (dev)     | `pip install -e ".[dev]"` or `pixi install`             |
| Test              | `pytest tests/` or `python -m unittest discover tests/` |
| Lint              | `ruff check`                                            |
| Lint (fix)        | `ruff check --fix`                                      |
| Format            | `ruff format`                                           |
| Format (check)    | `ruff format --check`                                   |
| Type check        | `mypy .`                                                |
| Build             | `hatch build`                                           |
| Docs              | `jupyter-book build docs`                               |
| All checks (Pixi) | `pixi run check` (fmt + lint + typing + test)           |
| CI checks (Pixi)  | `pixi run ci` (fmt_check + lint_check + typing + test)  |
| Rust component    | `cd rust && maturin develop --release`                  |

## Architecture

Package source lives under `nrel/routee/powertrain/`.

### Core layers

- **`core/`** — Central data types: `Model`, `ModelConfig`, `FeatureSet`, `DataColumn`, `TargetSet`, `Constraints`, `PowertrainType` (enum), `Metadata`
- **`estimators/`** — `Estimator` ABC with implementations:
  - `ONNXEstimator` — wraps ONNX models via onnxruntime
  - `SmartCoreEstimator` — wraps Rust SmartCore random forest (PyO3)
  - `NGBoostEstimator` — wraps NGBoost models (joblib + base64)
  - `SKLearnEstimator` — wraps sklearn RandomForestRegressor (legacy, JSON/pickle)
- **`trainers/`** — `Trainer` ABC with implementations:
  - `SklearnRandomForestTrainer` → produces `ONNXEstimator` (converts via skl2onnx)
  - `SmartcoreRandomForestTrainer` → produces `SmartCoreEstimator`
  - `NGBoostTrainer` → produces `NGBoostEstimator`
- **`io/`** — `load_model()`, `list_available_models()`, `load_sample_route()`, `to_lookup_table()`
- **`validation/`** — `ModelErrors`, `compute_errors()`, `visualize_features()`, `contour_plot()`
- **`resources/`** — Bundled pre-trained models and sample route data
- **`rust/`** (top-level) — PyO3/maturin Rust extension (`powertrain_rust`) providing SmartCore random forest. Build with `cd rust && maturin develop --release`.

### Key abstractions

- **`Model`** — Main user-facing object. Holds `Dict[FeatureSetId, Estimator]` + `Metadata` + `ModelErrors`. Supports `from_file`, `to_file`, `predict`, `visualize_features`, `contour`, `to_lookup_table`.
- **`ModelConfig`** — Vehicle description, powertrain type, feature sets, distance column, energy targets, predict method.
- **`Estimator`** (ABC) — Stateless prediction interface. All implementations provide `predict()`, `from_dict()`/`to_dict()`, `from_file()`/`to_file()`.
- **`Trainer`** (ABC) — `train()` splits data, delegates to `inner_train()`, computes errors, returns a `Model`.
- **`FeatureSetId`** — String key formed by sorting and `&`-joining feature column names. Models auto-select the right estimator at prediction time based on available columns.

### Predict methods

- **RATE** — Train on `energy/distance` (rate), predict rate then multiply by distance
- **RAW** — Train on raw energy with distance as a feature, predict total energy directly

## Coding Conventions

- `from __future__ import annotations` in all source files
- `@dataclass` for core types; `Enum` for `PowertrainType`, `PredictMethod`
- ABC + `@abstractmethod` for `Estimator` and `Trainer` interfaces
- Serializable types implement `to_dict()`/`from_dict()` and `to_file()`/`from_file()`
- Google-style docstrings (Args/Returns sections)
- Type hints throughout; `py.typed` marker present
- `logging.getLogger(__name__)` module-level logger pattern
- snake_case functions/variables, PascalCase classes, UPPER_SNAKE_CASE constants
- `TYPE_CHECKING` guard for circular import avoidance
- Absolute imports within the package: `from nrel.routee.powertrain.core.features import ...`

## Linting & Formatting Config

- **Ruff**: line-length 88, indent 4, rules `E4, E7, E9, F`. Includes `nrel/**/*.py`, `tests/*.py`.
- **Mypy**: `ignore_missing_imports = true`, `namespace_packages = true`, `explicit_package_bases = true`. Excludes `docs/`, `build/`, `dist/`, `py-notebooks/`.

## Test Structure

- **Framework**: `unittest.TestCase` (not pytest-style assertions, though pytest can run them)
- **Files**:
  - `tests/test_train_estimate_pipeline.py` — Train → predict → serialize → deserialize round-trip for sklearn, NGBoost, SmartCore
  - `tests/test_model_library.py` — Loads all catalog models (marked slow, skipped by default)
  - `tests/test_to_lookup.py` — Lookup table generation with varying feature counts
  - `tests/mock_resources.py` — Helper functions for mock data generation
- **Test data**: `tests/routee-powertrain-test-data/sample_train_data.csv`
- **Notes**: SmartCore tests are `@skip`-ped (require Rust build). Tests write temp files to `tmp/` and clean up.

## CI/CD

- **`.github/workflows/test.yaml`** — Runs on push to `main` and PRs. Matrix: Python 3.10, 3.11, 3.12. Steps: `pip install ".[dev]"` → `mypy .` → `ruff check` → `ruff format --check` → `python -m unittest discover tests/`
- **`.github/workflows/publish-pypi.yaml`** — On GitHub release: `hatch build` → publish to PyPI
- **`.github/workflows/deploy-docs.yaml`** — On push to `main` (docs/ path): build and deploy Jupyter Book to GitHub Pages

## Optional Dependency Groups

| Extra     | Purpose                                                            |
| --------- | ------------------------------------------------------------------ |
| `scikit`  | sklearn + skl2onnx for training                                    |
| `ngboost` | NGBoost for probabilistic training                                 |
| `plot`    | matplotlib for visualization                                       |
| `dev`     | All of the above + pytest, mypy, ruff, maturin, jupyter-book, etc. |

Install with: `pip install -e ".[scikit]"`, `pip install -e ".[ngboost,plot]"`, etc.
