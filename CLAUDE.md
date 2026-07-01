# CLAUDE.md — Project Guide for AI Coding Assistants

## Project Overview

**routee.powertrain** is a Python package for predicting vehicle energy consumption over road network links. It ships pre-trained mesoscopic vehicle energy models (ICE, HEV, BEV, PHEV, heavy-duty) and supports training custom models from drive-cycle data.

- **Maintainer**: National Laboratory of the Rockies
- **License**: BSD 3-Clause
- **Python**: >=3.10, <3.14
- **Build system**: Hatchling (`hatch build`)
- **Package manager**: Pixi (preferred) or pip
- **Core deps**: pandas, numpy, onnx, onnxruntime, boto3, rapidfuzz

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

## Architecture

Package source lives under `routee/powertrain/`.

### Core layers

- **`core/`** — Central data types: `Model` (with a computed `key` → `ModelKey`), `ModelConfig`, `FeatureSet`, `DataColumn`, `TargetSet`, `Constraints`, `PowertrainType` (enum), `Metadata`, `PredictMethod`, `Drivetrain`, `FuelType`, `Year` (type alias)
- **`estimators/`** — `Estimator` ABC + `InputSpec` dataclass, with implementations:
  - `ONNXEstimator` — wraps any ONNX model via onnxruntime. Handles both plain tabular input (`(N, F)`) and windowed sequence input (`(N, lookback, F)`) driven by an `InputSpec` embedded in the model's ONNX `metadata_props` (keys `routee_lookback`, `routee_grouping_column`, `routee_pad_strategy`).
  - `NGBoostEstimator` — wraps NGBoost models (joblib + base64); emits both a point prediction and a per-row standard deviation column.
- **`trainers/`** — `Trainer` ABC with implementations:
  - `SklearnRandomForestTrainer` → produces `ONNXEstimator` (converts via skl2onnx)
  - `CNNTrainer` → produces `ONNXEstimator` with a non-default `InputSpec` (1D CNN exported via `torch.onnx.export`); requires a `grouping_column` (e.g. `route_id`) to bucket sequences.
  - `NGBoostTrainer` → produces `NGBoostEstimator`
- **`io/`** — `load_model()`, `list_available_models()`, `query_available_models()`, `load_sample_route()`, plus archive helpers (`load_model_from_path`, `save_model_directory`, `save_archive`, `save_tar_archive`) and the `to_lookup_table` helper backing `Model.to_lookup_table()`.
- **`validation/`** — `ModelErrors`, `compute_errors()`, `visualize_features()`, `contour_plot()`
- **`resources/`** — Bundled pre-trained models (`bundled_registry/v2/...`) and sample route data
- **`registry/`** — Pluggable model discovery and retrieval system with multiple backends:
  - `ModelRegistry` (ABC) — Interface with `query()`, `load()`, `list_models()`, `get_metadata()`
  - `S3Registry` — Fetches models from a public S3 bucket (`routeecore-bucket`). Uses optional `index.json` for fast queries and hierarchical prefix walking for filtered searches
  - `LocalRegistry` — Reads models from a local directory tree using glob-based scanning
  - `ModelId` — Identifier with five segments: `make/model/year/config_slug/v<N>` (e.g. `toyota/camry_4cyl_2wd/2016/rf_c3326385/v1`). It composes a `ModelKey` (the version-less identity) plus a registry-assigned `version`. `config_slug` is **derived** from metadata (`registry/slug.py:derive_config_slug`) as `<arch_short>_<variant?>_<feature_hash>` — not user-supplied — so `make/model/year/config_slug` are all pure functions of a model's metadata; only `version` is a registry coordinate. `ModelId.from_metadata(metadata, version)` / `ModelId.from_key(key, version)` mint one.
  - `ModelKey` — The version-less identity (`make/model/year/config_slug`), frozen/hashable. `Model.key` (`ModelKey.from_metadata`) exposes it on any trained or loaded model; `filter_models` groups versions by it. Registry `load()` re-derives the slug and **raises on any mismatch** with the on-disk path (`assert_metadata_matches_id`).
  - `ModelInfo` — Lightweight summary returned by `query()` with metadata but no binary data
  - `get_default_registry()` — Factory that selects backend based on `ROUTEE_REGISTRY_BACKEND` env var (`"s3"` or `"local"`, default `"s3"`)
  - `filter_models()` / `latest_model_ids()` — Filtering supports exact + fuzzy matching (via `rapidfuzz`) on `make`, `model`, `year`, `config_slug`, `feature_names`, `powertrain_type`, `fuel_type`, `drivetrain`, `engine`, `trim`, `version`, plus optional `custom_filters` callables.

### Registry

The Registry system abstracts model discovery and retrieval, allowing pre-trained models to be served from S3 or the local filesystem with an identical API.

- **Directory/bucket layout**: `<root>/<schema_version>/<make>/<model>/<year>/<config_slug>/v<N>/` containing `metadata.json` + a binary estimator file (e.g. `model.onnx` or a `.joblib` blob)
- **Bundled models**: `routee/powertrain/resources/bundled_registry/v2/` (currently 2016 Toyota Camry 4cyl 2WD and 2017 Chevy Bolt, both `rf_c3326385/v1` — the derived slug for a random forest over speed & grade)
- **Public entry points** (in `io/load.py`):
  - `list_available_models(registry=None, version_strategy="latest")` — Returns all `ModelId`s. `version_strategy="all"` returns every version; default keeps only the latest per `(make, model, year, config_slug)` group.
  - `query_available_models(...)` — Filtered search returning `ModelInfo` objects; supports fuzzy matching (`fuzzy=True`, `fuzzy_threshold=80`) and the same filter set as `filter_models()`.
  - `load_model(name_or_path, registry=None)` — Loads from a local path/zip/tar if it exists, otherwise resolves the string as a `ModelId` (`v<N>` segment optional → latest) and fetches from the registry.
- **Environment variables**:
  - `ROUTEE_REGISTRY_BACKEND` — `"s3"` (default) or `"local"`
  - `ROUTEE_SCHEMA_VERSION` — default `"v2"`
  - `ROUTEE_S3_BUCKET` — default `routeecore-bucket`
  - `ROUTEE_S3_REGION` — default `us-west-2`
  - `ROUTEE_S3_ROOT_PREFIX` — default `routee-powertrain-model-library`
  - `ROUTEE_LOCAL_REGISTRY_ROOT` — local directory root (defaults to bundled registry)

### Key abstractions

- **`Model`** — Main user-facing object. Holds a single `estimator: Estimator` + `metadata: Metadata`; `metadata.errors` exposes the `ModelErrors`. Supports `from_file`, `to_file` (directory / `.zip` / `.tar.gz` auto-detected from suffix), `predict`, `visualize_features`, `contour`, `to_lookup_table`.
- **`ModelConfig`** — Vehicle description, powertrain type, feature set, distance column, energy target(s), predict method, plus optional `mass_lbs`, `fuel_type`, `drivetrain`, `engine`, `trim`, `variant` (short label — e.g. `steady`/`warmup` — folded into the derived `config_slug` to distinguish configs sharing architecture + feature set), `apply_real_world_adjustment` (default `True`).
- **`Estimator`** (ABC) — Stateless prediction interface. All implementations provide `predict()`, `from_dict()`/`to_dict()`, `from_file()`/`to_file()`, and expose an `input_spec: InputSpec` (lookback / grouping_column / pad_strategy) describing whether they want pointwise or windowed inputs.
- **`Trainer`** (ABC) — `train()` splits data, delegates to `inner_train()`, computes errors, returns a `Model`. Each subclass sets an `architecture_tag` (e.g. `random_forest`, `ngboost`, `cnn`) used in `metadata.json` and registry filtering.
- **`FeatureSetId`** — String alias (`core/features.py`) holding the sorted-`&`-joined feature column names. Used as a hashable feature-set fingerprint for registry queries; v2 models hold a single estimator per file, so there's no runtime feature-set dispatch inside `Model`.

### Predict methods

- **RATE** — Train on `energy/distance` (rate), predict rate then multiply by distance
- **RAW** — Train on raw energy with distance as a feature, predict total energy directly
- After the estimator runs, `Model.predict` multiplies each target by `ADJUSTMENT_FACTORS[powertrain_type]` unless `ModelConfig.apply_real_world_adjustment=False`.

### Versioning & schema

- `metadata.json` carries `schema_version: 2`; v1 model files raise on load.
- `ModelId.version` is an int (`v<N>` in paths). The default `version_strategy="latest"` collapses to the highest version per `(make, model, year, config_slug)` group; pass `"all"` to see every version.
- The package version is tracked in `routee/powertrain/__about__.py` and written into `metadata.routee_version` automatically at save time.

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
- Absolute imports within the package: `from routee.powertrain.core.features import ...`

## Linting & Formatting Config

- **Ruff**: line-length 88, indent 4, rules `E4, E7, E9, F`. Includes `routee/**/*.py`, `tests/*.py`.
- **Mypy**: `ignore_missing_imports = true`, `namespace_packages = true`, `explicit_package_bases = true`. Excludes `docs/`, `build/`, `dist/`, `py-notebooks/`.

## Test Structure

- **Framework**: `unittest.TestCase` (not pytest-style assertions, though pytest can run them)
- **Files**:
  - `tests/test_train_estimate_pipeline.py` — Train → predict → serialize → deserialize round-trip for sklearn → ONNX and NGBoost
  - `tests/test_cnn_pipeline.py` — CNN → ONNX round-trip (requires torch; skipped otherwise)
  - `tests/test_archive.py` — Directory / `.zip` / `.tar.gz` serialization round-trip
  - `tests/test_registry.py` — `ModelId` parsing, `LocalRegistry` filtering, `ModelInfo` round-trip
  - `tests/test_s3_registry.py` — `S3Registry` boto3 integration, key parsing
  - `tests/test_s3_index.py` — S3 `index.json` build / parse
  - `tests/test_year_range.py` — `Year` type (int or `tuple[int, int]`) parsing and serialization
  - `tests/test_to_lookup.py` — Lookup table generation with varying feature counts
  - `tests/mock_resources.py` — Helper functions for mock data generation
- **Test data**: `tests/routee-powertrain-test-data/sample_train_data.csv`
- **Notes**: Tests write temp files to `tmp/` and clean up.

## CI/CD

- **`.github/workflows/test.yaml`** — Runs on push to `main` and PRs. Matrix: Python 3.10, 3.11, 3.12, 3.13. Steps: `pip install ".[dev]"` → `mypy .` → `ruff check` → `ruff format --check` → `dprint check` → `python -m unittest discover tests/`
- **`.github/workflows/publish-pypi.yaml`** — On GitHub release: `hatch build` → publish to PyPI
- **`.github/workflows/deploy-docs.yaml`** — On push to `main` (docs/ path): build and deploy Jupyter Book to GitHub Pages

## Optional Dependency Groups

| Extra     | Purpose                                                   |
| --------- | --------------------------------------------------------- |
| `scikit`  | sklearn + skl2onnx for training                           |
| `ngboost` | NGBoost for probabilistic training                        |
| `pytorch` | torch + onnxscript for the CNN trainer                    |
| `plot`    | matplotlib for visualization                              |
| `dev`     | All of the above + pytest, mypy, ruff, jupyter-book, etc. |

Install with: `pip install -e ".[scikit]"`, `pip install -e ".[ngboost,plot]"`, etc.
