# CLAUDE.md — Project Guide for AI Coding Assistants

## General Instructions

Please use ISO 24495-1 Technical English for all text responses and comments, keeping the intent simple and easy to understand.
When writing comments, don't refer to the previous state of the code, just state plainly and simply what needs to be contextualized as the code is in its current state.
Don't over-comment the code, only in places where there is ambiguity or in user facing comments like doc-strings.
Make sure to ask for clarification when instructions are ambiguous to get clarification from the user.

## Project Overview

**routee.powertrain** is a Python package for predicting vehicle energy consumption over road network links. It ships pre-trained mesoscopic vehicle energy models (ICE, HEV, BEV, PHEV, heavy-duty) and supports training custom models from drive-cycle data.

- **Maintainer**: National Laboratory of the Rockies
- **License**: BSD 3-Clause
- **Python**: >=3.10, <3.14
- **Build system**: Hatchling (`hatch build`)
- **Package manager**: Pixi (preferred) or pip
- **Core deps**: pandas, numpy, onnx, onnxruntime, huggingface_hub, rapidfuzz

## Quick Commands

| Action            | Command                                                    |
| ----------------- | ---------------------------------------------------------- |
| Install (dev)     | `pip install -e ".[dev]"` or `pixi install`                |
| Test              | `pytest tests/` or `python -m unittest discover tests/`    |
| Lint              | `ruff check`                                               |
| Lint (fix)        | `ruff check --fix`                                         |
| Format            | `ruff format`                                              |
| Format (check)    | `ruff format --check`                                      |
| Type check        | `mypy .`                                                   |
| Build             | `hatch build`                                              |
| Docs              | `jupyter-book build docs`                                  |
| All checks (Pixi) | `pixi run check` (fmt + lint + typing + test)              |
| Physics report    | `routee-powertrain validate-physics <model...>` or `--all` |
| CI checks (Pixi)  | `pixi run ci` (fmt_check + lint_check + typing + test)     |

## Architecture

Package source lives under `routee/powertrain/`.

### Core layers

- **`core/`** — Central data types: `Model` (with a computed `key` → `ModelKey`), `ModelConfig`, `FeatureSet`, `DataColumn`, `TargetSet`, `Constraints`, `PowertrainType` (enum), `Metadata`, `Provenance` (+ `FastSimSource`/`RealWorldSource`/`LegacySource`, `TrainingConfig`), `PredictMethod`, `Drivetrain`, `FuelType`, `Year` (type alias)
- **`estimators/`** — `Estimator` ABC + `InputSpec`/`ColumnSpec` (pydantic), with implementations:
  - `ONNXEstimator` — wraps any ONNX model via onnxruntime. Handles both plain tabular input (`(N, F)`) and windowed sequence input (`(N, lookback, F)`) driven by an `InputSpec` embedded in the model's ONNX `metadata_props`. Windowing keys (`routee_lookback`, `routee_grouping_column`, `routee_pad_strategy`) are written only when `lookback > 0`; the **input/output contract** keys are written whenever present (including the common tabular case): `routee_input_columns` and `routee_output_columns` (JSON arrays of `{name, units, dtype}` in positional tensor order), `routee_predict_method` (`"rate"`/`"raw"`), and `routee_distance_column`. This makes a bare `.onnx` **self-describing** — a downstream consumer (e.g. routee-compass) can reconstruct the exact positional input order without any out-of-band assumption. The single positional tensor is still named `"input"`.
  - `NGBoostEstimator` — wraps NGBoost models (joblib + base64); emits both a point prediction and a per-row standard deviation column.
- **`trainers/`** — `Trainer` ABC with implementations:
  - `SklearnRandomForestTrainer` → produces `ONNXEstimator` (converts via skl2onnx)
  - `CNNTrainer` → produces `ONNXEstimator` with a non-default `InputSpec` (1D CNN exported via `torch.onnx.export`); requires a `grouping_column` (e.g. `route_id`) to bucket sequences.
  - `NGBoostTrainer` → produces `NGBoostEstimator`
- **`io/`** — `load_model()`, `list_available_models()`, `query_available_models()`, `load_sample_route()`, plus archive helpers (`load_model_from_path`, `save_model_directory`, `save_archive`, `save_tar_archive`) and the `to_lookup_table` helper backing `Model.to_lookup_table()`.
- **`validation/`** — `ModelErrors`, `compute_errors()`, `visualize_features()`, `contour_plot()`, plus `physics.py`: `check_physics()` / `check_model()` return a `PhysicsReport` of physical-plausibility checks (round-trip convexity, monotonicity in grade, fuel ≥ 0, climb floor, absolute bounds) and diagnostics (implied η_drive/η_regen, flat-ground economy, length invariance) from a synthetic link sweep — no ground truth needed. **Not run at train time and not stored in `metadata.json`**; it is a standalone report, exposed as `routee-powertrain validate-physics`.
- **`resources/`** — Bundled pre-trained models (`bundled_registry/v2/...`) and sample route data
- **`registry/`** — Pluggable model discovery and retrieval system with multiple backends:
  - `ModelRegistry` (ABC) — Interface with `query()`, `load()`, `list_models()`, `get_metadata()`. `find_by_digest()` is concrete and delegates to `query()`, so a new backend gets it free. Also holds `INDEX_FILENAME` and `IndexMissingError`, shared by the remote backends.
  - `HFRegistry` — **The default.** Fetches models from a public HuggingFace Hub repo, read anonymously via `HfApi`. Downloads go through `hf_hub_download` into the shared HF cache, so repeat loads are offline; `revision` pins a branch/tag/commit sha to freeze the whole library. Discovery requires `index.json` at the schema root.
  - `S3Registry` — Fetches models from a public S3 bucket (`routeecore-bucket`). Discovery requires `index.json` at the schema root. `boto3` is imported lazily inside `_get_client()`/`build_index()`, so this module imports fine without the `s3` extra installed and only errors when actually used.
  - `LocalRegistry` — Reads models from a local directory tree using glob-based scanning
  - `registry/entry.py` — Backend-agnostic decoding of the shared layout: `parse_model_id_from_segments()`, `parse_model_id_from_metadata_key()` (used by both object-store backends), and `model_info_from_metadata()`. Add a backend by reusing these rather than copying them.
  - `ModelId` — Identifier with five segments: `make/vehicle_slug/year/config_slug/v<N>` (e.g. `toyota/camry_ice/2016/rf_c3326385/v1`). It composes a `ModelKey` (the version-less identity) plus a registry-assigned `version`. Both slugs are **derived** from metadata (`registry/slug.py`), not user-supplied: `derive_vehicle_slug` is `<model>_<powertrain_family>` (family via `powertrain_family()`: `ice`/`hev`/`bev`/`phev`/`heavy_duty`; both PHEV modes collapse to `phev` since CD/CS models describe the same vehicle — that split lives in `config.variant`; UNDEFINED omitted; model token lowercased, whitespace/`/` → `-`), and `derive_config_slug` composes `<arch_short>_<variant?>_<feature_hash>`. The model name should carry the vehicle's full commercial designation (`golf_1.5tsi` vs `golf_2.0tdi`); `engine`/`drivetrain`/`trim`/`fuel_type` are descriptive metadata — filterable and correctable without renaming paths, NOT identity. So `make/vehicle_slug/year/config_slug` are all pure functions of a model's metadata; only `version` is a registry coordinate. `ModelId.from_metadata(metadata, version)` / `ModelId.from_key(key, version)` mint one.
  - `ModelKey` — The version-less identity (`make/vehicle_slug/year/config_slug`), frozen/hashable. `Model.key` (`ModelKey.from_metadata`) exposes it on any trained or loaded model; `filter_models` groups versions by it. Registry `load()` re-derives both slugs and **raises on any mismatch** with the on-disk path (`assert_metadata_matches_id`).
  - `ModelInfo` — Lightweight summary returned by `query()` with metadata but no binary data. Carries `vehicle_model` (the bare metadata model name, e.g. `"camry"`) alongside `model_id.vehicle_slug`; the `model` query filter matches either.
  - `get_default_registry()` — Factory that selects backend based on `ROUTEE_REGISTRY_BACKEND` env var (`"hf"`, `"s3"`, or `"local"`, default `"hf"`)
  - `filter_models()` / `latest_model_ids()` — Filtering supports exact + fuzzy matching (via `rapidfuzz`) on `make`, `model`, `year`, `config_slug`, `feature_names`, `powertrain_type`, `fuel_type`, `drivetrain`, `engine`, `trim`, `version`, plus optional `custom_filters` callables.

### Registry

The Registry system abstracts model discovery and retrieval, allowing pre-trained models to be served from HuggingFace Hub, S3, or the local filesystem with an identical API. All three read the **same tree**, so a `ModelId` path means the same thing everywhere.

- **Directory/bucket/repo layout**: `<root>/<schema_version>/<make>/<vehicle_slug>/<year>/<config_slug>/v<N>/` containing `metadata.json` + a binary estimator file (e.g. `model.onnx` or a `.joblib` blob)
- **Bundled models**: `routee/powertrain/resources/bundled_registry/v2/` (currently `toyota/rav4_xle_ice` 2022 `rf_fe510e40/v1` and `polestar/2_bev` 2023 `rf_base_fe510e40/v1` — the derived slug for a random forest over speed, grade & distance). Both carry `mass_lbs`, so the mass-dependent physics checks apply to them.
- **Public entry points** (in `io/load.py`):
  - `list_available_models(registry=None, version_strategy="latest")` — Returns all `ModelId`s. `version_strategy="all"` returns every version; default keeps only the latest per `(make, vehicle_slug, year, config_slug)` group.
  - `query_available_models(...)` — Filtered search returning `ModelInfo` objects; supports fuzzy matching (`fuzzy=True`, `fuzzy_threshold=80`) and the same filter set as `filter_models()`.
  - `load_model(name_or_path, registry=None)` — Loads from a local path/zip/tar if it exists, otherwise resolves the string as a `ModelId` (`v<N>` segment optional → latest) and fetches from the registry.
- **Environment variables**:
  - `ROUTEE_REGISTRY_BACKEND` — `"hf"` (default), `"s3"`, or `"local"`
  - `ROUTEE_SCHEMA_VERSION` — default `"v2"`
  - `ROUTEE_HF_REPO_ID` — default `NatLabRockies/routee-powertrain-model-library`
  - `ROUTEE_HF_REPO_TYPE` — `"model"` (default) or `"dataset"`
  - `ROUTEE_HF_REVISION` — branch/tag/commit sha (default: the repo's default branch)
  - `ROUTEE_HF_TOKEN` — Hub token; unset reads anonymously
  - `ROUTEE_S3_BUCKET` — default `routeecore-bucket`
  - `ROUTEE_S3_REGION` — default `us-west-2`
  - `ROUTEE_S3_ROOT_PREFIX` — default `routee-powertrain-model-library`
  - `ROUTEE_LOCAL_REGISTRY_ROOT` — local directory root (defaults to bundled registry)

### Key abstractions

- **`Model`** — Main user-facing object. Holds a single `estimator: Estimator` + `metadata: Metadata`; `metadata.errors` exposes the `ModelErrors`. Supports `from_file`, `to_file` (directory / `.zip` / `.tar.gz` auto-detected from suffix), `predict`, `visualize_features`, `contour`, `to_lookup_table`.
- **`ModelConfig`** — Vehicle description, powertrain type, feature set, distance column, energy target(s), predict method, plus optional `mass_lbs`, `fuel_type`, `drivetrain`, `engine`, `trim` (the latter three fold into the derived `vehicle_slug` — set them as structured fields instead of lumping into the `model` name), `variant` (short label — e.g. `steady`/`warmup` — folded into the derived `config_slug` to distinguish configs sharing architecture + feature set), `training_source` (provenance — carries the `dataset_name`/`dataset_hash` labels, persisted under `metadata.provenance`, excluded from the digest), `real_world_adjustment_factor` (a float; when not supplied it defaults via a `mode="before"` validator to `ADJUSTMENT_FACTORS[powertrain_type]` — set it to `1.0` to apply no adjustment).
- **`Provenance`** (`core/provenance.py`) — the `provenance` section of `metadata.json`: where a model came from and how it was built. Three parts. `source` is a **tagged union** discriminated on `method` — `FastSimSource` (`fastsim_vehicle_id` from [fastsim-vehicles](https://github.com/NatLabRockies/fastsim-vehicles) — names the vehicle _pre_ PHEV-split, so CD/CS models share one id and `variant` is what separates them — plus `fastsim_vehicles_ref`, `fastsim_version`, `pipeline_version`/`pipeline_repo_ref`, `pipeline_run_id` (the **training** run), and `dataset_run_ids`+`data_sources` (lists — a run samples across every matching dataset, optionally spanning sources). Run ids are **keys, not copies**: the pipeline's provenance db holds the full run config — filters, sampling seed, estimator settings, trip caps, the assembled frame's identity — and none of it is duplicated here, so nothing can drift. Hence `FastSimSource` has **no** `dataset_name`/`dataset_hash`/sampling seed, while `RealWorldSource`/`LegacySource` keep the dataset labels, since nothing but the artifact stands behind collected or converted data. Reproducing a FASTSim model therefore needs db access — a deliberate trade, cheap to revisit since provenance is outside the digest), `RealWorldSource` (`data_source`, `fleet`, collection window, sample size), or `LegacySource` (`original_source`, `converted_from`) for models predating the section. Shared fields are repeated per variant rather than lifted into a base class, so serialized field order stays method → distinctive fields → shared tail; `Provenance.dataset_name`/`.dataset_hash` read through to the set variant via `getattr` and are `None` for `FastSimSource`. All fields optional — record what you know. `training` is the slimmed former `training` block (`test_size`, `validation_size`, `random_seed`, `trip_column`, `trained_date`). Set it via `ModelConfig.training_source`; `Provenance.from_config` assembles all three. **Nothing here feeds the digest** — that is what keeps it correctable on a published model. `provenance` is a **required** field on `Metadata`, so metadata written before 2.0.1 (flat `training` key, no `provenance`) fails validation on load; there is no translation shim.
- **`Estimator`** (ABC) — Stateless prediction interface. All implementations provide `predict()`, `from_dict()`/`to_dict()`, `from_file()`/`to_file()`, and expose a settable `input_spec: InputSpec` — the **full input/output contract**: windowing (lookback / grouping_column / pad_strategy) plus the positional `input_columns`/`output_columns` (`ColumnSpec` = name/units/dtype), `predict_method`, and `distance_column`. `Trainer.train` calls `estimator.bind_io_contract(config)` after `inner_train`, deriving `input_columns` from `config.all_features` (single source of truth: `feature_set` order, distance appended for RAW) and `output_columns` from `output_column_specs()` (default = targets; `NGBoostEstimator` overrides to append the `_std` column). **Where the ordered contract is stored — normalize within an artifact, denormalize across artifacts.** Inside `metadata.json`, `contract` (`feature_set` order + `distance` + `target` + `predict_method`) is the _single_ ordered source; `estimator.input_spec` there persists only the estimator mechanics `contract` can't express — `lookback`/`grouping_column`/`pad_strategy` (`io/archive.py:_build_metadata_dict` writes just those). The resolved positional columns are **not** duplicated into the JSON — but they _are_ embedded in the estimator binary (ONNX `metadata_props`), a separate artifact that a consumer (e.g. routee-compass) reads without `metadata.json`. The contract is **required on persist**: `_require_input_contract` (same save choke point) **raises** if any of the four contract fields is missing on the in-memory estimator when saving (directory/zip/tar/registry) — the `InputSpec` fields stay `Optional` for the transient pre-`bind` construction state, but no model can be written without them. On load, `_verify_input_contract` (in `_model_from_metadata_and_bytes`, the single load choke point) **raises** if a binary's embedded `input_columns` disagrees with the metadata order (from `contract`; legacy binaries with no embedded contract skip the check); a binary that carries no embedded contract (NGBoost's joblib blob) has its in-memory `input_spec` rebuilt from `contract` via `bind_io_contract(metadata.config)` so a load→save round trip stays contract-complete. `scripts/convert_legacy_models.py` (batch-driven by `convert_nlr_library.py`) fills the contract in during v1→v2 conversion: it reconstructs the estimator, calls `bind_io_contract`, and re-serializes so the ONNX binary is re-embedded (metadata keeps only the mechanics) and the digest is minted over the self-describing bytes. `input_spec` is excluded from the digest payload, so it does not affect `model_digest`. `scripts/backfill_input_contract.py` re-embeds the contract onto pre-contract registry entries.
- **`Trainer`** (ABC) — `train()` splits data, delegates to `inner_train()`, computes errors, returns a `Model`. Each subclass sets an `architecture_tag` (e.g. `random_forest`, `ngboost`, `cnn`) used in `metadata.json` and registry filtering.
- **`FeatureSetId`** — String alias (`core/features.py`) holding the sorted-`&`-joined feature column names. Used as a hashable feature-set fingerprint for registry queries; v2 models hold a single estimator per file, so there's no runtime feature-set dispatch inside `Model`.

### Predict methods

- **RATE** — Train on `energy/distance` (rate), predict rate then multiply by distance
- **RAW** — Train on raw energy with distance as a feature, predict total energy directly
- After the estimator runs, `Model.predict` multiplies each target by `ModelConfig.real_world_adjustment_factor` (which defaults to `ADJUSTMENT_FACTORS[powertrain_type]`). `predict()` takes no adjustment argument — the factor is baked into the config, so `1.0` is how you opt out. The v1 `apply_real_world_adjustment: bool` survives only in `io/legacy.py`, which maps `False` → `1.0` when converting old configs.

### Versioning & schema

- `metadata.json` carries `schema_version: 2`; v1 model files raise on load. Pre-2.0.1 v2 files (flat `training` key, no `provenance`) also raise — the library was regenerated wholesale for 2.0.1 rather than migrated.
- `ModelId.version` is an int (`v<N>` in paths). The default `version_strategy="latest"` collapses to the highest version per `(make, model, year, config_slug)` group; pass `"all"` to see every version.
- The package version is tracked in `routee/powertrain/__about__.py` and written into `metadata.routee_version` automatically at save time.

### Instance identity (digests)

Two-layer identity, mirroring OCI/MLflow: an immutable **instance digest** minted at train time lives inside the artifact; the registry path (`v<N>`) is a **coordinate** assigned at publish. The registry maps coordinate → digest, never the reverse.

- `Metadata.model_digest` (`sha256:<64 hex>`) — minted by `Trainer.train` via `core/digest.py:stamp_digest`, computed over a frozen canonical payload (`digest_spec: 1`) of identity fields: vehicle identity (make/model/year/variant/powertrain_type — everything feeding the derived path slugs), contract, and `estimator_sha256`. Recomputable from `metadata.json` alone. Deliberately excluded so they stay correctable: the **entire `provenance` section** (source, dataset labels, seed, splits, `trained_date` — anything that changes predictions already changes the estimator bytes, which `estimator_sha256` pins), the descriptive vehicle fields (`vehicle_description`, `mass_lbs`, `engine`, `drivetrain`, `trim`, `fuel_type`), and `errors`.
- `EstimatorInfo.estimator_sha256` — bare-hex sha256 of the exact estimator binary bytes. On load, a mismatch **raises** (corrupt binary); a `model_digest` mismatch **warns** (post-mint metadata edit). Both `None` on legacy models → checks skipped.
- The spec-1 payload builder in `core/digest.py` is frozen as of 2.0.1 — never edit it; a payload change ships as spec 2. A golden test in `tests/test_digest.py` pins the exact hash. (Two pre-release amendments preceded the freeze, both made while no published library was in use: settling the vehicle section on the coordinate-feeding fields, and dropping provenance from the payload for 2.0.1.)
- `save_to_registry` is **idempotent**: with `version=None`, an existing version holding the same `model_digest` is returned instead of minting v<N+1>.
- Coordinate lookup: `registry.find_by_digest(digest)` / `query(model_digest=...)` / `query_available_models(model_digest=...)` resolve a metadata file in hand back to its registry entry (exact match, `sha256:` prefix optional).
- `dataset_name`/`dataset_hash` (optional) live on the training source (`ModelConfig.training_source`), not on `ModelConfig` itself. `pt.hash_dataframe(df)` computes a fingerprint. Neither feeds the digest.
- `Metadata.model_key` (`make/vehicle_slug/year/config_slug`) — the version-less identity, cached in `metadata.json` so a consumer holding a detached archive can place it in a registry tree without re-implementing slug derivation (the same "denormalize across artifacts" reason the input contract is embedded in the ONNX binary). **Required on save, optional on read**: `io/archive.py:_build_metadata_dict` re-derives and stamps it on every save path (directory/zip/tar/registry), so it is never carried forward from a loaded value; `_verify_model_key` re-derives on load and **warns** on disagreement, matching `model_digest`. Derivation (`Metadata.derived_model_key` → `ModelKey.from_metadata`) stays the source of truth — the stored value is a cache, never trusted over it. `None` on artifacts published before the field existed, which load clean. **The registry `version` is deliberately not stored**: `_next_version` assigns it from whatever tree is being written, so it is a property of a registry's history rather than of the model, and an artifact cannot own it. Use `find_by_digest` to resolve an artifact to a coordinate in a specific registry. Excluded from the digest payload (its inputs are already covered there).
- `scripts/backfill_digests.py` stamps digests onto pre-digest registry entries (requires the binaries).

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
  - `tests/test_hf_registry.py` — `HFRegistry` query/load/`build_index` against a fake `HfApi` injected on `registry._client`, plus the default-backend factory checks and a subprocess assertion that importing the package does not pull in `boto3`
  - `tests/test_s3_registry.py` — `S3Registry` boto3 integration, key parsing
  - `tests/test_s3_index.py` — S3 `index.json` build / parse
  - `tests/test_provenance.py` — `Provenance` tagged-union round trip, flat-config translation, and the hard break on pre-2.0.1 metadata
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
| `s3`      | boto3 for the legacy S3 registry backend                  |
| `dev`     | All of the above + pytest, mypy, ruff, jupyter-book, etc. |

Install with: `pip install -e ".[scikit]"`, `pip install -e ".[ngboost,plot]"`, etc.
