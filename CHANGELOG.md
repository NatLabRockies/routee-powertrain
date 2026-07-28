# Changelog

All notable changes to this project are documented here.

## [2.0.0]

A breaking rewrite of the packaging, model file format, and model distribution story.

**Upgrading from 1.x? See the [migration guide](docs/migrating_from_v1.md).**

### Headline changes

1. Package renamed `nrel.routee.powertrain` → `routee.powertrain`.
2. A `Model` now holds exactly **one** estimator instead of a dict of estimators keyed by feature set.
3. Model files are **archives** (directory / `.zip` / `.tar.gz`) with `metadata.json` + a binary
   estimator, replacing the single monolithic JSON.
4. A pluggable **registry** (HuggingFace Hub, S3, or local) replaces the hardcoded list of
   bundled/external models. The Hub is the default source of pre-trained models.
5. Models carry a content-addressed **digest** identity, separate from their registry coordinate.
6. Core types moved from `@dataclass` to **pydantic**.
7. New **CNN trainer** (PyTorch → ONNX) with sequence/lookback support.
8. Estimator binaries are **self-describing**: the ONNX file embeds its own input/output contract.

### Packaging and project identity

- Import path: `import nrel.routee.powertrain as pt` → `import routee.powertrain as pt`.
  All source moved from `nrel/routee/powertrain/` to `routee/powertrain/`.
- PyPI distribution renamed `nrel.routee.powertrain` → `routee.powertrain`.
- Homepage moved to `github.com/NatLabRockies/routee-powertrain`.
- Python support widened to `>=3.10,<3.14` (3.13 added to the CI matrix).
- New runtime dependencies: `pydantic`, `huggingface_hub`, `rapidfuzz`.
- New optional extras: `pytorch` (`torch`, `onnxscript`) for the CNN trainer, and `s3` (`boto3`)
  for the S3 registry backend. **`boto3` is no longer installed by default** — if you set
  `ROUTEE_REGISTRY_BACKEND=s3`, install `routee.powertrain[s3]`.
- New `routee-powertrain` console script (currently exposing `convert-v1`).
- `dprint` added for non-Python formatting (JSON/YAML/TOML/Markdown), enforced in CI.
- `.pre-commit-config.yaml` added, running `pixi run ci` (fmt + lint + typing + test).
- Corrected the PyPI license classifier to `License :: OSI Approved :: BSD License`.

### Migration support

- New `pt.convert_legacy_model()` and `routee-powertrain convert-v1` convert v1 `.json` models to
  the v2 archive format. Because v1 packed every feature set into one file, one v1 model converts
  to one v2 model **per feature set**.
- Passing a v1 flat model name (e.g. `"2016_TOYOTA_Camry_4cyl_2WD"`) to `load_model()` now raises a
  message explaining the id change and pointing at `query_available_models()` and the converter.
- Passing a `.json` path to `load_model()` now explains the format change and points at the
  converter, rather than only listing the accepted suffixes.

### Removed

- **The Rust / smartcore path**: `rust/` crate, `SmartCoreEstimator`, `SmartCoreRandomForestTrainer`.
- **The sklearn-native estimator** (`estimators/sklearn/`), including `port_to_c.py`. Random forests
  now go through ONNX only.
- **`resources/default_models/`** (the two bundled JSON models plus `external_model_links.json`),
  replaced by the bundled registry. The Box-hosted download-link workflow went with it.
- `Model.from_url()` — remote fetching is the registry's job now.
- `routee/powertrain/io/api.py` and `read_model`.
- Legacy notebook docs (`model_training.ipynb`, `model_prediction.ipynb`, etc.), replaced with
  `.py` / `.md` sources.

### Model structure: one estimator per model

v1 `Model` held `estimators: Dict[FeatureSetId, Estimator]` and dispatched at predict time based on
which columns the caller supplied. v2 holds a single `estimator: Estimator`; different feature sets
mean different published models.

- `ModelConfig.feature_sets: List[FeatureSet]` → `ModelConfig.feature_set: FeatureSet`.
- `Model.feature_sets` / `feature_set_lists` → `Model.feature_set` / `feature_names`.
- `Model.errors` → `Model.metadata.errors`.
- `ModelErrors.estimator_errors` is no longer a dict keyed by `FeatureSetId` — it is a single
  `EstimatorErrors`.
- `compute_errors()` takes one `estimator` instead of a dict.
- `Model.predict()` no longer accepts `feature_columns`, `distance_column`, or
  `apply_real_world_adjustment`; it raises on missing features instead of re-dispatching, and the
  real-world adjustment always comes from `config.real_world_adjustment_factor`.
- `FeatureSetId` survives only as a hashable fingerprint used for registry filtering.

### Serialization: archives instead of one JSON blob

`io/archive.py` replaces the old `Model.to_dict()` / `from_dict()` round trip.

- A saved model is a **directory** containing `metadata.json` plus the estimator binary
  (`model.onnx`, or a `.joblib` blob for NGBoost).
- `Model.to_file()` / `from_file()` auto-detect directory, `.zip`, or `.tar.gz` from the suffix.
- `metadata.json` carries `schema_version: 2`; v1 model files raise on load.
- `Metadata` is now decomposed into sections — `vehicle`, `contract`, `estimator`, `training`,
  `errors` — with a `Metadata.config` property that reconstitutes the flat `ModelConfig` on demand.

### Added: the model registry

- `ModelRegistry` (ABC) with `query()`, `load()`, `list_models()`, `get_metadata()`.
- `HFRegistry` — a public HuggingFace Hub repository, read anonymously. Downloads land in the
  shared HuggingFace cache, so loading the same model twice hits the network once, and
  `ROUTEE_HF_REVISION` pins a branch, tag, or commit sha to freeze the whole library.
- `S3Registry` — public S3 bucket, with an `index.json` for fast queries. Requires the `s3` extra.
- `LocalRegistry` — glob-based scan of a local directory tree.
- `get_default_registry()` picks the backend from `ROUTEE_REGISTRY_BACKEND` (`"hf"` default, or
  `"s3"` / `"local"`).
- Layout: `<root>/<schema_version>/<make>/<vehicle_slug>/<year>/<config_slug>/v<N>/`.
- `ModelId` — five-segment identifier, e.g. `toyota/camry_ice/2016/rf_c3326385/v1`. Both slugs are
  **derived from metadata**, never user-supplied; loading re-derives them and raises on any mismatch
  with the on-disk path.
- `ModelKey` — the version-less identity, frozen and hashable, exposed as `Model.key`.
- `ModelInfo` — lightweight metadata-only summary returned by `query()`.
- Bundled models now live at `routee/powertrain/resources/bundled_registry/v2/` (Toyota Camry 2016
  ICE and Chevrolet Bolt 2017 BEV).
- Environment variables: `ROUTEE_REGISTRY_BACKEND`, `ROUTEE_SCHEMA_VERSION`, `ROUTEE_HF_REPO_ID`,
  `ROUTEE_HF_REPO_TYPE`, `ROUTEE_HF_REVISION`, `ROUTEE_HF_TOKEN`, `ROUTEE_S3_BUCKET`,
  `ROUTEE_S3_REGION`, `ROUTEE_S3_ROOT_PREFIX`, `ROUTEE_LOCAL_REGISTRY_ROOT`.

Public API changes:

| v1                                                 | v2                                                                             |
| -------------------------------------------------- | ------------------------------------------------------------------------------ |
| `list_available_models(local=True, external=True)` | `list_available_models(registry=None, version_strategy="latest")` → `ModelId`s |
| —                                                  | `query_available_models(...)` → filtered `ModelInfo` list, with fuzzy matching |
| `load_model("2016_TOYOTA_Camry_4cyl_2WD")`         | `load_model("toyota/camry_ice/2016/rf_c3326385/v1")` (or a local path/archive) |
| —                                                  | `save_to_registry(...)` / `Model.save_to_registry(...)`                        |

`query_available_models()` filters on `make`, `model`, `year`, `config_slug`, `feature_names`,
`powertrain_type`, `fuel_type`, `drivetrain`, `engine`, `trim`, `version`, `model_digest`, plus
arbitrary `custom_filters` callables. Fuzzy matching via `rapidfuzz` is on by default
(`fuzzy_threshold=80`).

### Added: instance identity via digests

Two-layer identity, modeled on OCI/MLflow — an immutable digest minted at train time lives inside
the artifact; the registry path (`v<N>`) is a coordinate assigned at publish.

- `Metadata.model_digest` (`sha256:<64 hex>`) is computed over a frozen canonical payload
  (`digest_spec: 1`) of identity fields, and is recomputable from `metadata.json` alone. Descriptive
  fields (`vehicle_description`, `mass_lbs`, `engine`, `drivetrain`, `trim`, `fuel_type`) and
  `errors` are deliberately excluded so they stay correctable.
- `EstimatorInfo.estimator_sha256` is the sha256 of the exact estimator bytes. On load, a binary
  mismatch **raises**; a `model_digest` mismatch only **warns** (post-mint metadata edit).
- `save_to_registry` is **idempotent**: with `version=None`, an existing version holding the same
  digest is returned rather than minting `v<N+1>`.
- `registry.find_by_digest(digest)` and `query(model_digest=...)` resolve a metadata file in hand
  back to its registry coordinate.
- New helpers: `pt.compute_model_digest()`, `pt.hash_dataframe(df)`.
- `ModelConfig.dataset_name` / `dataset_hash` record training-data provenance and feed the digest.

### Pydantic migration

`ModelConfig`, `Metadata`, `EstimatorInfo`, `Errors`, `EstimatorErrors`, `ModelErrors`,
`FeatureSet`, `InputSpec` and friends are now `pydantic.BaseModel` subclasses instead of
`@dataclass` + hand-written `to_dict` / `from_dict`. Custom validated field types live in
`core/pydantic_fields.py`.

New structured config types exported at the top level: `Vehicle`, `Contract`, `TrainingConfig`,
`Metadata`, `EstimatorInfo`, `Drivetrain`, `FuelType`.

`ModelConfig` field changes:

- Added: `make`, `model`, `year` (structured vehicle identity — feeds derived slugs), `variant`,
  `mass_lbs`, `fuel_type`, `drivetrain`, `engine`, `trim`, `dataset_name`, `dataset_hash`,
  `validation_size`.
- Changed: `feature_sets` → `feature_set`; `apply_real_world_adjustment: bool` →
  `real_world_adjustment_factor: float`; `test_size` now optional.
- `Year` may be a single int or a `(start, end)` tuple.

### Added: CNN trainer

`trainers/cnn.py` adds a 1D CNN trainer built on PyTorch and exported via `torch.onnx.export`.

- Requires a `grouping_column` (e.g. `route_id`) to bucket rows into sequences.
- Produces an `ONNXEstimator` with a non-default `InputSpec` carrying `lookback` /
  `grouping_column` / `pad_strategy`.
- Includes an export drift check that verifies the exported ONNX matches the in-memory PyTorch model.

Each `Trainer` subclass now sets an `architecture_tag` (`random_forest`, `ngboost`, `cnn`) used in
`metadata.json` and for registry filtering, and gains `required_extra_columns` /
`split_grouping_column`.

### Estimator interface: the input/output contract

`ONNXEstimator` now handles both plain tabular input `(N, F)` and windowed sequence input
`(N, lookback, F)`, driven by an `InputSpec` embedded in the ONNX `metadata_props`.

A bare `.onnx` file is now **self-describing**: `routee_input_columns` and `routee_output_columns`
(JSON arrays of `{name, units, dtype}` in positional tensor order), plus `routee_predict_method` and
`routee_distance_column`, are written into the binary, so a downstream consumer such as
routee-compass can reconstruct the exact positional input order without reading `metadata.json`.

- `Trainer.train` calls `estimator.bind_io_contract(config)` after `inner_train`.
- The contract is **required on persist** — saving raises if it is missing.
- On load, an embedded contract that disagrees with the metadata order raises.
- The contract is excluded from the digest payload, so it does not change `model_digest`.

Also added: `utils/threading.py` detects restricted CPU affinity (e.g. containers) and passes
onnxruntime session options so ONNX respects the environment's thread limit.

### Fixed

- The "trained using a different version" warning now fires only for models built by a _newer_
  major version. It previously fired on every load of a converted v1 artifact, since those record
  the version that trained them; real format drift is caught hard by the `schema_version` check.
- README quickstart used the model id `chevrolet/bolt/2017/...`; the shipped bundled model is
  `chevrolet/bolt_bev/2017/...`.
- Docs diagrams are served from `docs/images/` instead of hotlinked GitHub asset URLs.

### Migration and operations scripts

| Script                               | Purpose                                                      |
| ------------------------------------ | ------------------------------------------------------------ |
| `scripts/convert_nlr_library.py`     | Batch conversion of the full legacy model library            |
| `scripts/upload_to_hf.py`            | Publish a local registry tree to a HuggingFace Hub repo      |
| `scripts/build_hf_index.py`          | Build the Hub `index.json` used for fast queries             |
| `scripts/upload_to_s3.py`            | Publish a local registry tree to S3                          |
| `scripts/build_s3_index.py`          | Build the S3 `index.json` used for fast queries              |
| `scripts/backfill_digests.py`        | Stamp digests onto pre-digest registry entries               |
| `scripts/backfill_input_contract.py` | Re-embed the I/O contract onto pre-contract registry entries |

(`scripts/convert_legacy_models.py` is now a thin shim over the packaged
`routee.powertrain.io.legacy`.)

### Documentation

- New [migration guide](docs/migrating_from_v1.md).
- Notebooks converted to `.py` (jupytext-style) and `.md` sources; the docs TOC was restructured
  around Getting Started / User Reference / Examples / Developers.
- New `docs/publishing_a_model.md` covering the registry publishing workflow.
- New `docs/examples/model_prediction_example.py` and `model_training_example.py`.
- README rewritten around the registry API and the new package name.
