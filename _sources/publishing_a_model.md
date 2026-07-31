# Publishing a Trained Model

After training a model with one of the `Trainer` classes, you can save it
into the v2 registry layout that every registry backend understands —
`LocalRegistry`, `HFRegistry`, and `S3Registry` all read the same tree.
This page walks through writing a trained model into a local registry,
understanding the derived `vehicle_slug` and `config_slug`, and loading it
back through `pt.load_model()`.

## The v2 Registry Layout

Every model in the registry lives in a five-segment path:

```
<registry_root>/v2/<make>/<vehicle_slug>/<year>/<config_slug>/v<N>/
    metadata.json
    model.onnx        # (or another binary, depending on the estimator)
```

The bundled example at
`routee/powertrain/resources/bundled_registry/v2/toyota/camry_ice/2016/rf_c3326385/v1/`
is a concrete reference for what the on-disk layout looks like.

## The derived `vehicle_slug`

The second path segment identifies the vehicle. Like the `config_slug`, **you
don't write it — it is derived from the model's metadata** as a pure function:

```
<model>_<powertrain_family>
```

- **model** — the `ModelConfig.model` name. Use the vehicle's full commercial
  designation, including whatever distinguishes same-year stablemates:
  `camry`, `golf_1.5tsi` vs `golf_2.0tdi`, `leaf_24_kwh` vs `leaf_30_kwh`.
- **powertrain_family** — the coarse family of `powertrain_type`: `ice`,
  `hev`, `bev`, `phev`, `heavy_duty`. The two PHEV operating modes
  (`PHEV_EV_MODE` / `PHEV_HEV_MODE`) collapse to one `phev` family — a
  charge-depleting and a charge-sustaining model describe the _same vehicle_,
  and that split lives in `ModelConfig.variant` (and thus the `config_slug`).

So `model="Camry"` with `powertrain_type=ICE` lands at
`.../toyota/camry_ice/...`, and both Volt operating-mode models share
`.../chevrolet/volt_phev/...`. The model token is sanitized (lowercased,
whitespace and `/` become `-`), and the slug is never parsed back apart — the
registry always re-derives it from metadata and compares.

The remaining vehicle attributes — `engine`, `drivetrain`, `trim`,
`fuel_type`, `mass_lbs` — are deliberately **not** identity. They are
descriptive metadata: individually filterable
(`pt.query_available_models(engine="4cyl")`) and correctable on an
already-published model without renaming its registry path. Only put a
distinction in the model name when it genuinely names a different vehicle.

## The derived `config_slug`

The `config_slug` disambiguates multiple trained configurations for the same
vehicle and year. **You don't pick it — it is derived from the model's
metadata**, so it stays consistent and can't drift from what the model actually
is. The slug is a pure function:

```
<architecture>_<variant?>_<feature_set_hash>
```

- **architecture** — a short code for the estimator family (`rf`, `ngb`, `cnn`),
  from `metadata.estimator.architecture_tag`.
- **variant** — the optional `ModelConfig.variant` label, included only when set.
- **feature_set_hash** — a short hash of the feature set, so different feature
  compositions get different slugs automatically.

For example, `rf_c3326385` (a random forest over speed & grade) or
`ngb_stochastic_96224f1f` (an NGBoost model with `variant="stochastic"`).

Because the hash already separates different feature sets, the only time you
need to intervene is when **two models share the same architecture _and_ feature
set** but represent different regimes — e.g. a "steady thermal state" model and
a "warm-up" model. Set `ModelConfig.variant` to tell them apart:

```python
config = pt.ModelConfig(..., variant="steady")   # -> rf_steady_<hash>
config = pt.ModelConfig(..., variant="warmup")   # -> rf_warmup_<hash>
```

The registry recomputes both derived slugs when loading and **raises if either
disagrees with the on-disk path**, so a moved or hand-edited model surfaces
loudly instead of silently mis-loading. `version` is the one coordinate the
registry assigns, not part of the derived identity — retraining the same
configuration bumps `v<N>`.

## Train and Publish

`Model.save_to_registry()` writes a trained model into the canonical layout in
one call. It pulls `make` and `year` from `model.metadata.vehicle` and derives
the `vehicle_slug` (model + powertrain family) and the `config_slug`, so make
sure those fields on your `ModelConfig` are correct before training.

```python
import routee.powertrain as pt
from routee.powertrain.trainers.sklearn_random_forest import (
    SklearnRandomForestTrainer,
)

config = pt.ModelConfig(
    vehicle_description="2024 Test Sedan",
    powertrain_type=pt.PowertrainType.ICE,
    feature_set=pt.FeatureSet(features=[
        pt.DataColumn(name="speed_mph", units="mph"),
        pt.DataColumn(name="grade_dec", units="decimal"),
    ]),
    distance=pt.DataColumn(name="miles", units="miles"),
    target=pt.TargetSet(targets=[
        pt.DataColumn(name="gallons_fastsim", units="gallons_gasoline"),
    ]),
    make="Test",
    model="Sedan",
    year=2024,
)

model = SklearnRandomForestTrainer().train(training_df, config)

# both slugs are derived; version defaults to the next unused version.
model_id = model.save_to_registry(registry_root="./my_local_registry")
print(model_id)      # test/sedan_ice/2024/rf_aaa9554f/v1
print(model.key)     # test/sedan_ice/2024/rf_aaa9554f  (version-less identity)
```

This creates `./my_local_registry/v2/test/sedan_ice/2024/rf_aaa9554f/v1/` with
`metadata.json` and `model.onnx` inside. If that exact version already has
files, the call raises `FileExistsError` — omit `version` to auto-increment, or
pass `overwrite=True`.

Every `Model` also exposes its version-less identity as `model.key` (a
`ModelKey`), available the moment it is trained — even before it is placed in a
registry.

## Load It Back

There are two equivalent ways to load a model from the registry you just
wrote.

**Option A — explicit `LocalRegistry`:**

```python
from routee.powertrain.registry.local import LocalRegistry

model = LocalRegistry("./my_local_registry").load(model_id)
predictions = model.predict(links_df)
```

**Option B — `pt.load_model()` driven by environment variables:**

```python
import os
import routee.powertrain as pt

os.environ["ROUTEE_REGISTRY_BACKEND"] = "local"
os.environ["ROUTEE_LOCAL_REGISTRY_ROOT"] = "./my_local_registry"

# Omit the trailing v<N> to get the latest version. model.key.to_path() gives
# exactly this version-less path.
model = pt.load_model("test/sedan_ice/2024/rf_aaa9554f")
```

You can also list and query what's in the registry the same way you would
with the default HuggingFace registry:

```python
pt.list_available_models()
pt.query_available_models(make="test", powertrain_type="ICE")
```

## What's in `metadata.json`

`metadata.json` is written automatically by the save call. Its fields are
grouped by the job a reader needs them for — `vehicle` (identity), `contract`
(input/output), `estimator` (how to load the binary), and `provenance` (where
the model came from). Most of the contents come from your `ModelConfig` and the
trainer:

| Field                                                       | Source                                                     |
| ----------------------------------------------------------- | ---------------------------------------------------------- |
| `schema_version`                                            | `routee.powertrain.core.metadata.SCHEMA_VERSION`           |
| `routee_version`                                            | Package version at save time                               |
| `errors`                                                    | Computed during `Trainer.train()`                          |
| `estimator.estimator_type`                                  | The trainer (e.g. `"ONNXEstimator"`)                       |
| `estimator.architecture_tag`                                | The trainer (e.g. `"random_forest"`, `"cnn"`)              |
| `estimator.input_spec`                                      | The estimator (lookback / grouping column / pad)           |
| `estimator.model_file`                                      | Filename of the binary (e.g. `"model.onnx"`)               |
| `vehicle.make / model / year`                               | From `ModelConfig` — drives the path                       |
| `vehicle.powertrain_type`                                   | From `ModelConfig` — feeds the `vehicle_slug` family token |
| `vehicle.variant`                                           | From `ModelConfig` (optional) — feeds the `config_slug`    |
| `vehicle.engine / drivetrain / trim`                        | From `ModelConfig` (optional) — descriptive, filterable    |
| `vehicle.vehicle_description`                               | From `ModelConfig`                                         |
| `vehicle.mass_lbs / fuel_type`                              | From `ModelConfig` (optional)                              |
| `contract.feature_set / target / distance`                  | From `ModelConfig`                                         |
| `contract.predict_method`                                   | From `ModelConfig`                                         |
| `provenance.source`                                         | From `ModelConfig.training_source` (optional)              |
| `provenance.training.test_size / random_seed / trip_column` | From `ModelConfig`                                         |
| `provenance.training.trained_date`                          | Stamped by `Trainer.train()`                               |

`Metadata.config` reconstructs the original flat `ModelConfig` from these
grouped sections on demand, so nothing is stored twice.

### Recording provenance

`provenance.source` records **how the training data was produced**. It is a
tagged union — set `ModelConfig.training_source` to one of three types and the
`method` discriminator picks the right one back out on load:

```python
# The standard path: a model pipeline over FASTSim simulation output.
config = pt.ModelConfig(
    ...,
    training_source=pt.FastSimSource(
        # NatLabRockies/fastsim-vehicles, plus a git tag pinning that repo
        fastsim_vehicle_id="v1/fastsim-3/conv/toyota/camry-4cyl-2wd/2016/base/r1",
        fastsim_vehicles_ref="v1.2.0",
        fastsim_version="3.1.0",
        # the pipeline, and the training run that produced this model
        pipeline_version="0.4.1",
        pipeline_run_id="gha-2026-07-14-8871",
        pipeline_repo_ref="9f3c1ab",
        # the prepare-training-data runs it was fit to, and their sources
        dataset_run_ids=["ptd-2026-07-14-001", "ptd-2026-07-14-002"],
        data_sources=["wm1", "wm2"],
    ),
)

# Trained on real-world vehicle data instead.
config = pt.ModelConfig(
    ...,
    training_source=pt.RealWorldSource(
        data_source="fleet_dna",
        fleet="delivery_vans",
        collection_start="2023-01-01",
        collection_end="2023-12-31",
        n_vehicles=42,
        dataset_name="fleet-dna-2023",
        dataset_hash=pt.hash_dataframe(training_df),
    ),
)
```

`RealWorldSource` and `LegacySource` end with two dataset labels — `dataset_name`
(a human-readable identifier) and `dataset_hash` (a fingerprint, from
`pt.hash_dataframe(df)`). They live on the source because labeling the data is
part of describing where it came from. `FastSimSource` deliberately has neither;
see below. `model.metadata.provenance.dataset_name` and `.dataset_hash` read
through to whichever variant is set, and are `None` for a `FastSimSource`.

Every field is optional — record what you know. `pt.LegacySource` is the third
variant, used by the v1 converter for models whose origin predates this section.

Provenance is **not** part of the model digest, so it stays correctable after
publish: backfilling a FASTSim version onto a published model does not change
its identity. What the digest does cover is the vehicle identity, the contract,
and the estimator binary's own sha256 — and that binary already changes whenever
the training data or hyperparameters do.

Neither derived slug in the path is stored in `metadata.json`: the
`vehicle_slug` is derived from `vehicle.model` + the `powertrain_type` family,
and the `config_slug` from `estimator.architecture_tag` + `vehicle.variant` +
`contract.feature_set` (see `routee/powertrain/registry/slug.py`).

See `routee/powertrain/core/metadata.py` for the full schema and
`routee/powertrain/core/model_config.py` for the `ModelConfig` fields.

## Sharing Your Model

Once your model loads cleanly from a local registry, that same directory tree
is what gets uploaded to the shared HuggingFace Hub repository. Publishing
requires write access to that repo and is currently a maintainer step —
package the `<registry_root>/v2/...` directory (or zip it) and hand it off.
The maintainer-side upload and index-refresh scripts live at
`scripts/upload_to_hf.py` and `scripts/build_hf_index.py` (with
`scripts/upload_to_s3.py` and `scripts/build_s3_index.py` still there for the
legacy S3 mirror).
