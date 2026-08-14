# Migrating from v1

RouteE-Powertrain 2.0 is a breaking release. Beyond the package rename, the model file format,
the model catalog, and much of the `Model` API changed. This page walks through the migration in
the order you are likely to hit each change.

If you just want the checklist, jump to
{ref}`Migration checklist <migration-checklist>`.

## 1. Install the new package

The project was previously published as `nrel.routee.powertrain`. Following the lab's rename it is
now published as **`routee.powertrain`**:

```bash
pip uninstall nrel.routee.powertrain
pip install routee.powertrain
```

The 1.x line is frozen. `nrel.routee.powertrain` 1.4.1 is the final 1.x release and emits a
`DeprecationWarning` on import; `nrel.routee.powertrain` 2.0.0 is a tombstone that installs
`routee.powertrain` and raises `ImportError` pointing here.

## 2. Update your imports

The import root dropped the `nrel` namespace:

```diff
-import nrel.routee.powertrain as pt
+import routee.powertrain as pt
```

Submodule imports change the same way:

```diff
-from nrel.routee.powertrain.trainers.sklearn_random_forest import SklearnRandomForestTrainer
+from routee.powertrain.trainers.sklearn_random_forest import SklearnRandomForestTrainer
```

## 3. Replace model names with model ids

v1 loaded models by flat name from a hardcoded catalog:

```python
# v1
model = pt.load_model("2016_TOYOTA_Camry_4cyl_2WD")
```

v2 resolves models through a [registry](publishing_a_model.md) using a structured id,
`<make>/<vehicle_slug>/<year>/<config_slug>`, with an optional trailing `v<N>`:

```python
# v2
model = pt.load_model("toyota/camry_ice/2016/rf_c3326385")       # latest version
model = pt.load_model("toyota/camry_ice/2016/rf_c3326385/v1")    # pinned version
```

Rather than translating names by hand, search the registry:

```python
for info in pt.query_available_models(make="toyota", model="camry", year=2016):
    print(info.model_id, info.feature_names)
```

`query_available_models()` also filters on `powertrain_type`, `fuel_type`, `drivetrain`, `engine`,
`trim`, `feature_names`, and `config_slug`, and does fuzzy matching by default — so
`model="camry"` will find `camry_ice` without you knowing the slug.

### One v1 model is now several v2 models

v1 packed **every** feature set into a single file and
picked an estimator at predict time based on which columns you passed. v2 publishes **one model per
feature set**, so a v1 model typically maps to three v2 models:

| v1 feature set                           | v2 `config_slug` |
| ---------------------------------------- | ---------------- |
| `speed_mph`                              | `rf_db8522fb`    |
| `speed_mph & grade_percent`              | `rf_c3326385`    |
| `speed_mph & grade_percent & turn_angle` | `rf_b80965c8`    |

Pick the one whose features match the columns you actually have. `rf_c3326385` (speed and grade) is
the closest match to what v1 usually selected.

Other slug patterns you may see:

- PHEV models carry the mode: `rf_charge_depleting_*`, `rf_charge_sustaining_*`
- Thermal models: `rf_steady_thermal_*`, `rf_transient_thermal_*`
- Probabilistic (NGBoost) models: `ngb_stochastic_*`

`list_available_models()` also changed shape — it now takes `registry` and `version_strategy`
instead of `local`/`external`, and returns `ModelId` objects instead of strings:

```diff
-names: list[str] = pt.list_available_models(local=True, external=True)
+ids: list[ModelId] = pt.list_available_models()
```

## 4. Convert your own v1 model files

v1 stored an entire model as one `.json` file with base64-encoded binaries. v2 models are
**archives** — a directory, `.zip`, or `.tar.gz` containing `metadata.json` plus a separate binary
(`model.onnx` or `model.joblib`). Loading a v1 `.json` now raises with a pointer to the converter.

Convert from the command line:

```bash
routee-powertrain convert-v1 MyModel.json out/ \
    --make toyota --model camry --year 2016 --trim 4cyl_2wd
```

or from Python:

```python
paths = pt.convert_legacy_model(
    "MyModel.json", "out/", make="toyota", model="camry", year=2016
)
models = [pt.load_model(p) for p in paths]
```

Because one v1 file holds several feature sets, conversion writes **one v2 model directory per
feature set** and returns all of their paths.

`--make`, `--model`, and `--year` are required: v1 metadata had no structured vehicle identity,
only a free-text `vehicle_description`, so there is nothing to infer them from. They matter because
the registry path is _derived_ from them. `--variant`, `--fuel-type`, `--drivetrain`, `--engine`,
and `--trim` are optional; run `routee-powertrain convert-v1 --help` for the full list.

```{note}
Models saved with the v1 `SmartCoreEstimator` cannot be converted — the smartcore/Rust path was
removed in v2. Retrain those with `SklearnRandomForestTrainer`, which exports to ONNX.
```

## 5. API changes

### `Model`

| v1                                              | v2                                                          |
| ----------------------------------------------- | ----------------------------------------------------------- |
| `model.estimators` (dict keyed by feature set)  | `model.estimator` (exactly one)                             |
| `model.errors`                                  | `model.metadata.errors`                                     |
| `model.feature_sets`, `model.feature_set_lists` | `model.feature_set`, `model.feature_names`                  |
| —                                               | `model.key` (`ModelKey`), `model.digest`                    |
| `Model.from_dict()` / `to_dict()`               | removed — use `from_file()` / `to_file()`                   |
| `Model.from_url()`                              | removed — remote fetching is the registry's job             |
| `to_file()` accepted `.json` only               | directory, `.zip`, or `.tar.gz`, chosen by suffix           |
| —                                               | `model.save_to_registry(...)`, `model.to_lookup_table(...)` |

`predict()` lost its optional arguments:

```diff
-model.predict(links_df, feature_columns=[...], distance_column="miles",
-              apply_real_world_adjustment=True)
+model.predict(links_df)
```

Rename your columns to match the model's contract before calling `predict()`; the model now raises
if a required feature is missing rather than silently choosing a different estimator. The real-world
adjustment is always applied, using `real_world_adjustment_factor` from the model config.

### `ModelConfig`

`ModelConfig` (and `Metadata`, `ModelErrors`, `FeatureSet`, and friends) moved from
`@dataclass` to **pydantic**, so hand-written `from_dict()` / `to_dict()` / `to_json()` methods are
gone — use `model_validate()` and `model_dump()`.

```diff
 config = pt.ModelConfig(
     vehicle_description="2016 Toyota Camry",
     powertrain_type=pt.PowertrainType.ICE,
-    feature_sets=[pt.FeatureSet([...])],
+    feature_set=pt.FeatureSet([...]),
+    make="toyota",
+    model="camry",
+    year=2016,
     distance=...,
     target=...,
-    apply_real_world_adjustment=False,
+    real_world_adjustment_factor=1.0,
 )
```

`make`, `model`, and `year` are now required — they feed the derived registry path. New optional
fields include `variant`, `mass_lbs`, `fuel_type`, `drivetrain`, `engine`, `trim`, and
`training_source` (which carries the `dataset_name` / `dataset_hash` labels).

Models converted from v1 record a `pt.LegacySource` under `provenance.source`: the v1 format stored
nothing about how a model was produced, so there is no simulator, pipeline, or dataset information
to carry over. Pass `provenance_source=` to `convert_legacy_json` when you know more than the format
does.

```{note}
Put the vehicle's full commercial designation in `model` (`golf_1.5tsi` vs `golf_2.0tdi`).
`engine`, `drivetrain`, and `trim` are descriptive metadata — filterable and correctable without
renaming registry paths.
```

### Removed

- `SmartCoreEstimator`, `SmartCoreRandomForestTrainer`, and the Rust crate
- the sklearn-native estimator and `port_to_c` — random forests go through ONNX only
- `routee/powertrain/io/api.py` and `read_model`
- `resources/default_models/` and the Box-hosted `external_model_links.json` catalog

### Custom trainers

If you subclass `Trainer`, add the two new members `required_extra_columns` and
`split_grouping_column`.

## 6. What's new

Worth knowing about even though nothing forces you to use it:

- **Registry** — pluggable model discovery over HuggingFace Hub (the default), S3, or a local
  directory, configured with `ROUTEE_REGISTRY_BACKEND` and friends. See
  [Publishing a model](publishing_a_model.md).
- **Digests** — every model carries a content-addressed `model_digest` minted at train time, and
  the estimator binary is checksummed on load. See `pt.compute_model_digest()`.
- **Self-describing binaries** — the `.onnx` file embeds its own input/output contract, so a
  downstream consumer can reconstruct the exact positional input order without `metadata.json`.
  See [Estimator input contract](estimator_input_contract.md).
- **CNN trainer** — a 1D CNN with sequence/lookback support, via the `pytorch` extra.

(migration-checklist)=

## Migration checklist

- [ ] `pip uninstall nrel.routee.powertrain` and `pip install routee.powertrain`
- [ ] `import nrel.routee.powertrain` → `import routee.powertrain`
- [ ] Replace flat model names in `load_model()` with model ids, choosing the feature set you have
- [ ] Replace `list_available_models(local=..., external=...)` calls; it returns `ModelId`s now
- [ ] Convert any v1 `.json` models with `routee-powertrain convert-v1`
- [ ] `ModelConfig(feature_sets=[...])` → `feature_set=...`, and add `make` / `model` / `year`
- [ ] `apply_real_world_adjustment=False` → `real_world_adjustment_factor=1.0`
- [ ] Drop the extra `predict()` arguments; rename input columns to match the model contract
- [ ] `model.errors` → `model.metadata.errors`
- [ ] Replace `Model.from_dict` / `to_dict` / `from_url`, smartcore, the sklearn native estimator,
      and `port_to_c`
- [ ] Custom `Trainer` subclasses: add `required_extra_columns` and `split_grouping_column`
