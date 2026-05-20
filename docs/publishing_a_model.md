# Publishing a Trained Model

After training a model with one of the `Trainer` classes, you can save it
into the v2 registry layout that `LocalRegistry` and `S3Registry` understand.
This page walks through writing a trained model into a local registry, picking
a `config_slug`, and loading it back through `pt.load_model()`.

## The v2 Registry Layout

Every model in the registry lives in a five-segment path:

```
<registry_root>/v2/<make>/<model>/<year>/<config_slug>/v<N>/
    metadata.json
    model.onnx        # (or another binary, depending on the estimator)
```

The bundled example at
`routee/powertrain/resources/bundled_registry/v2/toyota/camry_4cyl_2wd/2016/rf_default/v1/`
is a concrete reference for what the on-disk layout looks like.

## Picking a `config_slug`

The `config_slug` disambiguates multiple trained configurations for the same
vehicle and year. The full feature composition and estimator architecture live
inside `metadata.json`; the slug is a short human-readable handle.

The informal convention used by existing slugs (`rf_default`, `cnn_5link`,
`rf_steady_temp`) is:

- **Format**: lowercase snake_case, `<architecture>_<descriptor>`.
- **Use `_default`** when the model is the canonical configuration for that
  architecture on that vehicle.
- **The descriptor encodes the meaningful axis of variation** — feature
  subset (`rf_steady_temp`), input window (`cnn_5link`), or training-data
  scope. It is _not_ the date, the trainer's initials, or a version number.
- **Do not put version info in the slug.** Re-training the same configuration
  bumps `v<N>` (e.g. `rf_default/v2`). A materially different feature set or
  architecture should use a _new_ slug and start at `v1`.

## Train and Publish

`Model.save_to_registry()` writes a trained model into the canonical layout in
one call. It pulls `make`, `model`, and `year` from `model.metadata.config`,
so make sure those fields on your `ModelConfig` are correct before training.

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

model_id = model.save_to_registry(
    registry_root="./my_local_registry",
    config_slug="rf_default",
    version=1,
)
print(model_id)  # test/sedan/2024/rf_default/v1
```

This creates `./my_local_registry/v2/test/sedan/2024/rf_default/v1/` with
`metadata.json` and `model.onnx` inside. If the directory already has files,
the call raises `FileExistsError` — bump `version` or pass `overwrite=True`.

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

# Omit the trailing v<N> to get the latest version.
model = pt.load_model("test/sedan/2024/rf_default")
```

You can also list and query what's in the registry the same way you would
with the default S3 registry:

```python
pt.list_available_models()
pt.query_available_models(make="test", powertrain_type="ICE")
```

## What's in `metadata.json`

`metadata.json` is written automatically by the save call. Most of its
contents come from your `ModelConfig` and the trainer:

| Field                                                      | Source                                           |
| ---------------------------------------------------------- | ------------------------------------------------ |
| `schema_version`                                           | `routee.powertrain.core.metadata.SCHEMA_VERSION` |
| `estimator_type`                                           | The trainer (e.g. `"ONNXEstimator"`)             |
| `architecture_tag`                                         | The trainer (e.g. `"random_forest"`, `"cnn"`)    |
| `input_spec`                                               | The estimator (lookback / grouping column / pad) |
| `model_file`                                               | Filename of the binary (e.g. `"model.onnx"`)     |
| `errors`                                                   | Computed during `Trainer.train()`                |
| `routee_version`                                           | Package version at save time                     |
| `config.make / model / year`                               | From `ModelConfig` — drives the path             |
| `config.powertrain_type`                                   | From `ModelConfig`                               |
| `config.vehicle_description`                               | From `ModelConfig`                               |
| `config.feature_set / target / distance`                   | From `ModelConfig`                               |
| `config.mass_lbs / fuel_type / drivetrain / engine / trim` | From `ModelConfig` (optional)                    |

See `routee/powertrain/core/metadata.py` for the full schema and
`routee/powertrain/core/model_config.py` for the `ModelConfig` fields.

## Sharing Your Model

Once your model loads cleanly from a local registry, that same directory tree
is what gets uploaded to the shared S3 bucket. Publishing to the shared
bucket requires write access and is currently a maintainer step — package the
`<registry_root>/v2/...` directory (or zip it) and hand it off. The
maintainer-side upload and index-refresh scripts live at
`scripts/upload_to_s3.py` and `scripts/build_s3_index.py`.
