# What is RouteE?

RouteE-Powertrain is a Python package that allows users to work with a set of pre-trained mesoscopic vehicle energy prediction models for a varity of vehicle types. Additionally, users can train their own models if "ground truth" energy consumption and driving data are available. RouteE-Powertrain models predict vehicle energy consumption over links in a road network, so the features considered for prediction often include traffic speeds, road grade, turns, etc. Common applications of RouteE-Powertrain are energy-aware ("eco") routing (like [RouteE-Compass](https://nrel.github.io/routee-compass/intro.html)), energy accounting in mesoscopic simulations, and range estimation (especially for EVs). The diagrams below illustrate the logic and data flows for training custom RouteE-Powertrain models and performing prediction with previously trained models.

## Training

![image](https://github.com/NREL/routee-powertrain/assets/4818940/f5a89f64-241b-494b-9fe6-e13a34595da0)

Training new RouteE-Powertrain models requires a set of link aggregate driving data with energy consumption on each link in the road network. Often this data comes from high-frequency GPS or telematics data collected by dedicated loggers or from connected vehicles that are always streaming telematics data. The energy consumption can either be vehicle reported/measured or simulated using a powertrain simulation software like [NREL's FASTSim](https://github.com/NREL/fastsim).

## Prediction

![image](https://github.com/NREL/routee-powertrain/assets/4818940/b3a1d1af-5060-4bf4-a576-b0b11ffc9424)

In application, trained RouteE-Powertrain models expect link features as inputs and return predicted energy consumption for a particular vehicle over a link with the particular feature set. The RouteE developers maintain a separate repository for previously trained RouteE-Powertrain models, available for prediction "off the shelf". To see which models are available you can use the `pt.list_available_models()` function:

```python
import routee.powertrain as pt

pt.list_available_models()
```

This returns a list of `ModelId` objects. For more detail (including feature names, errors, and powertrain type), use `query_available_models`:

```python
# Query all available Toyota models
results = pt.query_available_models(make="toyota")

# Query with multiple filters
results = pt.query_available_models(make="chevrolet", model="bolt", year=2017)
```

To predict with any of these models you can use the `pt.load_model()` function. Pass the registry path string (or a `ModelId` object) to load a model:

```python
camry = pt.load_model('toyota/camry_ice/2016/rf_c3326385/v1')
bolt = pt.load_model('chevrolet/bolt_bev/2017/rf_c3326385/v1')
```

## Model Registry

Model discovery and retrieval go through a `ModelRegistry` abstraction. By default, models are fetched from a public S3 bucket; you can also point at a local directory tree by setting the `ROUTEE_REGISTRY_BACKEND` environment variable.

A `ModelId` is structured as `<make>/<vehicle_slug>/<year>/<config_slug>/v<N>`. Both slugs are _derived_ from the model's metadata: the `vehicle_slug` is the model name plus the coarse powertrain family (e.g. `camry_ice`, `bolt_bev`, `volt_phev` — both PHEV operating modes collapse to one `phev` family), and the `config_slug` disambiguates multiple trained configurations for the same vehicle as `<architecture>_<variant?>_<feature_hash>` (e.g. `rf_c3326385`, `ngb_stochastic_96224f1f`) — so different architectures, feature sets, or `variant` labels each get a distinct slug automatically. Omit the trailing `v<N>` to load the latest version. The version-less part (`<make>/<vehicle_slug>/<year>/<config_slug>`) is a model's `ModelKey`, exposed on any loaded or trained model as `model.key`.

For a step-by-step walkthrough of training a new model and writing it into the registry layout, see [Publishing a Model](publishing_a_model.md).

| Variable                     | Default                           | Meaning                               |
| ---------------------------- | --------------------------------- | ------------------------------------- |
| `ROUTEE_REGISTRY_BACKEND`    | `s3`                              | Backend to use: `s3` or `local`       |
| `ROUTEE_S3_BUCKET`           | `routeecore-bucket`               | S3 bucket name                        |
| `ROUTEE_S3_REGION`           | `us-west-2`                       | AWS region                            |
| `ROUTEE_S3_ROOT_PREFIX`      | `routee-powertrain-model-library` | S3 key prefix                         |
| `ROUTEE_SCHEMA_VERSION`      | `v2`                              | On-disk schema version                |
| `ROUTEE_LOCAL_REGISTRY_ROOT` | (bundled registry)                | Filesystem root for the local backend |

`query_available_models` supports fuzzy string matching by default (`fuzzy=True`, `fuzzy_threshold=80`) and accepts the additional filters `powertrain_type`, `fuel_type`, `drivetrain`, `engine`, `trim`, `feature_names`, and `custom_filters`. Pass `version_strategy="all"` to see every version of every model instead of just the latest.
