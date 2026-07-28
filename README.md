# <img src="docs/images/routeelogo.png" alt="Routee Powertrain" width="100"/>

<div align="left">
    <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue"/>
  <a href="https://pypi.org/project/routee.powertrain/">
    <img src="https://img.shields.io/pypi/v/routee.powertrain" alt="PyPi Latest Release"/>
  </a>
</div>

## Overview

RouteE-Powertrain is a Python package that allows users to work with a set of pre-trained mesoscopic vehicle energy prediction models for a varity of vehicle types. Additionally, users can train their own models if "ground truth" energy consumption and driving data are available. RouteE-Powertrain models predict vehicle energy consumption over links in a road network, so the features considered for prediction often include traffic speeds, road grade, turns, etc.

The typical user will utilize RouteE's catalog of pre-trained models. Currently, the
catalog consists of light-duty vehicle models, including conventional gasoline, diesel,
hybrid electric (HEV), plugin hybrid electric (PHEV) and battery electric (BEV). These models can be applied to link-level driving data (in the form
of [pandas](https://pandas.pydata.org/) dataframes) to output energy consumption predictions.

Users that wish to train new RouteE models can do so. The model training function of RouteE enables users to use their
own drive-cycle data, powertrain modeling system, and road network data to train custom models.

## Quickstart

RouteE Powertrain is available on PyPI and can be installed with `pip`:

```bash
pip install pip --upgrade
pip install routee.powertrain
```

If `pip` is unavailable, use `pip3`:

```bash
pip3 install pip --upgrade
pip3 install routee.powertrain
```

(For more detailed instructions, see [here](https://natlabrockies.github.io/routee-powertrain/installation.html))

Then, you can import the package and use a pre-trained model from the RouteE model catalog:

```python
import pandas as pd
import routee.powertrain as pt

# Query for a specific model
print(pt.query_available_models(make="chevrolet", model="bolt", year=2017))

# Load a pre-trained model
model = pt.load_model("chevrolet/bolt_bev/2017/rf_c3326385/v1")

# Inspect the model to see what it expects for input
print(model)

# Predict energy consumption for a set of road links
links_df = pd.DataFrame(
    {
        "distance": [0.1, 0.2, 0.3], # miles
        "speed_mph": [30, 40, 50], # mph
        "grade_percent": [-5.0, 0.0, 5.0], # percent
    }
)

energy_result = model.predict(links_df)
```

## Upgrading from v1

RouteE Powertrain 2.0 is a breaking release. It was previously published as
**`nrel.routee.powertrain`**; it is now **`routee.powertrain`**, and the import path changed
to match:

```bash
pip uninstall nrel.routee.powertrain
pip install routee.powertrain
```

```diff
-import nrel.routee.powertrain as pt
+import routee.powertrain as pt
```

Model names, the model file format, and much of the `Model` API changed as well. See the
[migration guide](https://natlabrockies.github.io/routee-powertrain/migrating_from_v1.html)
for the full list, and [CHANGELOG.md](CHANGELOG.md) for everything in 2.0.0.

Custom v1 `.json` models can be converted in place:

```bash
routee-powertrain convert-v1 MyModel.json out/ --make toyota --model camry --year 2016
```
