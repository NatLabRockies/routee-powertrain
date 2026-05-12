# RouteE Powertrain

RouteE-Powertrain is a Python package that allows users to work with a set of pre-trained mesoscopic vehicle energy prediction models for a varity of vehicle types. Additionally, users can train their own models if "ground truth" energy consumption and driving data are available. RouteE-Powertrain models predict vehicle energy consumption over links in a road network, so the features considered for prediction often include traffic speeds, road grade, turns, etc.

## Quickstart

RouteE Powertrain is available on PyPI and can be installed with `pip`:

```bash
pip install routee.powertrain
```

Then, you can import the package and use a pre-trained model from the RouteE model catalog:

```python
import pandas as pd
import routee.powertrain as pt

# List the available pre-trained models (returns a list of ModelId objects)
print(pt.list_available_models())

# [
#   toyota/camry_4cyl_2wd/2016/rf_default/v1,
#   chevrolet/bolt/2017/rf_default/v1,
#   ...
# ]

# You can also query available models with filters for more detail
results = pt.query_available_models(make="toyota", model="camry")

# Load a pre-trained model using its registry path
model = pt.load_model("toyota/camry_4cyl_2wd/2016/rf_default/v1")

# Inspect the model to see what it expects for input
print(model)

# ========================================
# Model Summary
# --------------------
# Vehicle description: 2016_TOYOTA_Camry_4cyl_2WD trained July 2024
# Powertrain type: ICE
# ========================================
# Estimator Summary
# --------------------
# Feature: speed_mph (mph)
# Feature: grade_percent (percent)
# Distance: distance (miles)
# Target: gge (gallons gasoline)
# Raw Predicted Consumption: 30.289 (miles/gallons gasoline)
# Real World Predicted Consumption: 25.977 (miles/gallons gasoline)
# Predict Method: RATE
# ========================================

# Predict energy consumption for a set of road links
links_df = pd.DataFrame(
    {
        "distance": [0.1, 0.2, 0.3], # miles
        "speed_mph": [30, 40, 50],   # mph
        "grade_percent": [-5.0, 0, 5.0], # percent
    }
)

energy_result = model.predict(links_df)
```
