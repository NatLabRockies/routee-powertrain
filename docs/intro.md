# RouteE Powertrain

RouteE-Powertrain is a Python package that allows users to work with a set of pre-trained mesoscopic vehicle energy prediction models for a variety of vehicle types. Additionally, users can train their own models if "ground truth" energy consumption and driving data are available. RouteE-Powertrain models predict vehicle energy consumption over links in a road network, so the features considered for prediction often include traffic speeds, road grade, turns, etc.

## Quickstart

RouteE Powertrain is available on PyPI and can be installed with `pip`:

```bash
pip install nrel.routee.powertrain
```

Then, you can import the package and use a pre-trained model from the RouteE model catalog:

```python
import pandas as pd
import nrel.routee.powertrain as pt

# Print the available pre-trained models
print(pt.list_available_models(local=True, external=True))

# [
#   '2016_TOYOTA_Camry_4cyl_2WD',
#   '2017_CHEVROLET_Bolt',
#   '2012_Ford_Focus',
#   ...
# ]

# Load a pre-trained model
model = pt.load_model("2016_TOYOTA_Camry_4cyl_2WD")

# Inspect the model to see what it expects for input
print(model)

# ========================================
# Model Summary
# --------------------
# Vehicle description: 2016_TOYOTA_Camry_4cyl_2WD
# Powertrain type: ICE
# Number of estimators: 2
# ========================================
# Estimator Summary
# --------------------
# Feature: speed_mph (mph)
# Distance: miles (miles)
# Target: gge (gallons_gasoline)
# Raw Predicted Consumption: 29.856 (miles/gallons_gasoline)
# Real World Predicted Consumption: 25.606 (miles/gallons_gasoline)
# ========================================
# Estimator Summary
# --------------------
# Feature: speed_mph (mph)
# Feature: grade_dec (decimal)
# Distance: miles (miles)
# Target: gge (gallons_gasoline)
# Raw Predicted Consumption: 29.845 (miles/gallons_gasoline)
# Real World Predicted Consumption: 25.596 (miles/gallons_gasoline)
# ========================================

# Predict energy consumption for a set of road links
links_df = pd.DataFrame(
    {
        "distance": [0.1, 0.2, 0.3], # miles
        "speed_mph": [30, 40, 50], # mph
        "grade_percent": [-0.5, 0, 0.5], # percent
    }
)

energy_result = model.predict(links_df)
```
