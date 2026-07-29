"""
# Model Training

If you have your own ground truth energy data, you can train a custom RouteE powertrain model.

You'll want to make sure you've installed the proper dependencies that are not installed by default when you do a pip install.

In this example, we'll use the scikit-learn based estimators which you can install by doing:

```bash
pip install routee.powertrain[scikit]
```

RouteE Powertrain v2 ships three trainer pipelines:

- `SklearnRandomForestTrainer` (extra: `scikit`) — produces an `ONNXEstimator` via `skl2onnx`.
- `NGBoostTrainer` (extra: `ngboost`) — probabilistic estimator that emits both a point estimate and a per-link standard deviation.
- `CNNTrainer` (extra: `pytorch`) — sequence-aware 1D CNN that exports to ONNX with an `InputSpec` describing the sliding window.

We'll use the Random Forest trainer below.
"""

import routee.powertrain as pt

from routee.powertrain.trainers.sklearn_random_forest import SklearnRandomForestTrainer

"""
For demonstration purposes, we'll use a very small set of training data.
You can access this dataset yourself [here](https://github.com/NatLabRockies/routee-powertrain/blob/main/tests/routee-powertrain-test-data/sample_train_data.csv)
"""
import pandas as pd

df = pd.read_csv("../../tests/routee-powertrain-test-data/sample_train_data.csv")
df.head()
"""
This dataframe represents a set of road network links (i.e. roads) in which we've already computed the energy consumption over. In this case, we've use the Fastsim software to simulate a vehicle driving over a high resolution drive cycle and then have aggregated everything up to the link level. We also have link level attributes like average driving speed in mile per hour (`speed`), road gradient as a decimal (`grade`), road distance in miles (`miles`) and road classification as a integer category (`road_class`). Lastly, we have a trip identifier column (`trip_id`) which is only 1 in this case, represeting a single trip taken by this vehicle.

Ok, onto setting up the training pipeline.

First, we need to tell the trainer what features we want to use for the internal estimator (a Random Forest in this case).
We define a single `FeatureSet` that describes all the features the model will be trained on. In this case, we'll use `speed_mph` and `grade_dec`.
"""
feature_set = [
    pt.DataColumn(name="speed_mph", units="mph"),
    pt.DataColumn(name="grade_dec", units="decimal"),
]
features = feature_set
"""
Note that we didn't include the distance column in the feature set. RouteE Powertrain always requires distance information, so we provide a separate designation for it in the training configuration. Let's define our distance column:
"""
distance = pt.DataColumn(name="miles", units="miles")
"""
Now, we need to define our energy target which is gallons of gasoline simualted by Fastsim:
"""
energy_target = pt.DataColumn(
    name="gallons_fastsim",
    units="gallons_gasoline",
)
"""
We also need to decide how we want to predict the energy.
We have two options: "rate" or "raw".
"rate" will take our energy values and divide them by the distance column to arrive at and energy rate.
Then, the estimator will be trained to predict the rate value (without using distance as a feature) and then the model will multiply the rate value by the incoming link distance to give a final raw energy value.
This can be useful in your training data is sparse as it allows the model to be flexible to distance.
"raw" will tell the estimator to predict the energy on the link directly, using distance as an explicit feature.
This can be more robust for situations where the energy rate on a link might vary with respect to distance but can lead to weird results if there are not a good representation of different distance values in the training dataset.
In our case we'll use "rate" since our training data is very sparse.
"""
predict_method = "rate"
"""
Finally, we can build a model configuration that we can pass to the trainer. This will also include things like the vehicle powertrain type and a model name
"""
config = pt.ModelConfig(
    vehicle_description="Test Vehicle",
    powertrain_type=pt.PowertrainType.ICE,
    feature_set=features,
    distance=distance,
    target=energy_target,
    make="test",
    model="vehicle",
    year=2024,
    test_size=0.2,
    predict_method=predict_method,
)
"""
Now we build the random forest trainer and give it the desired parameters
"""
trainer = SklearnRandomForestTrainer(
    max_depth=10, min_samples_split=10, n_estimators=20, cores=4
)
"""
All trainers have a `train` method on them which will return a trained vehicle model
"""
test_vehicle = trainer.train(df, config)
"""
With the model trained, we can inspect the errors for each estimator type and energy target (note, it's possible that we could have given multiple energy targets to the trainer, like gasoline and electricity for a plug-in hybrid vehicle)
"""
test_vehicle.metadata.errors
"""
While this training dataset is far too small to draw real conclusions, these metrics can give you an idea of how well the model performed on a holdout test set (20% of the training data as we specificed by the `test_size` parameter in the configuration. 
"""
"""
Now, we can write the model to a `.zip` archive, `.tar.gz` archive, or a flat directory (auto-detected from the path's suffix):

```python
test_vehicle.to_file("Test_Vehicle.zip")        # ZIP archive
test_vehicle.to_file("Test_Vehicle.tar.gz")     # tar archive
test_vehicle.to_file("Test_Vehicle/")           # flat directory
```

The saved artifact contains a `metadata.json` and a binary estimator file (e.g. `model.onnx`). Reload it with `pt.load_model("Test_Vehicle.zip")`.
"""
