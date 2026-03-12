"""
# Model Prediction

RouteE models can be loaded from a large library of pre-trained models. Conventional gasoline (CV), hybrid electric (HEV), plug-in hybrid electric (PHEV), and battery electric (BEV) powertrain types are all available.

__A note on PHEVs__: Plug-in hybrids have two general operating modes 1) "Charge Depleting" or "EV" mode, where the vehicle relies only on energy from the battery to power the motor and 2) "Charge Sustaining" or "Hybrid" mode, where the vehicle operates like a typical parallel hybrid, using a combination of the combustion energy and electric motor for tractive effort and regenerative braking. Since the operating mode depends on battery state-of-charge and driver decisions, pre-trained RouteE-Powertrain models for both operating modes are provided for all PHEVs and it is up to the user to decide which is most appropriate for a particular application.
"""
import routee.powertrain as pt
pt.list_available_models()
camry = pt.load_model("2016_TOYOTA_Camry_4cyl_2WD")
"""
After loading a model, we can inspect it to see what features (and units) the model expects. 
RouteE Powertrain models can have multiple estimators under the hood which have been trained on different feature sets.
For example, there might be an estimator that takes just `speed` as a link feature and another that takes in `speed` and `grade`.
This can be useful if you have sparse data for one feature (like grade) but still want to predict energy consumption.
"""
camry
"""
Now, let's predict energy consumption over a sample route.
RouteE Powertrain expects the inputs to be a pandas dataframe in which each row represents a road network link.
There is a sample route included with the package that we'll use for demonstration.
"""
sample_route = pt.load_sample_route()
sample_route
"""
If we just pass in the links DataFrame without any other information, the model will assume we want to use all the features and in this case will look for an internal estimator with features to match all the columns.

Based on the model summary shown above, we do have an estimator that takes in the link features `speed_mph` and `grade_percent` with a distance of `distance` and so it will automatically select that estimator when we predict.
"""
camry.predict(sample_route)
"""
If we want to use a different estimator, we can tell the predict method to only use a subset of the features. In this case, we'll tell the model to only use speed.
"""
camry.predict(sample_route, feature_columns=["speed_mph"])
"""
## Model Visualization

There are a few different functions we can visualize what a model is predicting over a range of inputs.

The first is the `visualize_features` function that sweeps a feature over a range and plots the results.
In order to use this we first have to define what ranges the features should be considered.
"""
feature_ranges = {
    "speed_mph": {"lower": 2, "upper": 100, "n_samples": 50},
    "grade_percent": {"lower": -20.0, "upper": 20.0, "n_samples": 50}
}
results = pt.visualize_features(camry, feature_ranges)
"""
We can also look at two features simultaneously with the `contour_plot` function. 
"""
pt.contour_plot(camry, x_feature="speed_mph", y_feature="grade_percent", feature_ranges=feature_ranges)
