"""
# Lookup Table Example

This example demonstrates how to convert a Routee Powertrain model into a lookup table format.
Lookup tables are useful for fast energy consumption predictions across a predefined grid of operating conditions.
"""

import routee.powertrain as pt
import numpy as np


"""
## Loading Models

First, let's load a few different models to demonstrate lookup table generation.
We'll use models with different feature sets to show the flexibility of the approach.
"""

toyota_camry = pt.load_model("2016_TOYOTA_Camry_4cyl_2WD")
tesla_model3 = pt.load_model("2022_Tesla_Model_3_RWD")
tesla_with_temp = pt.load_model("2022_Tesla_Model_3_RWD_0F_110F_steady")
"""
Let's examine the available features and targets for each model.
"""

print("Toyota Camry features:", toyota_camry.feature_set_lists)
print("Toyota Camry targets:", toyota_camry.metadata.config.target.target_name_list)
print()
print("Tesla Model 3 features:", tesla_model3.feature_set_lists)
print("Tesla Model 3 targets:", tesla_model3.metadata.config.target.target_name_list)
print()
print("Tesla with Temperature features:", tesla_with_temp.feature_set_lists)
print(
    "Tesla with Temperature targets:",
    tesla_with_temp.metadata.config.target.target_name_list,
)

"""
## Single Feature Lookup Table

Let's start with a simple single-feature lookup table using speed only.
This creates a 1D lookup table showing how energy consumption varies with vehicle speed.
"""

# Define feature parameters for speed-only lookup
speed_only_params = [
    {
        "feature_name": "speed_mph",
        "lower_bound": 5.0,
        "upper_bound": 80.0,
        "n_samples": 16,  # Every 5 mph from 5 to 80
    }
]

# Generate lookup table for Toyota Camry
camry_speed_lookup = toyota_camry.to_lookup_table(
    feature_parameters=speed_only_params,
    energy_target="gge",  # Gallons of gasoline equivalent
)

print("Single feature lookup table (first 5 rows):")
print(camry_speed_lookup.head())

"""
## Two Feature Lookup Table

Now let's create a more comprehensive 2D lookup table using both speed and grade.
This shows how energy consumption varies with both vehicle speed and road grade.
"""

# Define feature parameters for speed and grade
speed_grade_params = [
    {
        "feature_name": "speed_mph",
        "lower_bound": 25.0,
        "upper_bound": 65.0,
        "n_samples": 9,  # Every 5 mph from 25 to 65
    },
    {
        "feature_name": "grade_percent",
        "lower_bound": -6.0,
        "upper_bound": 6.0,
        "n_samples": 7,  # Every 2% grade from -6% to +6%
    },
]

# Generate lookup table for Tesla Model 3
tesla_speed_grade_lookup = tesla_model3.to_lookup_table(
    feature_parameters=speed_grade_params,
    energy_target="kwh",
)

print(f"Two feature lookup table shape: {tesla_speed_grade_lookup.shape}")
print("Sample rows:")
print(tesla_speed_grade_lookup.head(10))

"""
## Three Feature Lookup Table with Temperature

For models that include temperature, we can create a 3D lookup table.
This is particularly useful for electric vehicles where temperature significantly affects range.
"""

# Define feature parameters including temperature
temp_params = [
    {
        "feature_name": "speed_mph",
        "lower_bound": 35.0,
        "upper_bound": 55.0,
        "n_samples": 3,  # 35, 45, 55 mph
    },
    {
        "feature_name": "grade_percent",
        "lower_bound": -2.0,
        "upper_bound": 4.0,
        "n_samples": 4,  # -2%, 0%, 2%, 4%
    },
    {
        "feature_name": "ambient_temp_f",
        "lower_bound": 20.0,
        "upper_bound": 80.0,
        "n_samples": 4,  # 20°F, 40°F, 60°F, 80°F
    },
]

# Generate lookup table with temperature
tesla_temp_lookup = tesla_with_temp.to_lookup_table(
    feature_parameters=temp_params,
    energy_target="kwh",
)

print(f"Three feature lookup table shape: {tesla_temp_lookup.shape}")
print("Sample rows showing temperature effects:")
print(tesla_temp_lookup.head(12))

"""
## Practical Usage: Interpolation for Route Prediction

Lookup tables can be used for fast interpolation to predict energy consumption for specific driving conditions.
Here's how you might use a lookup table for route prediction:
"""


# Example: Using lookup table for fast prediction
def interpolate_energy_from_lookup(lookup_table, speed, grade=None, temp=None):
    if grade is None and temp is None:
        # 1D interpolation for speed only
        return np.interp(speed, lookup_table["speed_mph"], lookup_table.iloc[:, -1])
    else:
        # For multi-dimensional interpolation, you'd typically use scipy.interpolate
        # This is a simplified example
        closest_row = lookup_table.iloc[
            (lookup_table["speed_mph"] - speed).abs().argsort()[:1]
        ]
        return closest_row.iloc[0, -1]


# Example usage
example_speed = 42.5
predicted_energy = interpolate_energy_from_lookup(camry_speed_lookup, example_speed)
print(
    f"Interpolated energy consumption at {example_speed} mph: {predicted_energy:.4f} gge/mile"
)

"""
## Best Practices and Considerations

When creating lookup tables, consider:

1. **Resolution vs. Size**: More samples mean higher accuracy but larger tables
2. **Feature Ranges**: Ensure your lookup table covers the expected operating conditions
3. **Interpolation**: For values between grid points, you'll need interpolation
4. **Memory Usage**: Large multi-dimensional tables can consume significant memory
5. **Update Frequency**: Lookup tables are static - update them when models change

## Use Cases

Lookup tables are particularly useful for:
- Real-time applications requiring fast predictions
- Embedded systems with limited computational resources
- Integration with external systems that can't run Python models directly
"""
