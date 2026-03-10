"""One-time migration script to convert bundled models from multi-estimator
to single-estimator format. Picks the estimator with speed_mph and grade_percent features."""

import json


TARGET_FEATURES = {"speed_mph", "grade_percent"}


def migrate_model(filepath):
    with open(filepath, "r") as f:
        old = json.load(f)

    # Find the feature set with exactly speed_mph and grade_percent
    feature_sets = old["metadata"]["config"]["feature_sets"]
    best_idx = None
    for i, fs in enumerate(feature_sets):
        feature_names = {f["name"] for f in fs["features"]}
        if feature_names == TARGET_FEATURES:
            best_idx = i
            break

    if best_idx is None:
        raise ValueError(
            f"Could not find feature set with {TARGET_FEATURES} in {filepath}"
        )

    best_feature_set = feature_sets[best_idx]

    # Sort feature names to build the feature set id (same as feature_names_to_id)
    feature_names = sorted([f["name"] for f in best_feature_set["features"]])
    feature_set_id = "&".join(feature_names)

    print(f"  Selected feature set: {feature_set_id} ({len(TARGET_FEATURES)} features)")

    # Get the matching estimator
    estimator_entry = old["all_estimators"][feature_set_id]
    estimator_dict = estimator_entry["estimator"]
    estimator_type = estimator_entry["estimator_constructor_type"]

    # Get the matching errors
    old_errors = old["errors"]["estimator_errors"][feature_set_id]

    # Build new config without feature_sets (replaced by feature_set)
    old_config = old["metadata"]["config"]
    new_config = {k: v for k, v in old_config.items() if k != "feature_sets"}
    new_config["feature_set"] = best_feature_set

    # Build new format
    new = {
        "metadata": {
            "config": new_config,
            "routee_version": old["metadata"]["routee_version"],
        },
        "errors": {
            "estimator_errors": old_errors,
        },
        "estimator": estimator_dict,
        "estimator_type": estimator_type,
    }

    with open(filepath, "w") as f:
        json.dump(new, f)

    print("  Migrated successfully")


if __name__ == "__main__":
    models = [
        "nrel/routee/powertrain/resources/default_models/2016_TOYOTA_Camry_4cyl_2WD.json",
        "nrel/routee/powertrain/resources/default_models/2017_CHEVROLET_Bolt.json",
    ]

    for m in models:
        print(f"Migrating {m}...")
        migrate_model(m)
        print()
