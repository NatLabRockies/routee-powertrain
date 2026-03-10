import numpy as np


def serialize_tree(tree):
    serialized_tree = tree.__getstate__()
    dtypes = serialized_tree["nodes"].dtype
    serialized_tree["nodes"] = serialized_tree["nodes"].tolist()
    serialized_tree["values"] = serialized_tree["values"].tolist()

    return serialized_tree, dtypes


def deserialize_tree(tree_dict, n_features, n_classes, n_outputs):
    try:
        from sklearn.tree._tree import Tree
    except ImportError:
        raise ImportError(
            "Please install scikit-learn to use the SmartCoreRandomForest estimator."
        )
    tree_dict["nodes"] = [tuple(lst) for lst in tree_dict["nodes"]]

    names = [
        "left_child",
        "right_child",
        "feature",
        "threshold",
        "impurity",
        "n_node_samples",
        "weighted_n_node_samples",
    ]
    tree_dict["nodes"] = np.array(
        tree_dict["nodes"],
        dtype=np.dtype({"names": names, "formats": tree_dict["nodes_dtype"]}),
    )
    tree_dict["values"] = np.array(tree_dict["values"])

    tree = Tree(n_features, np.array([n_classes], dtype=np.intp), n_outputs)
    tree.__setstate__(tree_dict)

    return tree


def serialize_decision_tree_regressor(model):
    tree, dtypes = serialize_tree(model.tree_)
    serialized_model = {
        "meta": "decision-tree-regression",
        "feature_importances_": model.feature_importances_.tolist(),
        "max_features_": model.max_features_,
        "n_features_in_": model.n_features_in_,
        "n_outputs_": model.n_outputs_,
        "tree_": tree,
    }

    tree_dtypes = []
    for i in range(0, len(dtypes)):
        tree_dtypes.append(dtypes[i].str)

    serialized_model["tree_"]["nodes_dtype"] = tree_dtypes

    return serialized_model


def deserialize_decision_tree_regressor(model_dict):
    try:
        from sklearn.tree import DecisionTreeRegressor
    except ImportError:
        raise ImportError(
            "Please install scikit-learn to use the SmartCoreRandomForest estimator."
        )

    deserialized_decision_tree = DecisionTreeRegressor()

    deserialized_decision_tree.max_features_ = model_dict["max_features_"]
    deserialized_decision_tree.n_features_in_ = model_dict["n_features_in_"]
    deserialized_decision_tree.n_outputs_ = model_dict["n_outputs_"]

    tree = deserialize_tree(
        model_dict["tree_"], model_dict["n_features_in_"], 1, model_dict["n_outputs_"]
    )
    deserialized_decision_tree.tree_ = tree

    return deserialized_decision_tree


def serialize_random_forest_regressor(model):
    serialized_model = {
        "meta": "rf-regression",
        "max_depth": model.max_depth,
        "min_samples_split": model.min_samples_split,
        "min_samples_leaf": model.min_samples_leaf,
        "min_weight_fraction_leaf": model.min_weight_fraction_leaf,
        "max_features": model.max_features,
        "max_leaf_nodes": model.max_leaf_nodes,
        "min_impurity_decrease": model.min_impurity_decrease,
        "n_features_in_": model.n_features_in_,
        "n_outputs_": model.n_outputs_,
        "estimators_": [
            serialize_decision_tree_regressor(decision_tree)
            for decision_tree in model.estimators_
        ],
        "params": model.get_params(),
    }

    if "oob_score_" in model.__dict__:
        serialized_model["oob_score_"] = model.oob_score_
    if "oob_decision_function_" in model.__dict__:
        serialized_model["oob_prediction_"] = model.oob_prediction_.tolist()

    return serialized_model


def deserialize_random_forest_regressor(model_dict):
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        raise ImportError(
            "Please install scikit-learn to use the SmartCoreRandomForest estimator."
        )
    model = RandomForestRegressor(**model_dict["params"])
    estimators = [
        deserialize_decision_tree_regressor(decision_tree)
        for decision_tree in model_dict["estimators_"]
    ]
    model.estimators_ = np.array(estimators)

    model.n_features_in_ = model_dict["n_features_in_"]
    model.n_outputs_ = model_dict["n_outputs_"]
    model.max_depth = model_dict["max_depth"]
    model.min_samples_split = model_dict["min_samples_split"]
    model.min_samples_leaf = model_dict["min_samples_leaf"]
    model.min_weight_fraction_leaf = model_dict["min_weight_fraction_leaf"]
    model.max_features = model_dict["max_features"]
    model.max_leaf_nodes = model_dict["max_leaf_nodes"]
    model.min_impurity_decrease = model_dict["min_impurity_decrease"]

    if "oob_score_" in model_dict:
        model.oob_score_ = model_dict["oob_score_"]
    if "oob_prediction_" in model_dict:
        model.oob_prediction_ = np.array(model_dict["oob_prediction_"])

    return model
