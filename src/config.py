config = {
    "random_state": 42,
    "logistic_regression": {
        "max_iter": 1000,
        "solver": "newton-cholesky",
        "penalty": "l2",
        "class_weight": None,
    },
    "random_forest": {
        "max_depth": 100,
        "min_samples_leaf": 10,
        "max_leaf_nodes": 100,
        "n_estimators": 100,
    },
}
