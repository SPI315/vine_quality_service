import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import joblib

from data import load_data, split_data
from model import train, test

from clearml import Task

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
        "min_samples_leaf": 30,
        "max_leaf_nodes": 10,
        "n_estimators": 10,
    },
}


if __name__ == "__main__":
    data_path = os.path.join(sys.path[0], "..", "data", "winequality-red.csv")

    X, y = load_data(data_path, "quality")
    X_train, X_test, y_train, y_test = split_data(X, y)

    n = 1
    for max_depth in [100, 500]:
        for min_samples_leaf in [10, 100]:
            for max_leaf_nodes in [10, 100]:
                for n_estimators in [100, 500]:

                    config["random_forest"]["max_depth"] = max_depth
                    config["random_forest"]["min_samples_leaf"] = min_samples_leaf
                    config["random_forest"]["n_estimators"] = n_estimators
                    config["random_forest"]["max_leaf_nodes"] = max_leaf_nodes

                    model_1 = RandomForestClassifier(
                        random_state=config["random_state"],
                        max_depth=config["random_forest"]["max_depth"],
                        min_samples_leaf=config["random_forest"]["min_samples_leaf"],
                        n_estimators=config["random_forest"]["n_estimators"],
                        max_leaf_nodes=config["random_forest"]["max_leaf_nodes"],
                    )

                    task_1 = Task.init(
                        project_name="WineQuality", task_name=f"RF_test_{n}", tags="RF"
                    )
                    # Случайный лес с оценкой на тесте:
                    task_1.connect(config["random_forest"])
                    train(model_1, X_train, y_train)
                    test(model_1, X_test, y_test, task_1)
                    joblib.dump(model_1, "rf.pkl", compress=True)
                    task_1.close()

                    task_2 = Task.init(
                        project_name="WineQuality", task_name=f"RF_train_{n}", tags="RF"
                    )
                    # Случайный лес с оценкой на трейне:
                    task_2.connect(config["random_forest"])
                    test(model_1, X_train, y_train, task_2)
                    joblib.dump(model_1, "rf.pkl", compress=True)
                    task_2.close()

                    n += 1

    k = 1
    for max_iter in [100, 500, 1000]:
        for solver in ["sag", "saga", "newton-cholesky"]:
            for class_w in [None, "balanced"]:

                config["logistic_regression"]["max_iter"] = max_iter
                config["logistic_regression"]["solver"] = solver
                config["logistic_regression"]["class_weight"] = class_w

                model_2 = LogisticRegression(
                    random_state=config["random_state"],
                    max_iter=config["logistic_regression"]["max_iter"],
                    solver=config["logistic_regression"]["solver"],
                    penalty=config["logistic_regression"]["penalty"],
                    class_weight=config["logistic_regression"]["class_weight"],
                )

                task_3 = Task.init(
                    project_name="WineQuality", task_name=f"LR_test_{k}", tags="LR"
                )
                # Логистическая регрессия:
                task_3.connect(config["logistic_regression"])
                train(model_2, X_train, y_train)
                test(model_2, X_test, y_test, task_3)
                joblib.dump(model_2, "logreg.pkl", compress=True)
                task_3.close()
                k += 1
