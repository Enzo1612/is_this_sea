import json
import itertools

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import random

import numpy as np

import model

def mass_test(train_X_y, algos):
    top_models = {}
    for algo in algos:
        print(f"Testing {algo['algo']}")

        # print("Starting broad search...")
        # random_search = broad_search(algo, train_X_y, n_iter=10, folds=5)
        # print("Broad search completed.")

        # params = getCloseParams(random_search, algo["hyper"], amount=3, fill_with_random=False)

        print("Starting precise search...")
        grid_search = precise_search(algo, train_X_y, clean_hyperparameters(algo["hyper"]), cv=1)
        print("Precise search completed.")

        top_models[algo["algo"]] = {
            "model": grid_search.best_estimator_,
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_
        }

    for model in top_models:
        print("*" * 20)
        print(f"Model: {model}")
        print(f"\tBest Score: {top_models[model]['best_score']:.4f}")
        print(f"\tBest Parameters: {top_models[model]['best_params']}")
        # print(f"\tCV Results: {top_models[model]['cv_results']}\n\n")
    return top_models

def precise_search(algo, train_X_y, params, cv=5):
    grid_search = GridSearchCV(
        estimator=model.models[algo["algo"]](),
        param_grid=params,
        scoring="accuracy",
        cv=max(cv, 2),  # Needs to be at least 2
        verbose=1,
        n_jobs=-1,
        refit=True
    )
    grid_search.fit(train_X_y[0], train_X_y[1])
    return grid_search

def broad_search(algo, train_X_y, n_iter=50, folds=1):

    params = clean_hyperparameters(algo["hyper"])

    for i in range(folds):
        print(f"Broad search iteration {i + 1}/{folds} for {algo['algo']}")
        random_search = RandomizedSearchCV(
            estimator=model.models[algo["algo"]](),
            param_distributions=params,
            n_iter=n_iter,
            scoring="accuracy",
            cv=3,
            verbose=1,
            random_state=42,
            n_jobs=-1,
            refit=True
        )
        random_search.fit(train_X_y[0], train_X_y[1])

        params = getCloseParams(random_search, algo["hyper"], amount=50, fill_with_random=True)

    return random_search

def clean_hyperparameters(hyper):
    cleaned_hyper = {}
    for param, value in hyper.items():
        if value["type"] == "continuous":
            cleaned_hyper[param] = np.arange(value["range"][0], value["range"][1] + value["step"], value["step"])
        elif value["type"] == "discrete":
            cleaned_hyper[param] = value["values"]
        else:
            raise ValueError("Invalid hyperparameter type")
    return cleaned_hyper


def getCloseParams(random_search, hyper, amount=5, fill_with_random=True):
    best_params = random_search.best_params_
    close_params = {}

    for param, value in hyper.items():
        if value["type"] == "continuous":
            step = value["step"]
            close_params[param] = select_around(
                np.arange(value["range"][0], value["range"][1] + step, step),
                best_params[param],
                step,
                amount
            )
        elif value["type"] == "discrete":
            best_val = best_params.get(param)
            all_values = value["values"]
            
            if best_val is not None and best_val in all_values:
                # Start with the best value
                selected = [best_val]
                # Randomly sample remaining values
                others = [v for v in all_values if v != best_val]
                remaining_amount = amount - 1
                if len(others) > remaining_amount:
                    selected.extend(random.sample(others, remaining_amount))
                else:
                    selected.extend(others)  # Take all if not enough others
                close_params[param] = selected
            else:
                # Fallback if best_val not found
                close_params[param] = random.sample(all_values, min(amount, len(all_values)))
        else:
            raise ValueError("Invalid hyperparameter type")
    return close_params


def select_around(lst, center, step, max_size):
    candidates = []

    for x in lst:
        if (x - center) % step == 0:
            candidates.append((abs(x - center), x))

    candidates.sort(key=lambda t: t[0])
    
    return [x for _, x in candidates[:max_size]]



"""
algo = {
    "algo": "SVC",
    "hyper": {
        "C": {
            "type": "continuous",
            "step": 0.1,
            "range": [0.1, 10.0]
        },
        "kernel": {
            "type": "discrete",
            "values": ["linear", "rbf", "poly"]
        },
        "gamma": {
            "type": "discrete",
            "values": ["scale", "auto"]
        }
    }
}
"""     

def test (train_sample, test_sample, algo):
    clf, X, y = model.fitFromHisto(train_sample, algo=algo)
    _, _, test_err = model.predictFromHisto(test_sample, clf)
    train_err = model.get_train_error(clf, X, y)

    return (test_err, train_err)

def mass_test_from_json(train_X_y, json_path, model_name=None):
    with open(json_path, "r") as f:
        algos = json.load(f)

    if model_name is not None:
        algos = [algo for algo in algos if algo["algo"] == model_name]

    mass_test(train_X_y, algos)

# def mass_single_model_from_json(train_X_y, json_path, model_name):
#     with open(json_path, "r") as f:
#         algos = json.load(f)
    
#     algo = next((a for a in algos if a["algo"] == model_name), None)
#     if algo is None:
#         raise ValueError(f"Model {model_name} not found in JSON.")
    
#     return mass_test(train_X_y, [algo])

