import json

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

import random

import numpy as np
from sklearn.pipeline import Pipeline

import model

def mass_test(train_X_y, algos):
    top_models = {}
    for algo in algos:
        print(f"Testing {algo['algo']}")

        # for i in [j/100 for j in range(50, 100, 5)]:
        for i in [0.95]:  # Only test with PCA=0.95 for now
            print(f"Using PCA={i}...")

            print("Starting broad search...")
            random_search = broad_search(algo, train_X_y, n_iter=10, folds=5, pca=i)
            print("Broad search completed.")

            params = getCloseParams(random_search, algo["hyper"], amount=10, fill_with_random=False)
            # params = clean_hyperparameters(params)

            print("Starting precise search...")
            grid_search = precise_search(algo, train_X_y, params, cv=1, pca=i)
            print("Precise search completed.")

            top_models[f"{algo['algo']}_{i}"] = {
                "model": grid_search.best_estimator_,
                "best_params": grid_search.best_params_,
                "best_score": grid_search.best_score_,
                "cv_results": grid_search.cv_results_
            }

            print("*" * 20)
            print(f"Best Score for PCA={i}: {grid_search.best_score_:.4f}")
            print(f"Best Parameters for PCA={i}: {grid_search.best_params_}\n\n")
            print("\n", "-" * 50, "\n", sep="")

    for model in top_models:
        print("*" * 20)
        print(f"Model: {model}")
        print(f"\tBest Score: {top_models[model]['best_score']:.4f}")
        print(f"\tBest Parameters: {top_models[model]['best_params']}")
    return top_models

def precise_search(algo, train_X_y, params, cv=5, pca=0.95):
    grid_search = GridSearchCV(
        estimator=model.models[algo["algo"]](),
        param_grid=params,
        scoring="accuracy",
        cv=max(cv, 2),  # Needs to be at least 2
        verbose=1,
        n_jobs=-1,
        refit=True
    )

    pipeline = Pipeline([
        ('pca', PCA(n_components=pca)), # Reduce dimensionality while keeping pca% of variance
        ('clf', grid_search) # Learns of the processed histogram
    ])

    pipeline.fit(train_X_y[0], train_X_y[1])
    return grid_search

def broad_search(algo, train_X_y, n_iter=50, folds=1, pca=0.95):

    params = clean_hyperparameters(algo["hyper"])
    random_search = None

    for i in range(max(folds, 1)):  # Ensures at least one iteration
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

        pipeline = Pipeline([
            ('pca', PCA(n_components=pca)), # Reduce dimensionality while keeping pca% of variance
            ('clf', random_search) # Learns of the processed histogram
        ])

        pipeline.fit(train_X_y[0], train_X_y[1])

        params = getCloseParams(random_search, algo["hyper"], amount=100, fill_with_random=True)

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
                if (fill_with_random and len(all_values) > 1):
                    # Randomly sample remaining values
                    others = [v for v in all_values if v != best_val]
                    remaining_amount = amount - 1
                    if len(others) > remaining_amount:
                        selected.extend(random.sample(others, remaining_amount))
                    else:
                        selected.extend(others)  # Take all if not enough others
                close_params[param] = selected
            else:
                # In case best_val is not found
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

def test (train_sample, test_sample, algo):
    clf, X, y = model.fitFromHisto(train_sample, algo=algo)
    _, _, test_err, _ = model.predictFromHisto(test_sample, clf)
    train_err = model.get_train_error(clf, X, y)

    return (test_err, train_err)

def mass_test_from_json(train_X_y, json_path, model_name=None):
    with open(json_path, "r") as f:
        algos = json.load(f)

    if model_name is not None:
        algos = [algo for algo in algos if algo["algo"] == model_name]

    mass_test(train_X_y, algos)