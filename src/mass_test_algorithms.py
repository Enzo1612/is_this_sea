import json
import itertools

import random

import numpy as np

import model

def mass_test(train_sample, test_sample, algos, n_top=10):
    top_models = []
    for algo in algos:
        print(f"Testing {algo['algo']}")
        list_hyper = []
        for HP_name, HP_info in algo["hyper"].items():
            if HP_info["type"] == "continuous":
                #value = HP_info["range"][0]
                values = np.arange(HP_info["range"][0], HP_info["range"][1] + HP_info["step"], HP_info["step"])
                list_hyper.append((HP_name, values))
            elif HP_info["type"] == "discrete":
                list_hyper.append((HP_name, HP_info["values"]))
            else:
                raise ValueError("Invalid hyperparameter type")
        list_name = []
        list_value = []

        # # --- MODIFICATION FOR TESTING ---
        # # Generate all combinations and pick a small random sample
        # all_combinations = list(itertools.product(*list_value))
        # num_to_test = min(5, len(all_combinations)) # Test up to 5, or fewer if not enough combinations exist
        # combinations_to_try = random.sample(all_combinations, num_to_test)
        # # --- END MODIFICATION ---

        for HP_name, values in list_hyper:
            list_name.append(HP_name)
            list_value.append(values)
        combinations_to_try = list(itertools.product(*list_value))
        for combination in combinations_to_try:
            # print(f"starting combination")
            # hyper = dict(zip(list_name, combination))
            hyper = {}
            for i, j in zip(list_name, combination):
                if isinstance(j, float) and j.is_integer():
                    hyper[i] = int(j)
                else:
                    hyper[i] = j

            algos_test = {"algo": algo["algo"], "hyper": hyper }
            test_err, train_err = test(train_sample, test_sample, algos_test)
            print(f"Test error: {test_err:.4f}, Train error: {train_err:.4f} for hyperparameters: {hyper}")
            if len(top_models) < n_top:
                top_models.append((test_err, algos_test))
                top_models.sort(key=lambda x: x[0]) # Keep it sorted
            elif test_err < top_models[-1][0]:
                top_models[-1] = (test_err, algos_test)
                top_models.sort(key=lambda x: x[0]) # Keep it sorted

    print("\n--- Top 10 Models ---")
    for score, tested_algo in top_models:
        print(f"Score: {score:.4f}, Algorithm: {tested_algo['algo']}, Hyperparameters: {tested_algo['hyper']}")

    return top_models


        # params = []
        
        # for HP_name, type in algo["hyper"].items():
        #     if type == "discrete":
        #         params.append(algo["hyper"][HP_name])
        #     elif type == "continuous":
        #         params.append(np.linspace(algo["hyper"][HP_name][0], algo["hyper"][HP_name][1], num=5))
        #     else:
        #         raise ValueError("Invalid hyperparameter type")
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

def mass_test_from_json(train_sample, test_sample, json_path):
    with open(json_path, "r") as f:
        algos = json.load(f)
    mass_test(train_sample, test_sample, algos)