import json

import numpy as np

def mass_test(sample, algos):
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

def test (sample, algo):
    ...

def mass_test_from_json(sample, json_path):
    with open(json_path, "r") as f:
        algos = json.load(f)
    mass_test(sample, algos)