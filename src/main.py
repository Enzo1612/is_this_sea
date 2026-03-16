
import joblib

import model
from sample import Sample

import os

from mass_test_algorithms import mass_test_from_json

use_hog = False
use_histo = True

def predictFile(sample, EE, ER):
    # the method assumes the algorithm used is the one in model.pkl!
    with open("equipe.txt", "w") as f:
        f.write(f"# E.Nicaise, T.Rakesh, G.Elouard-Bucchini, A.Tomasi (Equipe KING_BE4RN2000)\n")
        f.write(f"# SVC\n")
        f.write(f"# C: 2, kernel: linear, gamma: auto, degree: 3, class_weight: balanced\n")
        f.write(f"# histogramme des couleurs (poids de 2 sur 33% du bas de l'image)\n")
        for s in sample:
            # Extract the image name from the path
            img_name = s["name_path"].split(os.sep)[-1] 
            pred = s["y_predicted_class"]
            f.write(f"{img_name} {pred:+d}\n")
        if EE is not None:
            f.write(f"\n# EE = {EE:.2f}\n")
        if ER is not None:
            f.write(f"# ER = {ER:.2f}\n")


algos = [

    {
        "algo": "SVC",
        "hyper": {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "class_weight": "balanced",
            "probability": False
        },
        "HP_str": "TODO",
        "descipteurs": "TODO"
    },

    {
        "algo": "KNN",
        "hyper": {
            "n_neighbors": 5,
            "weights": "distance",
            "metric": "minkowski",
            "p": 2,
            "algorithm": "auto"
        },
        "HP_str": "TODO",
        "descipteurs": "TODO"
    },

    {
        "algo": "DecisionTree",
        "hyper": {
            "criterion": "gini",
            "max_depth": 5,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features": "sqrt"
        },
        "HP_str": "TODO",
        "descipteurs": "TODO"
    },

    {
        "algo": "RandomForest",
        "hyper": {
            "n_estimators": 200,
            "max_depth": 5,
            "min_samples_split": 4,
            "min_samples_leaf": 2,
            "max_features": "sqrt"
        },
        "HP_str": "TODO",
        "descipteurs": "TODO"
    },

    {
        "algo": "Bagging",
        "hyper": {
            "n_estimators": 100,
            "max_samples": 0.5,
            "max_features": 0.8,
            "bootstrap": True,
            "bootstrap_features": False
        },
        "HP_str": "TODO",
        "descipteurs": "TODO"
    }

]

algo_test = [
    {"algo": "SVC", "hyper": {'C': 2, 'kernel': 'linear', 'gamma': 'auto', 'degree': 3, 'class_weight': 'balanced'}},
    
]

def main():
    s = Sample()
    # Build samples using default paths
    train_sample, test_sample = s.buildSampleFromPath()
    print(f"Train sample size: {len(train_sample)}, Test sample size: {len(test_sample)}")  
    
    for algo in algo_test:
        print(f"Testing {algo['algo']}")
        classifieur, X_train, y_train = model.fitFromHisto(train_sample, algo=algo, use_hog=use_hog, use_histo=use_histo)
        print("fit done")
        X_train, y_train, train_err = model.predictFromHisto(train_sample, classifieur, use_hog=use_hog, use_histo=use_histo)
        print("train predict done")
        X_test, y_test, test_err = model.predictFromHisto(test_sample, classifieur, use_hog=use_hog, use_histo=use_histo)
        print("test predict done")

        cv = model.cross_val_on_sample(X_test, y_test, classifieur, cv=5)
        print(cv, cv.mean(), sep="\t")
        print(f"Test error: {test_err:.2f}, Train error: {train_err:.2f}")
        print("\n", "-" * 50, "\n", sep="")

        # Saves the trained model using joblib
        joblib.dump(classifieur, "model.pkl")

if __name__ == "__main__":
    # main()

    folder_name = "data/raw/Init/Test"
    s = Sample()
    train_sample, test_sample = s.buildSampleFromPath(path1=f"{folder_name}/Mer", path2=f"{folder_name}/Ailleurs", apply_rotation=False, apply_flip=False, apply_brightness_modification=False)
    sample = train_sample + test_sample
    clf = joblib.load("model.pkl")

    X_test, y_test, test_err = model.predictFromHisto(sample, clf, use_hog=use_hog, use_histo=use_histo)
    print("test predict done")

    cv = model.cross_val_on_sample(X_test, y_test, clf, cv=5)
    print(cv, cv.mean(), sep="\t")
    print(f"Test error: {test_err:.2f}")

    predictFile(sample, test_err, cv.mean())



    # s = Sample()
    # train_sample, test_sample = s.buildSampleFromPath(apply_brightness_modification=True, apply_flip=True, apply_rotation=True)
    # print(f"Train sample size: {len(train_sample)}, Test sample size: {len(test_sample)}") 

    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # json_path = os.path.join(current_dir, "models.json")

    # mass_test_from_json(train_sample, test_sample, json_path)