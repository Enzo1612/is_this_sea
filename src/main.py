
import joblib
import numpy as np

import model
from sample import Sample

from sklearn.metrics import confusion_matrix

import os

from mass_test_algorithms import mass_test_from_json

use_hog = False
use_histo_hsv = True
use_histo_rgb = True
use_hsv_lbp = True

def predictFile(sample, EE, ER, filename="KING_BE4RN2000.txt"):
    # the method assumes the algorithm used is the one in model.pkl!
    with open(filename, "w") as f:
        f.write(f"# E.Nicaise, T.Rakesh, G.Elouard-Bucchini, A.Tomasi (Equipe KING_BE4RN2000)\n")
        f.write(f"# SVC\n")
        f.write(f"# C: 90, kernel: sigmoid, gamma: 0.31, degree: 2, class_weight: balanced\n")
        f.write(f"# histo. RGB (x2 sur 33% bas), histo. hue/sat, histo. lbp\n")
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

algo_test = [   # C: 2 / kernel: linear / gamma: auto
    {"algo": "SVC", "hyper": {'C': 90, 'class_weight': 'balanced', 'degree': 2, 'gamma': 0.31, 'kernel': 'sigmoid'}},
    {"algo": "SVC", "hyper": {'C': 78, 'class_weight': 'balanced', 'degree': 2, 'gamma': 0.347, 'kernel': 'sigmoid'}},
    {"algo": "SVC", "hyper": {'C': 90, 'class_weight': 'balanced', 'degree': 2, 'gamma': 0.157, 'kernel': 'rbf'}},
    {"algo": "SVC", "hyper": {'C': 98, 'class_weight': 'balanced', 'degree': 2, 'gamma': 'scale', 'kernel': 'linear'}},
    # {"algo": "KNN", "hyper": {'n_neighbors': 5, 'weights': 'distance', 'metric': 'minkowski', 'p': 2, 'algorithm': 'auto'}},
]

def main():
    s = Sample()
    # Build samples using default paths
    train_sample, test_sample = s.buildSampleFromPath()
    print(f"Train sample size: {len(train_sample)}, Test sample size: {len(test_sample)}")  
    
    for algo in algo_test:
        print(f"Testing {algo['algo']}")
        classifieur, X_train, y_train = model.fitFromHisto(train_sample, algo=algo, use_hog=use_hog, use_histo_hsv=use_histo_hsv, use_hsv_lbp=use_hsv_lbp, use_histo_rgb=use_histo_rgb)
        print("fit done")
        X_train, y_train, train_err, y_pred_train = model.predictFromHisto(train_sample, classifieur, use_hog=use_hog, use_histo_hsv=use_histo_hsv, use_hsv_lbp=use_hsv_lbp, use_histo_rgb=use_histo_rgb)
        print("train predict done")
        X_test, y_test, test_err, y_pred_test = model.predictFromHisto(test_sample, classifieur, use_hog=use_hog, use_histo_hsv=use_histo_hsv, use_hsv_lbp=use_hsv_lbp, use_histo_rgb=use_histo_rgb)
        print("test predict done")

        cm = confusion_matrix(y_test, y_pred_test)
        print("Confusion Matrix: ")
        print(cm)

        cv = model.cross_val_on_sample(X_test, y_test, classifieur, cv=5)
        print(cv, cv.mean(), sep="\t")
        print(f"Test error: {test_err:.2f}, Train error: {train_err:.2f}")
        print("\n", "-" * 50, "\n", sep="")

        # Saves the trained model using joblib
        joblib.dump(classifieur, "model.pkl")

def main_full_train(test_folder_path="data/raw/Init/Data"):
    s = Sample()
    # Build samples using default paths
    train_sample = s.buildSingleSampleFromPath()
    test_sample = s.buildSingleSampleFromPath(path1=test_folder_path+"/Mer", path2=test_folder_path+"/Ailleurs", augment=False)
    print(f"Train sample size: {len(train_sample)}, Test sample size: {len(test_sample)}")

    # meilleur :
    algo = {"algo": "SVC", "hyper": {'C': 90, 'class_weight': 'balanced', 'degree': 2, 'gamma': 0.31, 'kernel': 'sigmoid'}}
    
    # algo = {"algo": "SVC", "hyper": {'C': 90, 'class_weight': 'balanced', 'degree': 2, 'gamma': 0.157, 'kernel': 'rbf'}}
    # algo = {"algo": "SVC", "hyper": {'C': 10.7, 'class_weight': 'balanced', 'degree': 1, 'gamma': 3.6, 'kernel': 'sigmoid'}}

    print(f"Testing {algo['algo']}")
    classifieur, X_train, y_train = model.fitFromHisto(train_sample, algo=algo, use_hog=use_hog, use_histo_hsv=use_histo_hsv, use_hsv_lbp=use_hsv_lbp, use_histo_rgb=use_histo_rgb)
    print("fit done")
    X_train, y_train, train_err, y_pred_train = model.predictFromHisto(train_sample, classifieur, use_hog=use_hog, use_histo_hsv=use_histo_hsv, use_hsv_lbp=use_hsv_lbp, use_histo_rgb=use_histo_rgb)
    print("train predict done")
    X_test, y_test, test_err, y_pred_test = model.predictFromHisto(test_sample, classifieur, use_hog=use_hog, use_histo_hsv=use_histo_hsv, use_hsv_lbp=use_hsv_lbp, use_histo_rgb=use_histo_rgb)
    print("test predict done")

    cm = confusion_matrix(y_test, y_pred_test)
    print("Confusion Matrix: ")
    print(cm)

    # cv = model.cross_val_on_sample(X_test, y_test, classifieur, cv=5)
    # print(cv, cv.mean(), sep="\t")

    print(f"Test error: {test_err:.2f}, Train error: {train_err:.2f}")
    print("\n", "-" * 50, "\n", sep="")

    # Saves the trained model using joblib
    joblib.dump(classifieur, "model.pkl")

def mass_test_worker(model_name=None, json_file="models.json"):
    s = Sample()
    train_sample, test_sample = s.buildSampleFromPath(apply_hue_shift=True, apply_flip=True, apply_rotation=True, augment_train=False)
    print(f"Train sample size: {len(train_sample)}, Test sample size: {len(test_sample)}") 

    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(current_dir, json_file)

    train_X_y = model.build_X_y(train_sample, use_hog=use_hog, use_histo_hsv=use_histo_hsv, use_hsv_lbp=use_hsv_lbp, use_histo_rgb=use_histo_rgb)
    test_X_y = model.build_X_y(test_sample, use_hog=use_hog, use_histo_hsv=use_histo_hsv, use_hsv_lbp=use_hsv_lbp, use_histo_rgb=use_histo_rgb)

    mass_test_from_json(train_X_y, json_path, model_name=model_name)

def predict_from_data_file(folder_name="data/raw/Init/Data", filename="KING_BE4RN2000.txt"):
    s = Sample()
    sample = []
    sample.extend(s.make_path(os.path.join(folder_name, "Mer"), 0))
    sample.extend(s.make_path(os.path.join(folder_name, "Ailleurs"), 0))

    clf = joblib.load("model.pkl")

    model.predictFromHisto(sample, clf, use_hog=use_hog, use_histo_hsv=use_histo_hsv, use_hsv_lbp=use_hsv_lbp, use_histo_rgb=use_histo_rgb)

    predictFile(sample, None, None, filename=filename)

if __name__ == "__main__":
    # main()
    mass_test_worker(json_file="models_bis.json")

    # main_full_train()
    # predict_from_data_file()

    # clf = joblib.load("model.pkl")
    # s = Sample()
    # sample = s.buildSingleSampleFromPath(path1="data/raw/Init/Test/Mer", path2="data/raw/Init/Test/Ailleurs", augment=False)
    # X_test, y_test, test_err, y_pred_test = model.predictFromHisto(sample, clf, use_hog=use_hog, use_histo_hsv=use_histo_hsv, use_hsv_lbp=use_hsv_lbp, use_histo_rgb=use_histo_rgb)

    # cm = confusion_matrix(y_test, y_pred_test)
    # print("Confusion Matrix: ")
    # print(cm)
    # print(f"Test error: {test_err:.2f}")