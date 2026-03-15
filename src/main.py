
import joblib

import model
from sample import Sample

import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from mass_test_algorithms import mass_test_from_json

use_hog = False
use_histo = True


def hyperparametres_to_string(algo):
    algo_name = algo['algo']
    hypers = algo['hyper']

    hyper_to_string = ", ".join(f"{key}={value}" for key, value in hypers.items())
    return f"{algo_name}({hyper_to_string})"


def predictFile(sample, EE, ER):
    with open("equipe.txt", "w") as f:
        f.write(f"# E.Nicaise, T.Rakesh, G.Elouard-Bucchini, A.Tomasi (Equipe KING_BE4RN2000)\n")
        f.write(f"# SVC\n")
        f.write(f"# C: 2, kernel: linear, gamma: auto, degree: 3, class_weight: balanced\n")
        f.write(f"# histogramme des couleurs (poids de 2 sur 33% du bas de l'image)\n")
        for s in sample:
            #img_name = os.path.basename(s["name_path"]) if "name_path" in s else s.get("name", "unknown")
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

        # model.cross_val(classifieur, X_test, y_test)
        
        # _, test_err = model.empirical_error_on_sample(X_test, y_test, classifieur)
        cv = model.cross_val_on_sample(X_test, y_test, classifieur, cv=5)
        print(cv, cv.mean(), sep="\t")
        print(f"Test error: {test_err:.2f}, Train error: {train_err:.2f}")
        print("\n", "-" * 50, "\n", sep="")

        joblib.dump(classifieur, "model.pkl") 

    # for algo in algos:
    #     print(f"Testing {algo['algo']}")
        
    #     # 1. Entraîner le modèle
    #     classifieur, X_train, y_train = model.fitFromHisto(train_sample, algo=algo, use_hog=use_hog, use_histo=use_histo)

    #     # 2. Obtenir les prédictions pour les deux jeux de données
    #     y_train_pred = classifieur.predict(X_train)
        
    #     X_test, y_test = model._build_X_y(test_sample, use_hog=use_hog, use_histo=use_histo)
    #     y_test_pred = classifieur.predict(X_test)

    #     # 3. Calculer les erreurs
    #     train_err = model.empirical_error(y_train, y_train_pred)
    #     test_err = model.empirical_error(y_test, y_test_pred)
        
    #     # 4. Calculer la cross-validation sur le jeu d'entraînement (jamais sur le test !)
    #     cv = model.cross_val_on_sample(X_train, y_train, classifieur, cv=5)
        
    #     print(f"CV Error Mean: {cv.mean():.4f}")
    #     print(f"Test error: {test_err:.4f}, Train error: {train_err:.4f}")
    #     print("\n", "-" * 50, "\n", sep="")

    # predictFile(sample, test_err, cv.mean(), algo)

# def main():
#     s = Sample()
#     train_sample, val_sample, test_sample = s.buildSampleFromPath()

#     # nouveau: validation interne pour choisir l'algo
#     train_inner, val_sample = model.split_samples(train_sample, test_size=0.2, random_state=42)

#     best_algo = None
#     best_val_err = float("inf")

#     for algo in algos:
#         print(f"Testing {algo['algo']}")

#         classifieur, X_train, y_train = model.fitFromHisto(
#             train_inner, algo=algo, use_hog=use_hog, use_histo=use_histo
#         )

#         X_val, y_val = model.predictFromHisto(
#             val_sample, classifieur, use_hog=use_hog, use_histo=use_histo
#         )

#         train_err = model.get_train_error(classifieur, X_train, y_train)
#         _, val_err = model.empirical_error_on_sample(X_val, y_val, classifieur)
#         cv = model.cross_val_on_sample(X_train, y_train, classifieur, cv=5)

#         print(f"CV mean: {cv.mean():.4f} | Val error: {val_err:.4f} | Train error: {train_err:.4f}")

#         if val_err < best_val_err:
#             best_val_err = val_err
#             best_algo = algo

#     # test final UNE SEULE FOIS avec le meilleur algo
#     print(f"\nBest algo: {best_algo['algo']} (val_err={best_val_err:.4f})")
#     best_clf, _, _ = model.fitFromHisto(train_sample, algo=best_algo, use_hog=use_hog, use_histo=use_histo)
#     X_test, y_test = model.predictFromHisto(test_sample, best_clf, use_hog=use_hog, use_histo=use_histo)
#     _, test_err = model.empirical_error_on_sample(X_test, y_test, best_clf)
#     print(f"Final test error: {test_err:.4f}")

if __name__ == "__main__":
    # main()

    folder_name = "data/raw/Init/Test"
    s = Sample()
    train_sample, test_sample = s.buildSampleFromPath(path1=f"{folder_name}/Mer", path2=f"{folder_name}/Ailleurs", apply_rotation=False, apply_flip=False, apply_brightness_modification=False)
    sample = train_sample + test_sample
    clf = joblib.load("model.pkl")

    X_test, y_test, test_err = model.predictFromHisto(sample, clf, use_hog=use_hog, use_histo=use_histo)
    print("test predict done")

    # model.cross_val(classifieur, X_test, y_test)
    
    # _, test_err = model.empirical_error_on_sample(X_test, y_test, classifieur)
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