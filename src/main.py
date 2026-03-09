import model
from sample import Sample

import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

use_hog = True
use_histo = False


def hyperparametres_to_string(algo):
    algo_name = algo['algo']
    hypers = algo['hyper']

    hyper_to_string = ", ".join(f"{key}={value}" for key, value in hypers.items())
    return f"{algo_name}({hyper_to_string})"


def predictFile(sample, EE, ER, algo):
    with open("equipe.txt", "w") as f:
        f.write(f"# E.Nicaise, T.Rakesh, G.Elouard-Bucchini, A.Tomasi (Equipe KING_BE4RN2000)\n")
        f.write(f"# {algo['algo']}\n")
        f.write(f"# {hyperparametres_to_string(algo)}\n")
        f.write(f"# {algo['descipteurs']}\n")
        for s in sample:
            #img_name = os.path.basename(s["name_path"]) if "name_path" in s else s.get("name", "unknown")
            img_name = s["name_path"].split(os.sep)[-1] 
            pred = s["y_predicted_class"]
            f.write(f"{img_name} {pred:+d}\n")
        if model.compute_empirical_error is not None:
            f.write(f"\n# EE = {EE:.2f}\n")
        if model.cross_val is not None:
            f.write(f"# ER = {ER:.2f}\n")


def main():
    s = Sample()
    sample = s.buildSampleFromPath()

    algo = {
        "algo": "SVC",
        "hyper": {
            "kernel": "rbf",
            "C": 10.0,
            "gamma": "scale",
            "class_weight": "balanced",
            "probability": False
        },
        "HP_str": "TODO",
        "descipteurs": "TODO"
    }   

    #TODO: faire les modèles : KNN, arbre de décision, random forest (dans json)
    
    classifieur, X_test, y_test = model.fitFromHisto(sample, algo=algo, use_hog=use_hog, use_histo=use_histo)
    sample = model.predictFromHisto(sample, classifieur, use_hog=use_hog, use_histo=use_histo)
    # model.cross_val(classifieur, X_test, y_test)

    
    
    clf, test_err, train_err = model.train_test_eval(sample, algo=algo, use_hog=use_hog, use_histo=use_histo)
    cv = model.cross_val_on_sample(sample, algo=algo, use_hog=use_hog, use_histo=use_histo, cv=5)
    print(cv)
    print(f"Test error: {test_err:.2f}, Train error: {train_err:.2f}")

    predictFile(sample, test_err, cv.mean(), algo)

if __name__ == "__main__":
    main()