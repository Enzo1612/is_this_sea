from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

import numpy as np

models = {
    "GaussianNB": GaussianNB,
    "SVC": SVC,
    "KNN": KNeighborsClassifier,
    "DecisionTree": DecisionTreeClassifier,
    "RandomForest": RandomForestClassifier,
    "Bagging": BaggingClassifier

}

def _build_X_y(sample, use_hog=True, use_histo=True):
    X = []
    y = []
    for s in sample:
        features = []
        if use_histo:
            features.append(np.array(s["X_histo"]).ravel())
        if use_hog:
            features.append(np.array(s["X_hog"]).ravel())
        X.append(np.concatenate(features))
        y.append(s["y_true_class"])
    return np.array(X), np.array(y)

def fitFromHisto(sample, algo={"algo": "GaussianNB", "hyper": {}}, use_hog=False, use_histo=True):
    if (not areHyperValid(algo)) :
        raise ValueError("Invalid hyperparameters")

    X, y = _build_X_y(sample, use_hog=use_hog, use_histo=use_histo)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # model is valid from areHyperValid
    model_class = models[algo["algo"]]
    classifieur = model_class(**algo["hyper"])

    # if (algo["algo"] == "GaussianNB"):
    #     # No hyperparameters for GaussianNB
    #     classifieur = GaussianNB(**algo["hyper"]) # **algo["hyper"] unpacks the hyperparameters dictionary
    # elif (algo["algo"] == "SVC"):
    #     classifieur = SVC(**algo["hyper"])
    # else:
    #     raise TypeError("Model isn't valid")
    classifieur.fit(X_train, y_train)
    return (classifieur, X_test, y_test)

def predictFromHisto(sample, model, use_hog=False, use_histo=True):
    X, _ = _build_X_y(sample, use_hog=use_hog, use_histo=use_histo)
    predicted = model.predict(X)
    for i in range(len(sample)):
        sample[i]["y_predicted_class"] = predicted[i]
    return sample

def compute_empirical_error(sample, model, use_hog=False, use_histo=True): 
    X, y_true = _build_X_y(sample, use_hog=use_hog, use_histo=use_histo)
    y_pred = model.predict(X)
    err = 1 - accuracy_score(y_true, y_pred)
    print(f"Empirical error : {err}")

def cross_val(model, X, y):
    print(cross_val_score(model, X, y, cv=3))


def train_test_eval(sample, algo, use_hog=False, use_histo=True, test_size=0.2, random_state=42):
    X, y = _build_X_y(sample, use_hog=use_hog, use_histo=use_histo)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    model_class = models[algo["algo"]]
    clf = model_class(**algo["hyper"])
    clf.fit(X_train, y_train)

    y_pred_test = clf.predict(X_test)
    test_err = 1 - accuracy_score(y_test, y_pred_test)

    y_pred_train = clf.predict(X_train)
    train_err = 1 - accuracy_score(y_train, y_pred_train)

    return clf, test_err, train_err

def cross_val_on_sample(sample, algo, use_hog=False, use_histo=True, cv=5):
    X, y = _build_X_y(sample, use_hog=use_hog, use_histo=use_histo)
    model_class = models[algo["algo"]]
    clf = model_class(**algo["hyper"])
    return 1 - cross_val_score(clf, X, y, cv=cv)


def areHyperValid(algo):
    if (algo["algo"] not in models):
        raise ValueError("Invalid model name")
    
    model_class = models[algo["algo"]]
    valid_parameters = model_class().get_params().keys()

    return all(param in valid_parameters for param in algo["hyper"].keys())
