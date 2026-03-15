from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


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

def _build_X_y(sample, use_hog=False, use_histo=True):
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
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    # model is valid from areHyperValid
    model_class = models[algo["algo"]]
    classifieur = model_class(**algo["hyper"])

    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Important pour PCA et KNN/SVC
        ('pca', PCA(n_components=0.95)), # Conserve 95% de la variance
        ('clf', classifieur)
    ])

    pipeline.fit(X, y)
    return (pipeline, X, y)

    # classifieur.fit(X, y)
    # return (classifieur, X, y)

def predictFromHisto(sample, clf, use_hog=False, use_histo=True):
    X, y = _build_X_y(sample, use_hog=use_hog, use_histo=use_histo)
    y_pred = clf.predict(X)
    for i in range(len(sample)):
        sample[i]["y_predicted_class"] = y_pred[i]
    return (X, y, 1 - accuracy_score(y, y_pred))

# def compute_empirical_error(sample, clf, use_hog=False, use_histo=True): 
#     X, y_true = _build_X_y(sample, use_hog=use_hog, use_histo=use_histo)
#     y_pred = clf.predict(X)
#     err = 1 - accuracy_score(y_true, y_pred)
#     print(f"Empirical error : {err}")

# def cross_val(model, X, y):
#     print(cross_val_score(model, X, y, cv=3))


def empirical_error(y_true, y_pred):
    err = 1 - accuracy_score(y_true, y_pred)
    return err

def cross_val_on_sample(X, y, clf, cv=5):
    return 1 - cross_val_score(clf, X, y, cv=cv)

def get_train_error(clf, X_train, y_train):
    return 1 - accuracy_score(y_train, clf.predict(X_train))


# def split_samples(samples, test_size=0.2, random_state=42):
#     return train_test_split(samples, test_size=test_size, random_state=random_state)

def split_samples(samples, test_size=0.2, random_state=42):
    y = [s["y_true_class"] for s in samples]
    return train_test_split(samples, test_size=test_size, random_state=random_state, stratify=y)

def areHyperValid(algo):
    if (algo["algo"] not in models):
        raise ValueError("Invalid model name")
    
    model_class = models[algo["algo"]]
    valid_parameters = model_class().get_params().keys()

    return all(param in valid_parameters for param in algo["hyper"].keys())
