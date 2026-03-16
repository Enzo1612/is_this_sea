from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Importation of the models
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

    # Model is valid from areHyperValid
    model_class = models[algo["algo"]]
    classifieur = model_class(**algo["hyper"])

    # Pipeline is needed for the HOG feature
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Standardise the histogram
        ('pca', PCA(n_components=0.95)), # Reduce dimensionality while keeping 95% of variance
        ('clf', classifieur) # Learns of the processed histogram
    ])

    pipeline.fit(X, y) # fit makes the data (X) go through the pipeline
    return (pipeline, X, y)

def predictFromHisto(sample, clf, use_hog=False, use_histo=True):
    X, y = _build_X_y(sample, use_hog=use_hog, use_histo=use_histo)
    y_pred = clf.predict(X)
    for i in range(len(sample)):
        sample[i]["y_predicted_class"] = y_pred[i]
    return (X, y, 1 - accuracy_score(y, y_pred))

def empirical_error(y_true, y_pred):
    err = 1 - accuracy_score(y_true, y_pred)
    return err

def cross_val_on_sample(X, y, clf, cv=5):
    return 1 - cross_val_score(clf, X, y, cv=cv)

def get_train_error(clf, X_train, y_train):
    return 1 - accuracy_score(y_train, clf.predict(X_train))

def split_samples(samples, test_size=0.2, random_state=42):
    y = [s["y_true_class"] for s in samples]
    return train_test_split(samples, test_size=test_size, random_state=random_state, stratify=y)

def areHyperValid(algo):
    # Ensures the model name and all the hyperparameters are valid 

    if (algo["algo"] not in models):
        raise ValueError("Invalid model name")
    
    model_class = models[algo["algo"]]
    valid_parameters = model_class().get_params().keys()

    return all(param in valid_parameters for param in algo["hyper"].keys())
