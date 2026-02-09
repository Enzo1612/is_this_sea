from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sample import Sample


def fitFromHisto(sample, algo={"algo": "GaussianNB", "hyper": {}}):
    # For now, we don't use HyperParameters
    X = []
    y = []
    for s in sample:
        dic = {
            "X_histo": s["X_histo"]
        }
        X.append(dic)
        y.append(s["y_true_class"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    if (algo["algo"] == "GaussianNB"):
        classifieur = GaussianNB()
    else:
        raise TypeError("Model isn't valid")
    classifieur.fit(X_train, y_train)
    return (classifieur, X_test, y_test)


def predictFromHisto(sample, model):
    
    predicted = model.predict(sample.X_histo)
    for i in range(len(sample)):
        sample[i]["y_predicted_class"] = predicted[i]   # Need to verify the order stays the same
    return sample


def compute_empirical_error(Sample, model): 
    print (f"Empirical error : {1 - accuracy_score( Sample[['y_predicted_class']], predictFromHisto(Sample, model))}")

def cross_val(model, X, y):
    print(cross_val_score(model.predictFromHisto(), X, y, cv=3))