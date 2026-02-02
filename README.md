# is_this_sea

Machine Learning experiment to get the best result to the questions "is this sea"?

## Data

Which datasets are we going to use. How are we going to clean them. How are we going to engineer features.

### MNIST

The simplest way to get the first results is to use the MNIST dataset. Since it's a binary classification task, we're going to classify `5` and `not 5`.

**import**:

```python
# This is a helper function to download popular datasets
from sklearn.datasets import fetch_openml

# mnist is one of the first image datasets. It contains digits from 0 to 9. It was created by Yann Lecun to recognize postal codes in the 90s.
mnist = fetch_openml('mnist_784', as_frame=False)

# By convention, X is the data to train on, y is the labels.
X, y = mnist.data, mnist.target
```

The returned dataset is already split into a training set (first 60k images) and a test set (last 10k images). Data should always be split before being inspected.

```python
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
```

## Models

## Experiments
