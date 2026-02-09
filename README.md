NICAISE Enzo
ELOUARD-BUCCHINI Gaétan
RAKESH Tarun
TOMASI Antoine
CUBAHIRO Roch Joel

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

**Visualize**:
It's important to see what type of data the model is going to be working with. Let's print the first image of the dataset.

```python
import matplotlib.pyplot as plt
def plot_digit(image_data):
    image = image_data.reshape(28, 28) # MNIST images are 28 by 28
    plt.imshow(image, cmap="binary")
    plt.axis("off")

some_digit = X[0]
plot_digit(some_digit)
plt.show()
```

### Teacher's

- Number of images:

- Size of the images:

- Format:

- Labels:

**Methods:**

```python
buildSampleFromPath(path1 : string, path2 : string) -> list[Image]
# Returns data from path 1 (labeled 1 = sea) and from path 2 (labeled -1 = not sea).

resizeImage(i : int, h : int, l : int) -> Image
# Returns the resized image at index i of size h\l (height, length)

computeHisto(int i) -> list
# Stores and print the histogram of the image at index i using PIL
```

**Class:**

```python
class Image {
    @property
    __self__(string : path, resized_image : Image, X_histo : list, y_true_class : int, y_predicted_class : int)
}
```

## Models

## Experiments
