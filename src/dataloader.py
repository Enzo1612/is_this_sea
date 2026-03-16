import numpy as np
from sklearn.datasets import fetch_openml

class MNIST_DataLoader:
    def load_mnist(self):
        """Returns X, y for MNIST"""
        mnist = fetch_openml('mnist_784', parser="auto")
        return mnist.data, mnist.target

    def get_binary_target(self, y, target_class):
        """Converts [0, 5, 3, 5] -> [False, True, False, True]"""
        return (y == target_class)



# Classe pour sea
class SeaDataLoader:
    def load_sea_data(self, path : str):
        pass

    def get_binary_target(self, y, target_class):
        """Converts [???] -> [True, False, ...]"""
        return (y == target_class) # Voir pour le format
