import numpy as np
from sklearn.datasets import fetch_openml

class DataLoader:
    def load_mnist(self) -> (list[str], list[str]):
        """Returns X, y for MNIST"""
        mnist = fetch_openml('mnist_784', parser="auto")
        return mnist.data, mnist.target
    
    def load_sea_data(self, path : str):
        pass

    def get_binary_target(self, y : list[str], target_class : str) -> list[bool]:
        """Converts [0, 5, 3, 5] -> [False, True, False, True]"""
        return (y == target_class)
    