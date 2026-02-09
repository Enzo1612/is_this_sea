import numpy as np
from PIL import Image as PIM

class Image:
    name_path:str = ""
    resized_image = None    # mettre le type (Image de PIL)
    X_histo = None  # mettre le type (np.array)
    y_true_class:int = 1    # +1/-1
    y_predicted_class:int = 1    # None/+1/-1

    def __self__(self, name_path:str, y_true_class:int,
                  resized_image = None, X_histo = None,
                    y_predicted_class:int = 0) -> None:
        
        self.name_path = name_path
        self.resized_image = resized_image
        self.X_histo = X_histo
        self.y_predicted_class = y_predicted_class
        self.y_true_class = y_true_class