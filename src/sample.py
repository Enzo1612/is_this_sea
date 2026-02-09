from PIL import Image
import numpy as np
import os

class Sample:

    image_sample = None

    def __self__(self, image_array):
        self.image_array = image_array
        # constructor

    def make_path(self, path, label):
        samples = []
        for image_array in os.listdir(path):
            img_path = os.path.join(path, image_array)
            image = Image.open(img_path)
            samples.append({
                "name_path": path,
                "resized_image": self.resizeImage(image, 2, 2),
                "X_histo": self.computeHisto(image),
                "y_true_class": label,
                "y_predicted_class": None
            })
        return samples

    def buildSampleFromPath(self, path1 = "data/raw/Init/Mer",path2 = "/data/raw/Init/Ailleurs"):
        samples = []
        self.make_path(path1, 1)
        self.make_path(path2, -1)
        return samples

    def resizeImage(self, PImage, h, l):
        return PImage.resize(h, l)
    
    def computeHisto(self, PImage): 
        r, g, b = PImage.split() # pyright: ignore[reportAttributeAccessIssue]
        return [
            r.histogram(), 
            g.histogram(), 
            b.histogram()
        ]
