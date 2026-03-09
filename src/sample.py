from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.feature import hog

import model


class Sample:

    def make_path(self, path, label):
        samples = []
        for image_array in os.listdir(path):
            img_path = os.path.join(path, image_array)
            image = Image.open(img_path)
            resized = self.resizeImage(image, 200, 200)
            samples.append({
                "name_path": img_path,
                "resized_image": resized,
                "X_histo": self.computeHisto(image),
                "X_hog": self.computeHog(resized),
                "y_true_class": label,
                "y_predicted_class": None
            })
        return samples

    def buildSampleFromPath(self, path1 = "data/raw/Init/Mer", path2 = "data/raw/Init/Ailleurs"):
        samples = []
        samples.extend(self.make_path(path1, 1))
        samples.extend(self.make_path(path2, -1))
        return samples

    def resizeImage(self, PImage, h, l):
        return PImage.resize((h, l))
    
    def computeHisto(self, PImage): 
        if PImage.mode != 'RGB':
            PImage = PImage.convert('RGB')
        r, g, b = PImage.split() # pyright: ignore[reportAttributeAccessIssue]
        return [
            r.histogram(), 
            g.histogram(), 
            b.histogram()
        ]
    
    def computeHog(self, PImage):
        if PImage.mode != "RGB":
            PImage = PImage.convert("RGB")
        gray = np.array(PImage.convert("L"))
        return hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True
        )
    
    

if __name__ == "__main__":
    s = Sample()
    sample = s.buildSampleFromPath()

    classifieur, X_test, y_test = model.fitFromHisto(sample)
    sample = model.predictFromHisto(sample, classifieur)
    model.cross_val(classifieur, X_test, y_test)
    model.compute_empirical_error(sample, classifieur)

