from PIL import Image, ImageEnhance
import numpy as np
import os
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
                "name_path": img_path,  # Path to the image (not just the name of image)
                "resized_image": resized,
                "X_histo": self.computeWeightedHisto(image),
                "X_hog": self.computeHog(resized),
                "y_true_class": label,
                "y_predicted_class": None
            })
        return samples

    def buildSampleFromPath(self, path1 = "data/raw/Init/Mer", path2 = "data/raw/Init/Ailleurs", 
                            apply_rotation=True, apply_flip=True, apply_brightness_modification=True):
        samples = []
        samples.extend(self.make_path(path1, 1))
        samples.extend(self.make_path(path2, -1))

        train_sample, test_sample = model.split_samples(samples, test_size=0.3, random_state=42)

        # Augment the training sample to make the model more robust
        train_sample = self.apply_modifications_to_samples(train_sample, apply_rotation=apply_rotation, apply_flip=apply_flip, apply_brightness_modification=apply_brightness_modification)
        # The test sample is not augmented to ensure a realistic evaluation

        return (train_sample, test_sample)

    def resizeImage(self, PImage, h, l):
        return PImage.resize((h, l))
    
    def computeHisto(self, PImage): 
        # Ensures the image is in RGB mode
        if PImage.mode != 'RGB':
            PImage = PImage.convert('RGB')
        r, g, b = PImage.split()
        return [
            r.histogram(), 
            g.histogram(), 
            b.histogram()
        ]

    def computeWeightedHisto(self, PImage, bottom_weight=2.0, bottom_ratio=0.33):
        # Ensures the image is in RGB mode
        if PImage.mode != 'RGB':
            PImage = PImage.convert('RGB')

        width, height = PImage.size

        # Initialize histograms to zero
        r_hist = np.zeros(256, dtype=float)
        g_hist = np.zeros(256, dtype=float)
        b_hist = np.zeros(256, dtype=float)

        # Process the top part of the image
        top_height = height - int(height * bottom_ratio)
        if top_height > 0:
            # `top_box` is a rectangle defined by the coordinates: 
            # (0, 0) for the top-left corner and (width, top_height) for the bottom-right corner
            top_box = (0, 0, width, top_height)
            # Crop to only get the part represented by the `top_box` rectangle
            top = PImage.crop(top_box)
            if top.size[0] > 0 and top.size[1] > 0:
                r_top, g_top, b_top = top.split()
                r_hist += np.array(r_top.histogram(), dtype=float)
                g_hist += np.array(g_top.histogram(), dtype=float)
                b_hist += np.array(b_top.histogram(), dtype=float)

        # Process the bottom part of the image
        bottom_start = height - int(height * bottom_ratio)
        if bottom_start < height:
            # `bottom_box` is a rectangle defined by the coordinates:
            # (0, bottom_start) for the top-left corner and (width, height) for the bottom-right corner
            bottom_box = (0, bottom_start, width, height)
            # Crop to only get the part represented by the `bottom_box` rectangle
            bottom = PImage.crop(bottom_box)
            if bottom.size[0] > 0 and bottom.size[1] > 0:
                r_bottom, g_bottom, b_bottom = bottom.split()
                r_hist += np.array(r_bottom.histogram(), dtype=float)
                g_hist += np.array(g_bottom.histogram(), dtype=float)
                # Only weight the bottom blue histogram
                b_hist += (bottom_weight * np.array(b_bottom.histogram(), dtype=float))

        return [
            r_hist.tolist(),
            g_hist.tolist(),
            b_hist.tolist()
        ]
    
    def computeHog(self, PImage):
        # Converts the image to grayscale
        gray = np.array(PImage.convert("L"))
        return hog(
            gray,                       # Input image
            orientations=9,             # Number of orientation categories
            pixels_per_cell=(8, 8),     # Size of the cell in pixels
            cells_per_block=(2, 2),     # Number of cells in each block
            # blocks are overlapping regions of the image
            block_norm="L2-Hys",        # Defines the normalization method for the blocks
            # L2: normalizes using the euclidean distance
            # Hys (Hysteresis thresholding): applies a threshold to the normalized values to reduce the influence of outliers
            feature_vector=True         # Returns the HOG features as a 1D array (flattened) rather than a multi-dimensional array
        )

    def apply_modifications_to_samples(self, samples, apply_rotation=True, apply_flip=True, apply_brightness=True):
        modified_samples = []
        for sample in samples:
            modified_images = self.apply_modifications(sample["resized_image"], apply_rotation=apply_rotation, apply_flip=apply_flip, apply_brightness=apply_brightness)
            for img in modified_images:
                modified_samples.append({
                    "name_path": sample["name_path"],
                    "resized_image": img,
                    "X_histo": self.computeWeightedHisto(img),
                    "X_hog": self.computeHog(img),
                    "y_true_class": sample["y_true_class"],
                    "y_predicted_class": None
                })
        return modified_samples

    def apply_modifications(self, PImage, apply_rotation=True, apply_flip=True, apply_brightness=True):
        # Ensures the image is in RGB mode
        if (PImage.mode != "RGB"):
            PImage = PImage.convert("RGB")

        base_images = []
        # Keep the original image
        base_images.append(PImage)

        # Add rotations of the original image
        if apply_rotation:
            base_images.extend([PImage.rotate(angle) for angle in [90, 180, 270]])

        # Flip all of the above images (original + rotations)
        if apply_flip:
            flipped_images = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in base_images]
        else:
            flipped_images = []

        # Combine base images and their flips
        all_geometric_images = base_images + flipped_images

        # Apply brightness modifications to all images (original, rotations, and flips)
        final_images = []
        for img in all_geometric_images:
            final_images.append(img) # Keep the image with original brightness
            # Add versions with modified brightness
            if apply_brightness:
                final_images.append(ImageEnhance.Brightness(img).enhance(0.7))
                final_images.append(ImageEnhance.Brightness(img).enhance(1.3))

        return final_images
    

if __name__ == "__main__":
    s = Sample()
    sample = s.buildSampleFromPath()

    classifieur, X_test, y_test = model.fitFromHisto(sample)
    sample = model.predictFromHisto(sample, classifieur)
    model.cross_val(classifieur, X_test, y_test)
    model.compute_empirical_error(sample, classifieur)

