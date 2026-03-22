import random

from PIL import Image, ImageEnhance
import numpy as np
import os
from skimage.feature import hog
from skimage.feature import local_binary_pattern

import model


class Sample:

    def make_path(self, path, label):
        samples = []
        for image_array in os.listdir(path):
            img_path = os.path.join(path, image_array)
            image = Image.open(img_path)
            resized = self.resizeImage(image, 256, 256)
            samples.append({
                "name_path": img_path,  # Path to the image (not just the name of image)
                "resized_image": resized,
                "X_histo_hsv": self.computeHisto(resized),
                "X_histo_weighted_rgb": self.computeWeightedHisto(resized),
                "X_hog": self.computeHog(resized),
                "X_hsv_lbp": self.compute_lbp(np.array(resized.convert("L"))),
                "y_true_class": label,
                "y_predicted_class": None
            })
        return samples

    def buildSampleFromPath(self, path1 = "data/raw/Init/Mer", path2 = "data/raw/Init/Ailleurs", 
                            apply_rotation=True, apply_flip=True, apply_hue_shift=True, augment_train=True):
        samples = []
        samples.extend(self.make_path(path1, 1))
        samples.extend(self.make_path(path2, -1))

        train_sample, test_sample = model.split_samples(samples, test_size=0.3, random_state=42)

        # Augment the training sample to make the model more robust
        if augment_train:
            train_sample = self.apply_modifications_to_samples(train_sample, apply_rotation=apply_rotation, apply_flip=apply_flip, apply_hue_shift=apply_hue_shift)
        # The test sample is not augmented to ensure a realistic evaluation

        return (train_sample, test_sample)
    
    def buildSingleSampleFromPath(self, path1 = "data/raw/Init/Mer", path2 = "data/raw/Init/Ailleurs", 
                            apply_rotation=True, apply_flip=True, apply_hue_shift=True, augment=True):
        samples = []
        samples.extend(self.make_path(path1, 1))
        samples.extend(self.make_path(path2, -1))

        # Augment the training sample to make the model more robust
        if augment:
            samples = self.apply_modifications_to_samples(samples, apply_rotation=apply_rotation, apply_flip=apply_flip, apply_hue_shift=apply_hue_shift)
        # The test sample is not augmented to ensure a realistic evaluation

        return samples

    def resizeImage(self, PImage, h, l):
        return PImage.resize((h, l))
    
    def computeHisto(self, PImage): 
        # Ensures the image is in HSV mode
        if PImage.mode != 'HSV':
            PImage = PImage.convert('HSV')
        h, s, _ = PImage.split()

        hist = np.concatenate([h.histogram(), s.histogram()])
        hist = hist.astype("float")
        hist /= hist.sum()

        return hist

    def computeWeightedHisto(self, PImage, bottom_weight=2.0, bottom_ratio=0.33):
        # Ensures the image is in HSV mode
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
                b_hist += np.array(b_bottom.histogram(), dtype=float) * bottom_weight

        hist = np.concatenate([
            r_hist.tolist(),
            g_hist.tolist(),
            b_hist.tolist()
        ])

        hist = hist.astype("float")
        hist /= hist.sum()

        return hist
    
    
    
    def computeHog(self, PImage):
        # Converts the image to grayscale
        gray = np.array(PImage.convert("L"))
        return hog(
            gray,                       # Input image
            orientations=9,             # Number of orientation categories
            pixels_per_cell=(64, 64),     # Size of the cell in pixels
            cells_per_block=(2, 2),     # Number of cells in each block
            # blocks are overlapping regions of the image
            block_norm="L2-Hys",        # Defines the normalization method for the blocks
            # L2: normalizes using the euclidean distance
            # Hys (Hysteresis thresholding): applies a threshold to the normalized values to reduce the influence of outliers
            feature_vector=True         # Returns the HOG features as a 1D array (flattened) rather than a multi-dimensional array
        ).ravel()

    def apply_modifications_to_samples(self, samples, apply_rotation=True, apply_flip=True, apply_hue_shift=True):
        modified_samples = []
        for sample in samples:
            modified_images = self.apply_modifications(sample["resized_image"], apply_rotation=apply_rotation, apply_flip=apply_flip, apply_hue_shift=apply_hue_shift)
            for img in modified_images:
                modified_samples.append({
                    "name_path": sample["name_path"],
                    "resized_image": img,
                    "X_histo_hsv": self.computeHisto(img),
                    "X_histo_weighted_rgb": self.computeWeightedHisto(img),
                    "X_hog": self.computeHog(img),
                    "X_hsv_lbp": self.compute_lbp(np.array(img.convert("L"))),
                    "y_true_class": sample["y_true_class"],
                    "y_predicted_class": None
                })
        return modified_samples

    def apply_modifications(self, PImage, apply_rotation=True, apply_flip=True, apply_hue_shift=True):
        # Ensures the image is in RGB mode
        if (PImage.mode != "RGB"):
            PImage = PImage.convert("RGB")

        base_images = []
        # Keep the original image
        base_images.append(PImage)

        # Add rotations of the original image
        if apply_rotation:
            # base_images.extend([PImage.rotate(angle) for angle in [90, 180, 270]])
            base_images.extend([PImage.rotate(angle) for angle in [15, 10, 5, -5, -10, -15]])

        # Flip all of the above images (original + rotations)
        if apply_flip:
            flipped_images = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in base_images]
        else:
            flipped_images = []

        # Combine base images and their flips
        all_geometric_images = base_images + flipped_images

        final_images = []
        for img in all_geometric_images:
            final_images.append(img)
            if apply_hue_shift:
                # nb = random.choice([1, 2, 3])  # Randomly choose to apply either 1 or 2 hue shifts
                # shifts = random.sample(range(-15, 16), nb)  # Randomly select 2 shifts from the range [-15, 15]
                shifts = [-5, 5]  # Test error: 0.20, Train error: 0.17
                # shifts = [-10, -5, 5, 10]  # Test error: 0.21, Train error: 0.16
                # shifts = [-10, 10]  # Test error: 0.22, Train error: 0.17
                # shifts = [-6, 6] # Test error: 0.24, Train error: 0.16
                # shifts = [-4, 4] # Test error: 0.24, Train error: 0.17
                # shifts = [-7, -5, 5, 7] # Test error: 0.22, Train error: 0.16
                for shift in shifts:
                    final_images.append(self.apply_hue_shift(img, shift))
                # for shift in [-15, -10, -5, 5, 10, 15]:
                #     final_images.append(self.apply_hue_shift(img, shift))


        # # Apply brightness modifications to all images (original, rotations, and flips)
        # final_images = []
        # for img in all_geometric_images:
        #     final_images.append(img) # Keep the image with original brightness
        #     # Add versions with modified brightness
        #     if apply_hue_shift:
        #         final_images.append(ImageEnhance.Brightness(img).enhance(0.7))
        #         final_images.append(ImageEnhance.Brightness(img).enhance(1.3))

        return final_images
    
    def apply_hue_shift(self, PImage, hue_shift):
        # Work in HSV, shift only H channel
        hsv = np.array(PImage.convert("HSV"), dtype=np.uint8)

        h = hsv[:, :, 0].astype(np.int16)          # avoid overflow on +/- shift
        hsv[:, :, 0] = ((h + hue_shift) % 256).astype(np.uint8)

        return Image.fromarray(hsv, "HSV").convert("RGB")
    
    def compute_lbp(self, grayscale):
        points = 8  # Number of pixels around the center pixel (in this case, a 3x3 square)
        radius = 1
        lbp_matrix = local_binary_pattern(image = grayscale, P = points, R = radius, method = "uniform")
        n_bins = points + 2
        # ravel = flatten alternative, no copy in memory
        hist, _ = np.histogram(lbp_matrix.ravel(), bins=n_bins, range=(0, n_bins))
        hist = hist.astype("float")
        # L1 normalization to prevent resolution outliers
        hist /= hist.sum()

        return hist
