from PIL import Image, ImageEnhance
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.feature import hog

import model


class Sample:

    def make_path(self, path, label):
        samples = []
        for image_array in os.listdir(path):
            #TODO: for each image, apply modifications as well (rotation, flip, luminosity)
            img_path = os.path.join(path, image_array)
            image = Image.open(img_path)
            resized = self.resizeImage(image, 200, 200)
            samples.append({
                "name_path": img_path,
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

        # Augmentation séparée
        # test_sample = self.apply_modifications_to_samples(test_sample, apply_rotation=apply_rotation, apply_flip=apply_flip, apply_brightness_modification=apply_brightness_modification)
        train_sample = self.apply_modifications_to_samples(train_sample, apply_rotation=apply_rotation, apply_flip=apply_flip, apply_brightness_modification=apply_brightness_modification)

        return (train_sample, test_sample)

    def resizeImage(self, PImage, h, l):
        return PImage.resize((h, l))
    
    def computeHisto(self, PImage): 
        if PImage.mode != 'RGB':
            PImage = PImage.convert('RGB')
        r, g, b = PImage.split()
        return [
            r.histogram(), 
            g.histogram(), 
            b.histogram()
        ]

    def computeWeightedHisto(self, PImage, bottom_weight=2.0, bottom_ratio=0.33):
        if PImage.mode != 'RGB':
            PImage = PImage.convert('RGB')

        width, height = PImage.size
        # bottom_height = max(1, int(height * bottom_ratio))  # At least 1 pixel

        # # Top box is the rectangle from (0, 0){top left} to (width, height - bottom_height){bottom right}
        # top_box = (0, 0, width, height - bottom_height)
        # # Bottom box is the rectangle from (0, height - bottom_height){top left} to (width, height){bottom right}
        # bottom_box = (0, height - bottom_height, width, height)

        # top = PImage.crop(top_box)
        # bottom = PImage.crop(bottom_box)

        # r_top, g_top, b_top = top.split()
        # r_bottom, g_bottom, b_bottom = bottom.split()

        # r_hist = np.array(r_top.histogram(), dtype=float) + np.array(r_bottom.histogram(), dtype=float)
        # g_hist = np.array(g_top.histogram(), dtype=float) + np.array(g_bottom.histogram(), dtype=float)
        # # Only weight the bottom blue histogram
        # b_hist = np.array(b_top.histogram(), dtype=float) + (bottom_weight * np.array(b_bottom.histogram(), dtype=float))

        # return [
        #     r_hist.tolist(),
        #     g_hist.tolist(),
        #     b_hist.tolist()
        # ]

        # Initialize histograms to zero
        r_hist = np.zeros(256, dtype=float)
        g_hist = np.zeros(256, dtype=float)
        b_hist = np.zeros(256, dtype=float)

        # Process the top part of the image
        top_height = height - int(height * bottom_ratio)
        if top_height > 0:
            top_box = (0, 0, width, top_height)
            top = PImage.crop(top_box)
            if top.size[0] > 0 and top.size[1] > 0:
                r_top, g_top, b_top = top.split()
                r_hist += np.array(r_top.histogram(), dtype=float)
                g_hist += np.array(g_top.histogram(), dtype=float)
                b_hist += np.array(b_top.histogram(), dtype=float)

        # Process the bottom part of the image
        bottom_start = height - int(height * bottom_ratio)
        if bottom_start < height:
            bottom_box = (0, bottom_start, width, height)
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

    def apply_modifications_to_samples(self, samples, apply_rotation=True, apply_flip=True, apply_brightness_modification=True):
        modified_samples = []
        for sample in samples:
            modified_images = self.apply_modifications(sample["resized_image"], apply_rotation=apply_rotation, apply_flip=apply_flip, apply_brightness_modification=apply_brightness_modification)
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

    def apply_modifications(self, PImage, apply_rotation=True, apply_flip=True, apply_brightness_modification=True):
        if (PImage.mode != "RGB"):
            PImage = PImage.convert("RGB")

        base_images = []
        # Start with the original image
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

        # Apply brightness modifications to all geometrically augmented images
        final_images = []
        for img in all_geometric_images:
            final_images.append(img) # Keep the image with original brightness
            # Add versions with modified brightness
            if apply_brightness_modification:
                final_images.append(ImageEnhance.Brightness(img).enhance(0.7))
                final_images.append(ImageEnhance.Brightness(img).enhance(1.3))

        return final_images
    
    def apply_modifications_(self, PImage):
        if (PImage.mode != "RGB"):
            PImage = PImage.convert("RGB")

        modified_images = []
        # Original image
        modified_images.append(PImage)

        # Rotations (sur l'original)
        modified_images.extend([PImage.rotate(angle) for angle in [90, 180, 270]])

        # Flips (sur les images actuelles: original + rotations)
        current = list(modified_images)
        modified_images.extend([img.transpose(Image.FLIP_LEFT_RIGHT) for img in current])
        modified_images.extend([img.transpose(Image.FLIP_TOP_BOTTOM) for img in current])

        # Luminosité (sur toutes les images actuelles)
        current = list(modified_images)
        for factor in [0.5, 1.5]:
            modified_images.extend([
                ImageEnhance.Brightness(img).enhance(factor) for img in current
            ])

        return modified_images
    
    

if __name__ == "__main__":
    s = Sample()
    sample = s.buildSampleFromPath()

    classifieur, X_test, y_test = model.fitFromHisto(sample)
    sample = model.predictFromHisto(sample, classifieur)
    model.cross_val(classifieur, X_test, y_test)
    model.compute_empirical_error(sample, classifieur)

