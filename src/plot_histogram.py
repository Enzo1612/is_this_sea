import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sample import Sample

base_dir = os.path.dirname(os.path.dirname(__file__))
output_dir = os.path.join(base_dir, "histograms")
os.makedirs(output_dir, exist_ok=True)

image_path = os.path.join(base_dir, "data", "raw", "Init", "Mer", "838s.jpg")
img = Image.open(image_path)
s = Sample()

hist_hs = s.computeHisto(img)
hist_rgb = s.computeWeightedHisto(img)
hist_hog = s.computeHog(img)
hist_lbp = s.compute_lbp(np.array(img.convert("L")))

def plot_histogram(data, title, filename, color='blue'):
    plt.figure(figsize=(10, 5))
    
    # Ensures data is a 1D array for plotting
    data = np.asanyarray(data).flatten()
    bins = np.arange(len(data))
    
    # Bar plot
    plt.bar(bins, data, color=color, width=1.0, edgecolor='black', linewidth=0.5, alpha=0.8)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Bin index")
    plt.ylabel("Frequency / Value")
    
    if len(data) > 20:
        # Prevent overcrowding of x-ticks for large histograms
        step = max(1, len(data) // 15)
        plt.xticks(bins[::step])
    else:
        plt.xticks(bins)

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()


plot_histogram(hist_hs, "HSV Descriptor", "hist_hsv.png", color='skyblue')
plot_histogram(hist_rgb, "Weighted RGB Descriptor", "hist_rgb.png", color='salmon')
plot_histogram(hist_hog, "HOG (Histogram of Oriented Gradients)", "hist_hog.png", color='gray')
plot_histogram(hist_lbp, "LBP (Local Binary Patterns)", "hist_lbp.png", color='green')

print(f"Graphs saved to: {output_dir}")