from sample import Sample
from PIL import Image
import matplotlib.pyplot as plt


i = Image.open(r"data/raw/Init/Mer/838s.jpg")

s = Sample()

hist = s.computeHisto(i)

plt.hist(hist[2])
plt.show()