import matplotlib.pyplot as plt

from model import train_model
from util import split_chapters

path = "data/worm"
filename = "worm.txt"
sep = "\n" * 6

split_chapters(path, filename, sep)

pts_arrays = {vs: train_model(path, pca=2, vector_size=vs, window=10,
                              min_count=5, workers=6, epochs=40)
              for vs in [2, 8, 32, 128]}

fig, axs = plt.subplots(2, 2, figsize=(12, 9))
for i, (ax, (vs, pts)) in enumerate(zip(axs.flat, pts_arrays.items())):
    n = len(pts)
    colors = [(j / n, 0, 1 - j / n) for j in range(n)]
    ax.scatter([x for x, y in pts], [y for x, y in pts], c=colors)
    ax.set_title(f"Vector Size: {vs}")

plt.show()
