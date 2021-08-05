import matplotlib.pyplot as plt
import numpy as np

from model import train_model
from util import load_index, split_chapters

path = "data/worm"
filename = "worm.txt"
sep = "\n" * 6


def preprocessing():
    split_chapters(path, filename, sep)


def pca_experiment():
    pts_arrays = {vs: train_model(path, pca=2, vector_size=vs, epochs=40)
                  for vs in [2, 8, 32, 128]}

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    for i, (ax, (vs, pts)) in enumerate(zip(axs.flat, pts_arrays.items())):
        n = len(pts)
        colors = [(j / n, 0, 1 - j / n) for j in range(n)]
        ax.scatter([x for x, y in pts], [y for x, y in pts], c=colors)
        ax.set_title(f"Vector Size: {vs}")

    plt.show()


def window_experiment():
    pts_arrays = {w: train_model(path, pca=2, vector_size=2, window=w, epochs=40)
                  for w in [2, 5, 10, 15]}

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    for i, (ax, (w, pts)) in enumerate(zip(axs.flat, pts_arrays.items())):
        n = len(pts)
        colors = [(j / n, 0, 1 - j / n) for j in range(n)]
        ax.scatter([x for x, y in pts], [y for x, y in pts], c=colors)
        ax.set_title(f"Window Size: {w}")

    plt.show()


def min_count_experiment():
    pts_arrays = {mc: train_model(path, pca=2, vector_size=2, min_count=mc, epochs=40)
                  for mc in [0, 1, 2, 5]}

    fig, axs = plt.subplots(2, 2, figsize=(12, 9))
    for i, (ax, (mc, pts)) in enumerate(zip(axs.flat, pts_arrays.items())):
        n = len(pts)
        colors = [(j / n, 0, 1 - j / n) for j in range(n)]
        ax.scatter([x for x, y in pts], [y for x, y in pts], c=colors)
        ax.set_title(f"Min Count: {mc}")

    plt.show()


if __name__ == "__main__":
    pca_experiment()
    window_experiment()
    min_count_experiment()
