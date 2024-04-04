import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_heatmap(
    array: np.ndarray,
    fontsize: int = 8,
    fmt: str = ".2f",
    save_path: str = None,
    vmin: float = None,
    vmax: float = None
):
    plt.figure()

    total = 0
    n = 0
    for i in range(len(array)):
        for j in range(i + 1, len(array)):
            total += array[i, j]
            n += 1
    avg = round(total / n, 4)
    sns.heatmap(
        array,
        annot=True,
        fmt=fmt,
        annot_kws={"fontsize": fontsize},
        vmin=vmin,
        vmax=vmax,
    )
    plt.title(f"Average Similarity = {avg}")
    if save_path:
        plt.savefig(save_path)