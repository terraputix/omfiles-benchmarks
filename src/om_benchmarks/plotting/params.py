import matplotlib
import matplotlib.pyplot as plt


def _set_matplotlib_behaviour():
    # plotting backend does not need to be interactive
    # this will otherwise cause problems when matplotlib objects are gc-ed
    matplotlib.use("Agg")

    # Configure matplotlib for better appearance
    plt.style.use("seaborn-v0_8-whitegrid")  # Use seaborn style for better defaults
    matplotlib.rcParams.update(
        {
            "text.usetex": True,  # Enable LaTeX rendering
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 12,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titlepad": 8,
            "figure.titlesize": 16,
            "axes.grid": True,
            "axes.grid.which": "both",
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": True,
            "axes.spines.bottom": True,
            "axes.linewidth": 0.8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 600,
            "figure.subplot.wspace": 0.35,
            "figure.subplot.hspace": 0.35,
        }
    )
