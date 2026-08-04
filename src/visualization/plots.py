import matplotlib.pyplot as plt


def plot_residuals(y_true, y_pred, output_path):
    """Plots true vs predicted bird counts."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, y_pred, alpha=0.7)
    ax.plot([0, max(y_true)], [0, max(y_true)], "r--")
    ax.set_xlabel("True Bird Count")
    ax.set_ylabel("Predicted Count")
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
