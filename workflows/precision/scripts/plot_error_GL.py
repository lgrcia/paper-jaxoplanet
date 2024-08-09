import numpy as np
import matplotlib.pyplot as plt

data = np.load(snakemake.input[0], allow_pickle=True)

errors_f = np.array([[i.max() for i in err] for err in data["errors_f"]])
errors_s = np.array([[i.max() for i in err] for err in data["errors_s"]])
orders = np.array(data["orders"])
radii = np.array(data["rs"])

plt.figure(figsize=(8, 3))
fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
ax1, ax2 = axes
cmap = plt.get_cmap("plasma")

for i, r in enumerate(radii):
    color = cmap(i / len(radii))
    axes[0].plot(orders, errors_s[i], ".-", label=f"r={r}", color=color)
    axes[1].plot(orders, errors_f[i], ".-", label=f"r={r}", color=color)

for ax in axes:
    ax.set_yscale("log")
    ax.set_xlabel("order")
    ax.set_ylim(1e-16, 1)

axes[0].set_ylabel("max relative error")
axes[0].set_title("solution vector")
axes[1].set_title("flux")
axes[0].legend()

plt.tight_layout()

for axis in axes:
    axis.axhline(1e-6, ls="--", lw=1, color="k", alpha=0.2)
    axis.axhline(1e-9, ls="--", lw=1, color="k", alpha=0.2)
    axis.annotate(
        "ppm",
        xy=(0, 1e-6),
        xycoords="data",
        xytext=(3, -3),
        textcoords="offset points",
        ha="left",
        va="top",
        alpha=0.75,
    )
    axis.annotate(
        "ppb",
        xy=(0, 1e-9),
        xycoords="data",
        xytext=(3, -3),
        textcoords="offset points",
        ha="left",
        va="top",
        alpha=0.75,
    )

plt.tight_layout()
plt.savefig(snakemake.output[0])
