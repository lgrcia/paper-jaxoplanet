"""Plot of the errors on the flux for different radii, orders of the
Gauss-Legendre quadrature, and spherical harmonic degrees.
"""

from jaxoplanet.experimental.starry import Ylm
import matplotlib.pyplot as plt
import numpy as np

data = np.load(snakemake.input[0], allow_pickle=True)
l_max = snakemake.params.l_max
cmap = plt.get_cmap("plasma")

orders = np.array(data["orders"])
radii = np.array(data["rs"])

indices = {
    l: np.array([Ylm.index(l, m) for m in range(-l, l + 1)]) for l in range(l_max + 1)
}

errors_f = np.array([[i.max(0) for i in err] for err in data["errors_f"]])
errors_f = np.array([errors_f[:, :, indices[l]].max(-1) for l in range(l_max + 1)])

degrees = np.arange(l_max + 1)

fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharey=True)

for axi, ri in enumerate([0, 2, 4]):
    ax = axes[axi]
    for i, error in enumerate(errors_f[:, ri, :]):
        color = cmap(i / l_max)
        ax.plot(orders, error, ".-", color=color)
        ax.set_title(f"$r={radii[ri]}$")
        ax.set_yscale("log")

ax.set_ylim(1e-16, 1)
axes[1].set_xlabel("order $n$ of the Gauss-Legendre quadrature")
axes[0].set_ylabel("max relative error on the flux")

l_arr = [0, 5, 10, 20]

# Dummy curves & a legend
lines = [None for l in l_arr]
leg_labels = ["%d" % l for l in l_arr]
for i, l in enumerate(l_arr):
    (lines[i],) = ax.plot((0, 1), (1e-20, 1e-20), color=cmap(l / (l_max + 2)), lw=2)
leg = fig.legend(lines, leg_labels, title="Degree")

for axis in axes:
    axis.axhline(1e-6, ls="--", lw=1, color="k", alpha=0.2)
    axis.axhline(1e-9, ls="--", lw=1, color="k", alpha=0.2)


axes[0].annotate(
    "ppm",
    xy=(0, 1e-6),
    xycoords="data",
    xytext=(3, 3),
    textcoords="offset points",
    ha="left",
    va="bottom",
    alpha=0.75,
)
axes[0].annotate(
    "ppb",
    xy=(0, 1e-9),
    xycoords="data",
    xytext=(3, 3),
    textcoords="offset points",
    ha="left",
    va="bottom",
    alpha=0.75,
)

plt.tight_layout()
plt.savefig(snakemake.output[0])
