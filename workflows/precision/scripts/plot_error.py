# Plot the errors of s, f on each of the (l,m) spherical harmonic components
# The scaled error is computed in the `errors` function and the error for a
# given degree l is the maximum error across all m \in [-l, l] components of that degree
# (indices are for each degree are stored in the `indices` dictionary)

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from jaxoplanet.experimental.starry.ylm import Ylm
from jaxoplanet.experimental.starry.mpcore import mp, utils

ref = "num"
comp = "comp"

data = {
    "small": {
        ref: np.load(snakemake.input.ref[0], allow_pickle=True),
        comp: np.load(snakemake.input.comp[0], allow_pickle=True),
    },
    "large": {
        ref: np.load(snakemake.input.ref[1], allow_pickle=True),
        comp: np.load(snakemake.input.comp[1], allow_pickle=True),
    },
}

S = {
    "small": {
        ref: data["small"][ref]["s"],
        comp: data["small"][comp]["s"],
    },
    "large": {
        ref: data["large"][ref]["s"],
        comp: data["large"][comp]["s"],
    },
}

F = {
    "small": {
        ref: data["small"][ref]["f"],
        comp: data["small"][comp]["f"],
    },
    "large": {
        ref: data["large"][ref]["f"],
        comp: data["large"][comp]["f"],
    },
}

data_b = {
    "small": np.load(snakemake.input.b[0], allow_pickle=True),
    "large": np.load(snakemake.input.b[1], allow_pickle=True),
}

Bs = {"small": data_b["small"]["bs"], "large": data_b["large"]["bs"]}

labels = {
    "small": data_b["small"]["labels"],
    "large": data_b["large"]["labels"],
}

r = {
    "small": data_b["small"]["r"],
    "large": data_b["large"]["r"],
}


def mp_diff(M_mp, M_np):
    return utils.to_numpy(utils.to_mp(M_np) - M_mp)


def errors(M_mp, M_np):
    if isinstance(M_mp, mp.matrix):
        d = mp_diff(M_mp, M_np)
    else:
        d = M_mp - M_np
    rel = np.abs(d)
    frac = np.abs(
        d
        / max(1e-9, np.nanmax(np.min([np.abs(utils.to_numpy(M_mp)), np.abs(M_np)], 0)))
    )
    frac[rel < 1e-16] = 1e-16
    return rel, frac


def plot_occ(b, r, ax):
    ax.set_aspect(1)
    ax.axis("off")

    occultor = plt.Circle((0, 0), 1, facecolor=("k", 0.1), edgecolor="0.8")
    ax.add_artist(occultor)
    occultor = plt.Circle((b, 0), r, facecolor=("k", 0.4), edgecolor="0.3")
    ax.add_artist(occultor)
    h = 2 * r
    ax.set_ylim(-h, h)
    ax.set_xlim(-h + b, h + b)


from matplotlib import gridspec
import matplotlib.pyplot as plt

l_arr = [0, 1, 2, 3, 5, 8, 10, 13, 15, 18, 20]

l_max = snakemake.params.l_max

cmap = plt.get_cmap("plasma")

fig = plt.figure(constrained_layout=True, figsize=(8.0, 7))

N = 8
gs = gridspec.GridSpec(3, N, figure=fig, height_ratios=[1, 2, 2])

# circles

b_vals = {
    "small": [0, 1 - r["small"], 1, 1 + r["small"]],
    "large": np.linspace(r["large"] - 1, r["large"] + 1, N // 2),
}

for i in range(N // 2):
    ax = fig.add_subplot(gs[0, i])

    if i == 0:
        ax.set_title(f"$r={r['small']}$", x=2.5, y=1.2, ha="center")

    b = b_vals["small"][i]
    plot_occ(b, r["small"], ax)
    h = 2 * r["small"]
    ax.set_ylim(-h, h)
    ax.set_xlim(-h + b, h + b)

for i in range(N // 2):
    ax = fig.add_subplot(gs[0, N // 2 + i])
    if i == 0:
        ax.set_title(f"$r={r['large']:0.0f}$", x=2.5, y=1.2, ha="center")
    b = b_vals["large"][i]
    plot_occ(b, r["large"], ax)
    h = 2.0
    ax.set_ylim(-h, h)
    ax.set_xlim(-h, h)

indices = {
    l: np.array([Ylm.index(l, m) for m in range(-l, l + 1)]) for l in range(l_max + 1)
}

for i, size in enumerate(["small", "large"]):

    ax_top = fig.add_subplot(gs[1, N // 2 * i : (N // 2 * i + N // 2)])
    ax_bot = fig.add_subplot(gs[2, N // 2 * i : (N // 2 * i + N // 2)])

    err_s = errors(S[size][ref], S[size][comp])[1]
    err_f = errors(F[size][ref], F[size][comp])[1]

    for l in range(l_max + 1):
        n = indices[l]
        ax_top.plot(err_s.T[n].max(0), color=cmap(l / (l_max)), zorder=-1)
        ax_bot.plot(err_f.T[n].max(0), color=cmap(l / (l_max)), zorder=-1)

    bs = np.concatenate(Bs[size])

    ax_top.set_yscale("log")
    ax_bot.set_yscale("log")
    ax_top.set_ylim(1e-16, 1e-5)
    ax_bot.set_ylim(1e-16, 1e-5)
    ax_top.set_xlim(0, len(bs))
    ax_bot.set_xlim(0, len(bs))
    ax_top.set_xticklabels([])

    bounds = np.cumsum([0] + [len(bb) for bb in Bs[size]]) - 1
    bounds[0] = 0

    # for v in bounds:
    #     for axis in [ax_top, ax_bot]:
    #         axis.axvline(v, lw=0.5, color="k", alpha=0.5, zorder=10, ls="--")

    ax_bot.set_xticks(bounds)
    ax_bot.set_xticklabels(labels[size], rotation=45, fontsize=10, ha="right")

    if i == 0:
        ax_top.set_ylabel("Error on $s$")
        ax_bot.set_ylabel("Error on $f$")

    ax_bot.set_xlabel("Impact parameter")

    if i == 1:
        ax_top.set_yticklabels([])
        ax_bot.set_yticklabels([])

    for axis in (ax_top, ax_bot):
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


# Dummy curves & a legend
lines = [None for l in l_arr]
leg_labels = ["%d" % l for l in l_arr]
for i, l in enumerate(l_arr):
    (lines[i],) = ax_top.plot((0, 1), (1e-20, 1e-20), color=cmap(l / (l_max + 2)), lw=2)
leg = fig.legend(
    lines, leg_labels, title="Degree", bbox_to_anchor=(0.52, 0.1, 0.5, 0.5)
)

plt.savefig(snakemake.output[0], bbox_inches="tight")
