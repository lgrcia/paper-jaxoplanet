import jax

jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np
from jaxoplanet.experimental.starry.multiprecision import mp, utils

l_max = snakemake.params.l_max

q_order = int(eval(snakemake.input.jax[1].split("order=")[1].split(".")[0]))

data = {
    "small": {
        "jax": np.load(snakemake.input.jax[0], allow_pickle=True)["f"],
        "num": np.load(snakemake.input.num[0], allow_pickle=True)["f"],
        "starry": np.load(snakemake.input.starry[0], allow_pickle=True)["f"],
    },
    "large": {
        "jax": np.load(snakemake.input.jax[1], allow_pickle=True)["f"],
        "num": np.load(snakemake.input.num[1], allow_pickle=True)["f"],
        "starry": np.load(snakemake.input.starry[1], allow_pickle=True)["f"],
    },
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
    return frac


from jaxoplanet.experimental.starry.ylm import Ylm

indices = {
    l: np.array([Ylm.index(l, m) for m in range(-l, l + 1)]) for l in range(l_max + 1)
}

degree = np.arange(l_max + 1)
err = {
    "small": {
        "jax": errors(data["small"]["num"], data["small"]["jax"]).max(0),
        "starry": errors(data["small"]["num"], data["small"]["starry"]).max(0),
    },
    "large": {
        "jax": errors(data["large"]["num"], data["large"]["jax"]).max(0),
        "starry": errors(data["large"]["num"], data["large"]["starry"]).max(0),
    },
}


def err_per_degree(err):
    return [err[indices[l]].max() for l in range(l_max + 1)]


s_color = "k"
j_color = "C0"


fig = plt.figure(figsize=(8, 3))
x = np.arange(0, l_max + 2, 2)

ax1 = plt.subplot(121)
ax1.plot(
    degree, err_per_degree(err["small"]["jax"]), ".-", label="jaxoplanet", c=j_color
)
ax1.plot(
    degree, err_per_degree(err["small"]["starry"]), ".-", label="starry", c=s_color
)
ax1.set_xlabel("degree of spherical harmonics")
ax1.set_ylabel("relative error")
ax1.set_title("$r=0.01$")
ax1.set_yscale("log")
ax1.set_ylim(1e-17, 1e-4)
ax1.set_xticks(x)
ax1.annotate(
    f"q={q_order}",
    xy=(1 - 0.02, 0.05),
    xycoords="axes fraction",
    fontsize=10,
    ha="right",
)

ax2 = plt.subplot(122)
ax2.plot(
    degree, err_per_degree(err["large"]["jax"]), ".-", label="jaxoplanet", c=j_color
)
ax2.plot(
    degree, err_per_degree(err["large"]["starry"]), ".-", label="starry", c=s_color
)
ax2.legend()
ax2.set_xlabel("degree of spherical harmonics")
ax2.set_title("$r=100$")
ax2.set_yscale("log")
ax2.set_ylim(1e-17, 1e-4)
ax2.set_yticklabels([])
ax2.set_xticks(x)

for axis in (ax1, ax2):
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
