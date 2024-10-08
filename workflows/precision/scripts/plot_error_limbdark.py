import jax

jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import numpy as np
from jaxoplanet.experimental.starry.multiprecision import mp, utils

degree = snakemake.params.degree
q_order = int(snakemake.params.order)

data = {
    "small": {
        "jax": np.load(snakemake.input.jax[0], allow_pickle=True)["f"],
        "jax_nu": np.load(snakemake.input.jax_nu[0], allow_pickle=True)["f"],
        "num": np.load(snakemake.input.num[0], allow_pickle=True)["f"],
        "starry": np.load(snakemake.input.starry[0], allow_pickle=True)["f"],
        "exo": np.load(snakemake.input.exo[0], allow_pickle=True)["f"],
    },
    "large": {
        "jax": np.load(snakemake.input.jax[1], allow_pickle=True)["f"],
        "jax_nu": np.load(snakemake.input.jax_nu[0], allow_pickle=True)["f"],
        "num": np.load(snakemake.input.num[1], allow_pickle=True)["f"],
        "starry": np.load(snakemake.input.starry[1], allow_pickle=True)["f"],
        "exo": np.load(snakemake.input.exo[1], allow_pickle=True)["f"],
    },
}


def errors(M1, M2):
    return np.abs(utils.diff_mp(M1, M2))


err = {
    "small": {
        "jax": errors(data["small"]["jax"], data["small"]["num"]).max(1)[1:],
        "jax_nu": errors(data["small"]["jax_nu"], data["small"]["num"]).max(1)[1:],
        "starry": errors(data["small"]["starry"], data["small"]["num"]).max(1)[1:],
        "exo": errors(data["small"]["exo"], data["small"]["num"][0:3, :]).max(1)[1:],
    },
    "large": {
        "jax": errors(data["large"]["jax"], data["large"]["num"]).max(1)[1:],
        "jax_nu": errors(data["small"]["jax_nu"], data["small"]["num"]).max(1)[1:],
        "starry": errors(data["large"]["starry"], data["large"]["num"]).max(1)[1:],
        "exo": errors(data["large"]["exo"], data["large"]["num"][0:3, :]).max(1)[1:],
    },
}
color = {
    "jax": "C0",
    "starry": "k",
    "exo": "C4",
}


fig = plt.figure(figsize=(8, 3))
x = np.arange(0, degree + 2, 2)

degrees = np.arange(1, degree + 1)

ax1 = plt.subplot(121)
ax1.plot(degrees, err["small"]["jax"], ".-", label="jaxoplanet (limb-darkening)", c="c")
ax1.plot(degrees, err["small"]["jax_nu"], ".-", label="jaxoplanet", c=color["jax"])
ax1.plot(degrees, err["small"]["starry"], ".-", label="starry", c=color["starry"])
ax1.plot(degrees[0:2], err["small"]["exo"], ".-", label="exoplanet", c=color["exo"])
ax1.set_xlabel("order of limb-darkening")
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
ax2.plot(degrees, err["large"]["starry"], ".-", label="starry", c=color["starry"])
ax2.plot(degrees[0:2], err["large"]["exo"], ".-", label="exoplanet", c=color["exo"])
ax2.plot(degrees, err["large"]["jax_nu"], ".-", label="jaxoplanet", c=color["jax"])
ax2.plot(degrees, err["large"]["jax"], ".-", label="jaxoplanet (PLD)", c="c")
ax2.legend()
ax2.set_xlabel("order of limb-darkening")
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
