import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from jaxoplanet.experimental.starry.multiprecision import mp
from jaxoplanet.experimental.starry.multiprecision.utils import diff_mp

l_max = snakemake.params.l_max

other_data = np.load(snakemake.input[1], allow_pickle=True)
jax_A = np.array(other_data["jax_A"])
sta_A = other_data["sta_A"]
num_A = other_data["num_A"]

jax_sT = np.array(other_data["jax_sT"])
sta_sT = other_data["sta_sT"]
num_sT = other_data["num_sT"]

jax_R = [np.array(r) for r in other_data["jax_R"]]
sta_R = other_data["sta_R"]
num_R = other_data["num_R"]

b = other_data["b"]
r = other_data["r"]

u = other_data["u"]
theta = other_data["theta"]


def diff_R(M1, M2):
    return np.hstack([np.abs(diff_mp(a, b)).max(0) for a, b in zip(M1, M2)])


import matplotlib.pyplot as plt

starry_color = "0.6"
jax_color = "C0"

fig, subplot = plt.subplots(1, 3, figsize=(8.0, 3), sharey=True, sharex=True)
ax = subplot[0]
ax.plot(np.abs(diff_mp(num_sT, jax_sT)).T, label="jaxoplanet", color=jax_color)
ax.plot(np.abs(diff_mp(num_sT, sta_sT)).T, label="starry", color=starry_color)
ax.set_title(r"$s^{T}$")
ax.set_ylabel("Error")
# annotation in the corner
ax.annotate(
    f"$b={b}$\n$r={r}$",
    xy=(0.05, 0.95),
    xycoords="axes fraction",
    textcoords="axes fraction",
    color="black",
    va="top",
)
ax.legend()

ax = subplot[1]
ax.axhline(2e-17, color=jax_color)
ax.plot(np.abs(diff_mp(num_A, jax_A)).max(0), label="jaxoplanet", color=starry_color)
ax.plot(np.abs(diff_mp(num_A, sta_A)).max(0), label="starry", color=starry_color)
ax.set_title(r"$A$")
ax.set_xlabel("spherical harnonics degree")

ax = subplot[2]
plt.plot(np.abs(diff_R(num_R, jax_R)).T, label="jaxoplanet", color=jax_color)
plt.plot(np.abs(diff_R(num_R, sta_R)).T, label="starry", color=starry_color)
ax.set_title(r"$R$")
degrees = np.array([0, 5, *np.arange(8, 21, 2)])
ax.set_xticks((degrees + 1) ** 2)
ax.set_xticklabels(degrees)
ax.annotate(
    f"$u=(1, 0, 0)$\n" + rf"$\theta={theta}$",
    xy=(0.05, 0.95),
    xycoords="axes fraction",
    textcoords="axes fraction",
    color="black",
    va="top",
)


plt.yscale("log")
plt.ylim(1e-17, 1e-6)
plt.tight_layout()

plt.savefig(snakemake.output[0])
