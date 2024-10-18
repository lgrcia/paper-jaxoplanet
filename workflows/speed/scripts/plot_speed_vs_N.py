import numpy as np

Ns = snakemake.params.Ns
order_lmax = snakemake.params.order_lmax
order_quad = snakemake.params.order_quad

jax_times_quadratic = np.array(
    [np.median(np.load(f)["time"]) for f in snakemake.input.jaxoplanet_quadratic]
)
starry_time_quadratic = np.array(
    [np.median(np.load(f)["time"]) for f in snakemake.input.starry_quadratic]
)

exoplanet_time_quadratic = np.array(
    [np.median(np.load(f)["time"]) for f in snakemake.input.exoplanet_quadratic]
)

jax_times_lmax = np.array(
    [np.median(np.load(f)["time"]) for f in snakemake.input.jaxoplanet_lmax]
)
starry_time_lmax = np.array(
    [np.median(np.load(f)["time"]) for f in snakemake.input.starry_lmax]
)

jax_times_lmax_gpu = np.array(
    [np.median(np.load(f)["time"]) for f in snakemake.input.jaxoplanet_lmax_gpu]
)

jax_times_quadratic_gpu = np.array(
    [np.median(np.load(f)["time"]) for f in snakemake.input.jaxoplanet_quadratic_gpu]
)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 3.5))

plt.subplot(121)
plt.plot(Ns, starry_time_quadratic * 1e3, ".-", label="starry", c="C3")
plt.plot(Ns, exoplanet_time_quadratic * 1e3, ".-", label="exoplanet", c="k")
plt.plot(Ns, jax_times_quadratic * 1e3, ".-", label="jaxoplanet (CPU)", c="C0")
plt.plot(Ns, jax_times_quadratic_gpu * 1e3, ".--", label="jaxoplanet (GPU)", c="C0")
plt.legend()
plt.xlabel("number of points")
plt.ylabel("time (ms)")
plt.title("Quadratic limb-darkened map ($l=2$)")
plt.yscale("log")
plt.xscale("log")
plt.annotate(
    f"q={order_quad}",
    xy=(1 - 0.02, 0.05),
    xycoords="axes fraction",
    fontsize=10,
    ha="right",
)

plt.subplot(122)
plt.plot(Ns, starry_time_lmax * 1e3, ".-", label="starry", c="C3")
plt.plot(Ns, jax_times_lmax * 1e3, ".-", label="jaxoplanet (CPU)", c="C0")
plt.plot(Ns, jax_times_lmax_gpu * 1e3, ".--", label="jaxoplanet (GPU)", c="C0")
plt.legend()
plt.xlabel("number of points")
plt.title("Non-uniform map ($l=20$)")
plt.yscale("log")
plt.xscale("log")
plt.annotate(
    f"q={order_lmax}",
    xy=(1 - 0.02, 0.05),
    xycoords="axes fraction",
    fontsize=10,
    ha="right",
)

plt.tight_layout()
plt.savefig(snakemake.output[0])
