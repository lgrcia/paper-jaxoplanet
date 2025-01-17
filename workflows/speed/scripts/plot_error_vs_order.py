import numpy as np

orders = snakemake.params.orders

jax_times_quadratic = np.array(
    [np.median(np.load(f)["time"]) for f in snakemake.input.jaxoplanet_quadratic]
)
jax_times_quadratic_gpu = np.array(
    [np.median(np.load(f)["time"]) for f in snakemake.input.jaxoplanet_quadratic_gpu]
)
starry_time_quadratic = np.median(np.load(snakemake.input.starry_quadratic)["time"])
exoplanet_time_quadratic = np.median(
    np.load(snakemake.input.exoplanet_quadratic)["time"]
)

jax_times_lmax = np.array(
    [np.median(np.load(f)["time"]) for f in snakemake.input.jaxoplanet_lmax]
)

jax_times_lmax_gpu = np.array(
    [np.median(np.load(f)["time"]) for f in snakemake.input.jaxoplanet_lmax_gpu]
)

starry_times_lmax = np.median(np.load(snakemake.input.starry_lmax)["time"])

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 3.5))

plt.subplot(121)
plt.axhline(starry_time_quadratic * 1e3, color="C3", label="starry")
plt.axhline(exoplanet_time_quadratic * 1e3, color="k", label="exoplanet")
plt.plot(orders, jax_times_quadratic * 1e3, ".-", label="jaxoplanet (CPU)", c="C0")
plt.plot(orders, jax_times_quadratic_gpu * 1e3, ".--", label="jaxoplanet (GPU)", c="C0")
plt.legend()
plt.xlabel("order of Gauss-Legendre quadrature")
plt.ylabel("time (ms)")
plt.title("Quadratic limb-darkened map ($l=2$)")
plt.yscale("log")

plt.subplot(122)
plt.axhline(starry_times_lmax * 1e3, color="C3", label="starry")
plt.plot(orders, jax_times_lmax * 1e3, ".-", label="jaxoplanet (CPU)")
plt.plot(orders, jax_times_lmax_gpu * 1e3, ".--", label="jaxoplanet (GPU)", c="C0")
plt.legend()
plt.xlabel("order of Gauss-Legendre quadrature")
plt.title("Non-uniform map ($l=20$)")
plt.yscale("log")

plt.tight_layout()
plt.savefig(snakemake.output[0])
