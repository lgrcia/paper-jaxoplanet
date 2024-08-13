import numpy as np

Ns = snakemake.params.Ns

jax_times_quadratic = np.array(
    [np.median(np.load(f)["time"]) for f in snakemake.input.jaxoplanet_quadratic]
)
starry_time_quadratic = np.array(
    [np.median(np.load(f)["time"]) for f in snakemake.input.starry_quadratic]
)
exoplanet_time_quadratic = np.array(
    [np.median(np.load(f)["time"]) for f in snakemake.input.exoplanet_quadratic]
)

jax_times_l20 = np.array(
    [np.median(np.load(f)["time"]) for f in snakemake.input.jaxoplanet_l20]
)
starry_time_l20 = np.array(
    [np.median(np.load(f)["time"]) for f in snakemake.input.starry_l20]
)

jax_times_l20_gpu = np.array(
    [np.median(np.load(f)["time"]) for f in snakemake.input.jaxoplanet_l20_gpu]
)

jax_times_quadratic_gpu = np.array(
    [np.median(np.load(f)["time"]) for f in snakemake.input.jaxoplanet_quadratic_gpu]
)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 3.5))

plt.subplot(121)
plt.plot(Ns, starry_time_quadratic * 1e3, ".-", label="starry", c="k")
plt.plot(Ns, exoplanet_time_quadratic * 1e3, ".-", label="exoplanet", c="C4")
plt.plot(Ns, jax_times_quadratic * 1e3, ".-", label="jaxoplanet (CPU)", c="C0")
plt.plot(Ns, jax_times_quadratic_gpu * 1e3, ".-", label="jaxoplanet (GPU)", c="C3")
plt.legend()
plt.xlabel("number of points")
plt.ylabel("time (ms)")
plt.title("Quadratic limb-darkened map ($l=2$)")
plt.yscale("log")
plt.xscale("log")

plt.subplot(122)
plt.plot(Ns, starry_time_l20 * 1e3, ".-", label="starry", c="k")
plt.plot(Ns, jax_times_l20 * 1e3, ".-", label="jaxoplanet (CPU)", c="C0")
plt.plot(Ns, jax_times_l20_gpu * 1e3, ".-", label="jaxoplanet (GPU)", c="C3")
plt.legend()
plt.xlabel("number of points")
plt.title("Non-uniform map ($l=20$)")
plt.yscale("log")
plt.xscale("log")

plt.tight_layout()
plt.savefig(snakemake.output[0])
