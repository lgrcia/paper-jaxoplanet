import numpy as np

Ns = snakemake.params.Ns
order_l20 = snakemake.params.order_l20
order_quad = snakemake.params.order_quad

jax_times_quadratic = np.array(
    [np.median(np.load(f)["time"]) for f in snakemake.input.jaxoplanet_quadratic]
)
exoplanet_time_quadratic = np.array(
    [np.median(np.load(f)["time"]) for f in snakemake.input.exoplanet_quadratic]
)

jax_times_quadratic_gpu = np.array(
    [np.median(np.load(f)["time"]) for f in snakemake.input.jaxoplanet_quadratic_gpu]
)

jax_times_l3 = np.array(
    [np.median(np.load(f)["time"]) for f in snakemake.input.jaxoplanet_l3]
)
starry_time_l3 = np.array(
    [np.median(np.load(f)["time"]) for f in snakemake.input.starry_l3]
)

jax_times_l3_gpu = np.array(
    [np.median(np.load(f)["time"]) for f in snakemake.input.jaxoplanet_l3_gpu]
)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 3.5))

plt.subplot(121)
plt.plot(Ns, exoplanet_time_quadratic * 1e3, ".-", label="exoplanet", c="k")
plt.plot(Ns, jax_times_quadratic * 1e3, ".-", label="jaxoplanet (CPU)", c="C0")
plt.plot(Ns, jax_times_quadratic_gpu * 1e3, ".--", label="jaxoplanet (GPU)", c="C0")
plt.legend()
plt.xlabel("number of points")
plt.ylabel("time (ms)")
plt.title("TRAPPIST-1 (quadratic, $r\sim 0.1$)")
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
plt.plot(Ns, starry_time_l3 * 1e3, ".-", label="starry", c="k")
plt.plot(Ns, jax_times_l3 * 1e3, ".-", label="jaxoplanet (CPU)", c="C0")
plt.plot(Ns, jax_times_l3_gpu * 1e3, ".--", label="jaxoplanet (GPU)", c="C0")
plt.legend()
plt.xlabel("number of points")
plt.title(r"HD 189733b ($l_{max}=3$, r~0.3)")
plt.yscale("log")
plt.xscale("log")
plt.annotate(
    f"q={order_l20}",
    xy=(1 - 0.02, 0.05),
    xycoords="axes fraction",
    fontsize=10,
    ha="right",
)

plt.tight_layout()
plt.savefig(snakemake.output[0])
