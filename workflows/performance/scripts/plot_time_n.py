import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

orders = snakemake.params.orders

data = {
    "starry": {
        1000: np.load(snakemake.input["starry_1000"]),
        # 10000: np.load(snakemake.input["starry_10000"]),
    },
    "jaxoplanet": {
        1000: {
            orders[i]: np.load(snakemake.input["jaxoplanet_1000"][i])
            for i in range(len(orders))
        },
        # 10000: {
        #     orders[i]: np.load(snakemake.input["jaxoplanet_10000"][i])
        #     for i in range(len(orders))
        # },
    },
}


def get_data(N):
    precisions = np.array(
        [
            np.abs(
                data["jaxoplanet"][N][order]["value"] - data["starry"][N]["value"]
            ).max()
            for order in orders
        ]
    )

    times = np.array([data["jaxoplanet"][N][order]["time"] for order in orders])

    reference_precision = np.min(precisions)
    reference_time = data["starry"][N]["time"]
    return precisions, times, reference_precision, reference_time


plt.figure(figsize=(5, 3.5))

plt.subplot(111)
precisions, times, reference_precision, reference_time = get_data(1000)
plt.plot(preci sions, times * 1e6, ".-", c="k", label="jaxoplanet (CPU)")
plt.plot(reference_precision, reference_time * 1e6, "+", label="starry (CPU)")
plt.axhline(reference_time * 1e6, c="C0", ls="-", alpha=0.2)
plt.axvline(reference_precision, c="C0", ls="-", alpha=0.2)
plt.xscale("log")
plt.xlim(1e-7, 0.5e-15)
plt.ylim(0)
plt.xlabel("relative error")
plt.ylabel("computation time (µs)")

for i, order in enumerate(orders):
    plt.text(precisions[i], times[i] * 1e6 + 20, f"{order}", fontsize=9, ha="right")

# plt.title(f"N = 1000")
plt.legend(loc="upper left")

# plt.subplot(122)
# precisions, times, reference_precision, reference_time = get_data(10000)
# plt.plot(precisions, times * 1e6, ".-", c="k", label="jaxoplanet (CPU)")
# plt.plot(reference_precision, reference_time * 1e6, "+", label="starry (CPU)")
# plt.axhline(reference_time * 1e6, c="C0", ls="-", alpha=0.2)
# plt.axvline(reference_precision, c="C0", ls="-", alpha=0.2)
# plt.xscale("log")
# plt.xlim(1e-7, 0.5e-15)
# plt.ylim(50)
# plt.xlabel("relative error")

# for i, order in enumerate(orders):
#     plt.text(precisions[i], times[i] * 1e6 + 20, f"{order}", fontsize=9, ha="right")

# plt.title(f"N = 10000")
# plt.legend(loc="upper left")

plt.tight_layout()
plt.savefig(snakemake.output[0])
