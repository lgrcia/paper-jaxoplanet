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
    relative_errors = []

    for order in orders:
        error = np.abs(
            data["jaxoplanet"][N][order]["value"] - data["starry"][N]["value"]
        )
        norm = np.abs(data["jaxoplanet"][N][order]["value"])
        norm[norm < 1e-9] = 1e-9
        error /= norm
        relative_errors.append(error.max())

    relative_errors = np.array(relative_errors)

    times = np.array([data["jaxoplanet"][N][order]["time"] for order in orders])

    reference_error = 1e-15
    norm = np.abs(data["jaxoplanet"][N][order]["value"])
    norm[norm < 1e-9] = 1e-9
    reference_relative_error = reference_error / norm

    reference_time = data["starry"][N]["time"]
    return relative_errors, times, reference_relative_error.max(), reference_time


plt.figure(figsize=(5, 3.5))

ax = plt.subplot(111)
precisions, times, reference_precision, reference_time = get_data(1000)
print(precisions.shape)
plt.plot(precisions, times * 1e6, ".-", c="k", label="jaxoplanet (CPU)")
plt.plot(reference_precision, reference_time * 1e6, "+", label="starry (CPU)")
plt.axhline(reference_time * 1e6, c="C0", ls="-", alpha=0.2)
plt.axvline(reference_precision, c="C0", ls="-", alpha=0.2)
plt.xscale("log")
# plt.yscale("log")
xlim = plt.xlim()
plt.xlim(xlim[1], xlim[0])
# plt.xlim(1e-7, 0.5e-15)
plt.ylim(0)
plt.xlabel("relative error")
plt.ylabel("computation time (µs)")
y_labels = ax.get_yticks()
import matplotlib.ticker as ticker

ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))

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
