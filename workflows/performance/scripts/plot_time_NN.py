import matplotlib.pyplot as plt
import numpy as np


Ns = snakemake.params.Ns
data = {
    "starry": [np.load(f)["time"] for f in snakemake.input["starry"]],
    "jaxoplanet": [np.load(f)["time"] for f in snakemake.input["jaxoplanet"]],
}

plt.plot(Ns, data["starry"], ".-", label="starry (CPU)")
plt.plot(Ns, data["jaxoplanet"], ".-", label="jaxoplanet (CPU)")
plt.xlabel("N")
plt.ylabel("computation time (s)")
plt.xscale("log")
plt.yscale("log")

plt.legend()
plt.tight_layout()
plt.savefig(snakemake.output[0])
