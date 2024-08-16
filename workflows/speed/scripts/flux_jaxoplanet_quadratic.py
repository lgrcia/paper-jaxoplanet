import jax

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

import numpy as np
from jaxoplanet.core.limb_dark import light_curve
from time import time

radius = float(snakemake.params.radius)
u = snakemake.params.u
N = int(snakemake.wildcards.N)
assert radius < 1
b = np.linspace(0, 1 + radius, N)
order = int(snakemake.wildcards.order)


flux_function = jax.jit(jax.vmap(lambda b: light_curve(u, b, radius, order=order)))
flux_function(b).block_until_ready()

times = []

for _ in range(20):
    t0 = time()
    flux_function(b).block_until_ready()
    times.append(time() - t0)


np.savez(snakemake.output[0], time=times, N=N, b=b)
