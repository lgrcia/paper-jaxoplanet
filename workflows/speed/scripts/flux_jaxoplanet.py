import jax

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

import numpy as np
from jaxoplanet.experimental.starry import Surface, Ylm
from jaxoplanet.experimental.starry.light_curves import surface_light_curve
from time import time

radius = float(snakemake.params.radius)
u = snakemake.params.u
N = int(snakemake.wildcards.N)
assert radius < 1
b = np.linspace(0, 1 + radius, N)
order = int(snakemake.wildcards.order)

if u is not None:
    surface = Surface(period=None, u=u)
else:
    l_max = 20
    y = Ylm.from_dense(np.ones((l_max + 1) ** 2))
    surface = Surface(period=None, y=y)

flux_function = jax.jit(
    jax.vmap(lambda b: surface_light_curve(surface, r=radius, y=b, z=10.0, order=order))
)

times = []

flux_function(b).block_until_ready()

for _ in range(20):
    t0 = time()
    flux_function(b).block_until_ready()
    times.append(time() - t0)

np.savez(snakemake.output[0], time=times, N=N, b=b)
