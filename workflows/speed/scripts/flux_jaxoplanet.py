import jax

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

import numpy as np
from jaxoplanet.experimental.starry import Surface, Ylm
from jaxoplanet.experimental.starry.light_curves import surface_light_curve
from time import time

radius = float(snakemake.wildcards.radius)
u = snakemake.params.u
N = int(snakemake.wildcards.N)
assert radius < 1
b = np.linspace(0, 1 + radius, N)
order = int(snakemake.wildcards.order)
lmax = snakemake.wildcards.lmax
lmax = None if lmax == "None" else int(lmax)

if lmax is None:
    from jaxoplanet.core.limb_dark import light_curve

    flux_function = jax.jit(jax.vmap(lambda b: light_curve(u, b, radius, order=order)))
else:
    l_max = lmax
    y = Ylm.from_dense(np.ones((l_max + 1) ** 2))
    surface = Surface(period=None, y=y, u=u)

    flux_function = jax.jit(
        jax.vmap(
            lambda b: surface_light_curve(surface, r=radius, y=b, z=10.0, order=order)
        )
    )

times = []

flux_function(b).block_until_ready()

for _ in range(20):
    t0 = time()
    flux_function(b).block_until_ready()
    times.append(time() - t0)

np.savez(snakemake.output[0], time=times, N=N, b=b)
