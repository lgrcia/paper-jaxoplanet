import jax
from tqdm import tqdm

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

import timeit
import numpy as np
from jaxoplanet.experimental.starry import Surface, Ylm
from jaxoplanet.experimental.starry.light_curves import surface_light_curve
import jax.numpy as jnp

radius = float(snakemake.params.radius)
u = snakemake.params.u
N = int(snakemake.wildcards.N)
assert radius < 1
b = np.linspace(0, 1 + radius, N)
order = int(snakemake.wildcards.order)


def timeit_f(strf, n):
    times = np.array(timeit.repeat(f"{strf}", number=n, globals=globals()))[1:] / (
        n - 1
    )
    return times


if u is not None:
    surface = Surface(period=None, u=u)
    n = 50
else:
    l_max = 20
    y = Ylm.from_dense(np.ones((l_max + 1) ** 2))
    surface = Surface(period=None, y=y)
    n = 10


flux_function = jax.jit(
    jax.vmap(lambda b: surface_light_curve(surface, r=radius, y=b, z=10.0, order=order))
)
times = timeit_f(f"flux_function(b).block_until_ready()", n=n)


np.savez(snakemake.output[0], time=times, N=N, b=b)
