import jax
from tqdm import tqdm

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

import numpy as np
from jaxoplanet.starry import Surface
from jaxoplanet.starry.light_curves import surface_light_curve

b, r = np.load(snakemake.input[0]).values()
order = int(snakemake.wildcards.order)

surface = Surface(period=None, u=(1,))

f = jax.jit(
    jax.vmap(surface_light_curve, in_axes=(None, None, None, 0, None, None, None)),
    static_argnames=("order",),
)

result = np.zeros((len(r), len(b)))


def flux(order):
    for i, _r in enumerate(tqdm(r)):
        calc = f(surface, _r, 0.0, b, 10.0, 0.0, order)
        result[i, :] = calc

    return result


calc = flux(order)

np.savez(snakemake.output[0], flux=calc)
