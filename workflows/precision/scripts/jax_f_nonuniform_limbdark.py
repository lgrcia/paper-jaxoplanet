import jax

jax.config.update("jax_enable_x64", True)

from jaxoplanet.starry import Surface
from jaxoplanet.starry.light_curves import surface_light_curve
import numpy as np
from tqdm import tqdm


r = float(snakemake.wildcards.r)
degree = snakemake.params.degree
bs = np.load(snakemake.input[0], allow_pickle=True)["bs"]
b = np.abs(np.concatenate(bs))
order = snakemake.params.order


def u(deg):
    u = np.zeros(degree)
    if degree == 0:
        return u
    else:
        u[deg - 1] = 1
        return u


function = jax.vmap(
    lambda deg, b: surface_light_curve(
        Surface(u=u(deg)), y=b, z=10.0, r=r, order=order, higher_precision=True
    ),
    (None, 0),
)

fluxes = np.array([function(deg, b) for deg in tqdm(range(degree + 1))])
np.savez(snakemake.output[0], f=fluxes)
