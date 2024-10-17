import jax

jax.config.update("jax_enable_x64", True)

from jaxoplanet.starry.solution import solution_vector
from jaxoplanet.starry.light_curves import surface_light_curve
from jaxoplanet.starry.surface import Surface
from jaxoplanet.starry.ylm import Ylm
import numpy as np
import jax.numpy as jnp
from functools import partial

r = float(snakemake.wildcards.r)
l_max = snakemake.params.l_max
bs = np.load(snakemake.input[0], allow_pickle=True)["bs"]
bs = np.abs(np.concatenate(bs))
ys = snakemake.params.ys
order = int(snakemake.wildcards.order)


@jax.jit
def jax_flux(y, b):
    surface = Surface(y=Ylm.from_dense(y, normalize=False), normalize=False)
    return surface_light_curve(
        surface, r=r, y=b, z=10.0, order=order, higher_precision=True
    )


jax_s_function = jax.jit(
    jax.vmap(
        jax.jit(solution_vector(l_max, order)),
        in_axes=(0, None),
    )
)

jax_f_function = jax.vmap(jax_flux, in_axes=(None, 0))

jax_f = []
jax_s = []

from tqdm import tqdm

for y in tqdm(ys):
    jax_f.append(jax_f_function(y, bs))

jax_f = jnp.array(jax_f).T
jax_s = jax_s_function(bs, r)
np.savez(snakemake.output[0], f=jax_f, s=jax_s)
