import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

from jaxoplanet.experimental.starry.solution import solution_vector
from jaxoplanet.experimental.starry.light_curves import *
import numpy as np
import jax.numpy as jnp
from jaxoplanet.experimental.starry.solution import rT
from jaxoplanet.experimental.starry.mpcore.basis import A1
from jaxoplanet.experimental.starry.basis import A2_inv
from jaxoplanet.experimental.starry.mpcore.utils import to_numpy


def jax_flux(deg, order=20, inc=np.pi / 2, obl=0.0):

    def impl(y, b, r, theta):
        return surface_light_curve(
            deg, y, inc, obl, r, 0.0, b, 10.0, theta=theta, order=order
        )

    return jax.jit(impl)


data = np.load(snakemake.input[0])
b = np.hstack(data["bs"])
orders = data["orders"]
ys = snakemake.params.ys
r = float(snakemake.wildcards.r)
l_max = snakemake.params.l_max

results = {}


for order in orders:
    jax_s_function = jax.jit(
        jax.vmap(jax.jit(solution_vector(l_max, order)), in_axes=(0, None))
    )

    jax_f_function = jax.jit(
        jax.vmap(jax.jit(jax_flux(l_max, order=order)), in_axes=(None, 0, None, None))
    )

    jax_f = []
    jax_s = []

    from tqdm import tqdm

    for y in tqdm(ys):
        jax_f.append(jax_f_function(y, b, r, 0.0))

    results.update(
        {f"f_{order}": jnp.array(jax_f).T, f"s_{order}": jax_s_function(b, r)}
    )

np.savez(snakemake.output[0], **results)
