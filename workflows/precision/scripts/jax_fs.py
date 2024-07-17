import jax

jax.config.update("jax_enable_x64", True)

from jaxoplanet.experimental.starry.solution import solution_vector
from jaxoplanet.experimental.starry.light_curves import *
import numpy as np
import jax.numpy as jnp
from jaxoplanet.experimental.starry.mpcore.solution import rT
from jaxoplanet.experimental.starry.mpcore.basis import A1, A2_inv
from jaxoplanet.experimental.starry.mpcore.utils import to_numpy


def surface_light_curve(
    deg,
    y,
    inc,
    obl,
    r: float = None,
    xo: float = None,
    yo: float = None,
    zo: float = None,
    theta: float = 0.0,
    order: int = 20,
):
    rT_deg = to_numpy(rT(deg))

    # occulting body
    if True:
        b = jnp.sqrt(jnp.square(xo) + jnp.square(yo))
        b_rot = jnp.logical_or(jnp.greater_equal(b, 1.0 + r), jnp.less_equal(zo, 0.0))
        b_occ = jnp.logical_not(b_rot)
        theta_z = jnp.arctan2(xo, yo)
        sT = solution_vector(deg, order=order)(b, r)

        if deg > 0:
            A2 = to_numpy(A2_inv(deg) ** -1)
        else:
            A2 = jnp.array([1])

        x = jnp.where(b_occ, sT @ A2, rT_deg)

    rotated_y = left_project(deg, inc, obl, theta, theta_z, y)

    A1_val = to_numpy(A1(deg))
    p_y = Pijk.from_dense(A1_val @ rotated_y, degree=deg)

    b_full_occ = jnp.logical_and(r >= 1.0, b <= r - 1)
    norm = jnp.where(b_full_occ, 0.0, 1.0)

    return p_y.tosparse() @ x * norm


def jax_flux(deg, order=20, inc=np.pi / 2, obl=0.0):

    def impl(y, b, r, theta):
        return surface_light_curve(
            deg, y, inc, obl, r, 0.0, b, 10.0, theta=theta, order=order
        )

    return jax.jit(impl)


r = float(snakemake.wildcards.r)
l_max = snakemake.params.l_max
bs = np.load(snakemake.input[0], allow_pickle=True)["bs"]
bs = np.abs(np.concatenate(bs))
ys = snakemake.params.ys

jax_s_function = jax.jit(
    jax.vmap(jax.jit(solution_vector(l_max, 500)), in_axes=(0, None))
)

jax_f_function = jax.jit(
    jax.vmap(jax.jit(jax_flux(l_max, order=500)), in_axes=(None, 0, None, None))
)


jax_f = []
jax_s = []

from tqdm import tqdm

for y in tqdm(ys):
    jax_f.append(jax_f_function(y, bs, r, 0.0))

jax_f = jnp.array(jax_f).T
jax_s = jax_s_function(bs, r)
np.savez(snakemake.output[0], f=jax_f, s=jax_s)
