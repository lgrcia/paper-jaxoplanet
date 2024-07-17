import jax
from tqdm import tqdm

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

import numpy as np
from jaxoplanet.experimental.starry import Surface
import jax.numpy as jnp

data = dict(np.load(snakemake.input[0]))
order = int(snakemake.wildcards.order)
u = data["u"]
surface = Surface(period=None, u=(u))

from jaxoplanet.experimental.starry.light_curves import *

import timeit


def timeit_f(strf, n=10):
    times = np.array(timeit.repeat(f"{strf}", number=n, globals=globals()))[1:] / n
    return np.median(times)


# TODO: figure out the sparse matrices (and Pijk) to avoid todense()
def surface_light_curve(
    map,
    r: float = None,
    xo: float = None,
    yo: float = None,
    zo: float = None,
    theta: float = 0.0,
    order: int = 20,
):
    """Light curve of an occulted map.

    Args:
        map (Map): map object
        r (float or None): radius of the occulting body, relative to the current map
           body
        xo (float or None): x position of the occulting body, relative to the current
           map body
        yo (float or None): y position of the occulting body, relative to the current
           map body
        zo (float or None): z position of the occulting body, relative to the current
           map body
        theta (float): rotation angle of the map

    Returns:
        ArrayLike: flux
    """
    rT_deg = rT(map.deg)

    # no occulting body
    if r is None:
        b_rot = True
        theta_z = 0.0
        x = rT_deg

    # occulting body
    else:
        b = jnp.sqrt(jnp.square(xo) + jnp.square(yo))
        b_rot = jnp.logical_or(jnp.greater_equal(b, 1.0 + r), jnp.less_equal(zo, 0.0))
        b_occ = jnp.logical_not(b_rot)
        theta_z = jnp.arctan2(xo, yo)
        sT = solution_vector(map.deg, order=order)(b, r)

        # scipy.sparse.linalg.inv of a sparse matrix[[1]] is a non-sparse [[1]], hence
        # `from_scipy_sparse`` raises an error (case deg=0)
        if map.deg > 0:
            A2 = scipy.sparse.linalg.inv(A2_inv(map.deg))
            A2 = jax.experimental.sparse.BCOO.from_scipy_sparse(A2)
        else:
            A2 = jnp.array([1])

        x = jnp.where(b_occ, sT @ A2, rT_deg)

    # TODO(lgrcia): Is this the right behavior when map.y is None?
    if map.y is None:
        rotated_y = jnp.zeros(map.ydeg)
    else:
        rotated_y = left_project(
            map.ydeg, map.inc, map.obl, theta, theta_z, map.y.todense()
        )

    # limb darkening
    U = jnp.array([1, *map.u])
    A1_val = jax.experimental.sparse.BCOO.from_scipy_sparse(A1(map.ydeg))
    p_y = Pijk.from_dense(A1_val @ rotated_y, degree=map.ydeg)
    p_u = Pijk.from_dense(U @ U0(map.udeg), degree=map.udeg)
    p_y = p_y * p_u

    norm = np.pi / (p_u.tosparse() @ rT(map.udeg))

    return (p_y.tosparse() @ x) * norm


f = jax.jit(
    jax.vmap(surface_light_curve, in_axes=(None, None, None, 0, None, None, None)),
    static_argnames=("order",),
)


ro = data["ro"]
r = data["r"]
b = data["b"]
expected = data["value"]

result = np.zeros_like(expected)


def flux(order):
    for i, _r in enumerate(tqdm(r)):
        calc = f(surface, _r, 0.0, b, 10.0, 0.0, order)
        result[i, :] = calc

    return result


time = timeit_f(f"f(surface, {ro}, 0.0, b, 10.0, 0.0, {order}).block_until_ready()")
calc = flux(order)

np.savez(snakemake.output[0], r=r, b=b, value=calc, ro=ro, u=u, time=time)
