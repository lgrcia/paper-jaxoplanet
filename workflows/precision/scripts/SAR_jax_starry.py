import jax

jax.config.update("jax_enable_x64", True)


import starry
import numpy as np

starry.config.lazy = False

l_max = snakemake.params.l_max

# R

u = (1.0, 0.0, 0.0)
theta = 0.1


from jaxoplanet.experimental.starry.mpcore.rotation import R

num_R = R(l_max, u, theta)


def R_starry(lmax, u, theta):
    shapes = np.array([2 * l + 1 for l in range(l_max + 1)])

    R = []
    ms = starry.Map(lmax)
    y0 = np.zeros((l_max + 1) ** 2)

    for i in range(len(y0)):
        y = y0.copy()
        y[i] = 1
        R.append(ms.ops.dotR(y[None, :], *u, theta)[0])

    R = np.array(R)

    R_blocks = []

    n = 0
    for j in range(0, len(shapes)):
        i = shapes[j]
        a, b = n, n + i
        n += i
        R_blocks.append(R[a:b][:, a:b])

    return R_blocks


sta_R = R_starry(l_max, u, theta)

from jaxoplanet.experimental.starry.s2fft_rotation import compute_rotation_matrices

jax_R = compute_rotation_matrices(l_max, *u, theta)

# A
from jaxoplanet.experimental.starry.basis import A1, A2_inv
import jax.numpy as jnp


jax_A1 = A1(l_max).todense()
jax_A2 = jnp.linalg.inv(A2_inv(l_max).todense())
jax_A = jax_A2 @ jax_A1

ms = starry.Map(l_max)
sta_A1 = ms.ops.A1.eval().todense()
sta_A = ms.ops.A.eval().todense()

# S

from jaxoplanet.experimental.starry.solution import solution_vector
from jaxoplanet.experimental.starry.mpcore.solution import sT

r = 1.0
b = 1.0

num_sT = sT(l_max, b, r)
jax_sT = solution_vector(l_max, 500)(b, r)
sta_sT = ms.ops.sT([b], [r])[0]

import pickle

results = {
    "u": u,
    "theta": theta,
    "sta_R": sta_R,
    "jax_R": jax_R,
    "num_R": num_R,
    "b": b,
    "r": r,
    "sta_A": sta_A,
    "jax_A": jax_A,
    "sta_sT": sta_sT,
    "jax_sT": jax_sT,
    "num_sT": num_sT,
}

with open(snakemake.output[0], "wb") as f:
    pickle.dump(
        results,
        f,
    )
