import numpy as np

r = float(snakemake.wildcards.r)
l_max = snakemake.params.l_max
bs = np.load(snakemake.input[0], allow_pickle=True)["bs"]
bs = np.abs(np.concatenate(bs))
y = snakemake.params.y

num_matrices = np.load(snakemake.input[1], allow_pickle=True)
ST = np.load(snakemake.input[2], allow_pickle=True)

from jaxoplanet.experimental.starry.mpcore import mp
from jaxoplanet.experimental.starry.mpcore.rotation import (
    dot_rotation_matrix,
    dot_rz,
)

y = mp.matrix(y.tolist())


def flux_function(l_max, inc, obl):
    _A1 = num_matrices["A1"]
    _A2 = num_matrices["A2"]

    R_px = num_matrices["R_px"]
    R_mx = num_matrices["R_mx"]
    R_obl = num_matrices["R_obl"]
    R_inc = num_matrices["R_inc"]

    def rotate_y(y, phi):
        y_rotated = dot_rotation_matrix(l_max, None, None, R_px)(y)
        y_rotated = dot_rz(l_max, phi)(y_rotated)
        y_rotated = dot_rotation_matrix(l_max, None, None, R_mx)(y_rotated)
        y_rotated = dot_rotation_matrix(l_max, None, None, R_obl)(y_rotated)
        y_rotated = dot_rotation_matrix(l_max, None, None, R_inc)(y_rotated)
        return y_rotated

    def occ_flux(y, b, r, phi):
        xo = 0.0
        yo = b
        theta_z = mp.atan2(xo, yo)
        _sT = ST[b]
        x = _sT.T @ _A2
        y_rotated = rotate_y(y, phi)
        y_rotated = dot_rz(l_max, theta_z)(y_rotated)
        py = (_A1 @ y_rotated).T
        return (py @ x.T)[0], _sT

    def impl(y, b, r, phi):
        if abs(b) >= (r + 1):
            raise ValueError("Impact parameter out of bounds")
        else:
            if r > 1 and abs(b) <= (r - 1):
                raise ValueError("Impact parameter out of bounds")
            else:
                return occ_flux(y, b, r, phi)

    return impl


def is_occ(b, r):
    return np.logical_and(abs(b) < (r + 1), abs(b) > (r - 1))


if not np.all(is_occ(bs, r)):
    raise ValueError("Impact parameter out of bounds")

func = flux_function(l_max, mp.pi, 0.0)

from tqdm import tqdm

results = []

for b in tqdm(bs):
    results.append(func(y, b, r, 0.0))

f = [r[0] for r in results]

np.savez(snakemake.output[0], f=f)
