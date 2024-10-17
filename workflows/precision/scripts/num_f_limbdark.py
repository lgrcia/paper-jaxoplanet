import numpy as np
import pickle
from jaxoplanet.starry.multiprecision.utils import mp
from jaxoplanet.starry import basis
from jaxoplanet.starry.multiprecision import solution
from tqdm import tqdm

# some params
r = snakemake.wildcards.r
degree = snakemake.params.degree

# load the fs for each y such as
# y0 = (1, 0, 0, 0, 0, ...)
# y1 = (1, 1, 0, 0, 0, ...)
# y2 = (1, 1, 1, 0, 0, ...)
# etc.
fs = np.load(snakemake.input.fs, allow_pickle=True)["f"].T

fs = mp.matrix(
    [
        fs[0, :].tolist()[0],
        *[(fs[i, :] - fs[0, :]).tolist()[0] for i in range(1, len(fs))],
    ]
)

# load b
bs = np.load(snakemake.input.b, allow_pickle=True)["bs"]
b = np.hstack(bs)

# load A1
num_matrices = np.load(snakemake.input.matrices, allow_pickle=True)
A1inv = num_matrices[degree]["A1inv"]


# the function to compute the flux
def num_flux(u):
    deg = len(u)
    n = (deg + 1) ** 2
    _u = mp.matrix([[1, *u]])
    U = mp.matrix(basis.U(deg))
    pu = _u @ U
    yu = A1inv[0:n, 0:n] @ pu.T
    norm = (pu @ solution.rT(deg))[0]
    yufs = mp.matrix([(yu[i] * fs[i, :]).tolist()[0] for i in range((deg + 1) ** 2)])
    return mp.matrix([sum(yufs[:, i]) for i in range(len(yufs.T))]) / norm


def u(deg):
    u = np.zeros(degree)
    if degree == 0:
        return u
    else:
        u[deg - 1] = 1
        return u


print("running num_flux")
fluxes = mp.matrix([num_flux(u(deg)).T.tolist()[0] for deg in tqdm(range(degree + 1))])
with open(snakemake.output[0], "wb") as f:
    pickle.dump({"f": fluxes}, f)
