import numpy as np
from jaxoplanet.starry.multiprecision import mp
from jaxoplanet.starry.multiprecision.flux import flux_function

r = float(snakemake.wildcards.r)
l_max = snakemake.params.l_max
bs = np.load(snakemake.input[0], allow_pickle=True)["bs"]
bs = np.abs(np.concatenate(bs))
y = snakemake.params.y

num_matrices = np.load(snakemake.input[1], allow_pickle=True)

ST = np.load(snakemake.input[2], allow_pickle=True)

for b in bs:
    num_matrices[l_max]["sT"][(b, r)] = ST[b]

y = mp.matrix(y.tolist())


def is_occ(b, r):
    return np.logical_and(abs(b) < (r + 1), abs(b) > (r - 1))


if not np.all(is_occ(bs, r)):
    raise ValueError("Impact parameter out of bounds")

func = flux_function(l_max, mp.pi / 2, 0.0, cache=num_matrices)

from tqdm import tqdm

results = []
sT = []

for b in tqdm(bs):
    results.append(func(y, b, r, 0.0))
    sT.append(num_matrices[l_max]["sT"][(b, r)])

import pickle

pickle.dump(
    {"f": mp.matrix(results), "s": sT},
    open(snakemake.output[0], "wb"),
)
