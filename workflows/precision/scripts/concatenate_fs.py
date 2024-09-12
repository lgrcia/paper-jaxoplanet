import pickle
import numpy as np
from tqdm import tqdm
from jaxoplanet.experimental.starry.multiprecision import mp

flux = []
s = []

bs = np.load(snakemake.input.bs, allow_pickle=True)["bs"]
bs = np.abs(np.concatenate(bs))

n = len(snakemake.input.i)
F = mp.zeros(len(bs), n)
S = mp.zeros(len(bs), n)

for i, f in enumerate(snakemake.input.i):
    data = pickle.load(open(f, "rb"))
    F[:, i] = data["f"]

for i in range(len(bs)):
    S[i, :] = data["s"][i].T

with open(snakemake.output[0], "wb") as f:
    pickle.dump(
        {"f": F, "s": S},
        f,
    )
