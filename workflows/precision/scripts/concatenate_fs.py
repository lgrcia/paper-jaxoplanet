import pickle
import numpy as np
from tqdm import tqdm
from jaxoplanet.experimental.starry.mpcore import mp

flux = []
s = []

bs = np.load(snakemake.input[0], allow_pickle=True)["bs"]
bs = np.abs(np.concatenate(bs))

for file in tqdm(snakemake.input.i):
    with open(file, "rb") as f:
        data = np.load(f, allow_pickle=True)
        flux.append(data["f"].tolist())

s_data = np.load(snakemake.input.s, allow_pickle=True)

for b in bs:
    s.append(s_data[b])

s = mp.matrix(np.array(s)[:, :, 0].tolist())
flux = mp.matrix(flux).T

with open(snakemake.output[0], "wb") as f:
    pickle.dump(
        {"f": flux, "s": s},
        f,
    )
