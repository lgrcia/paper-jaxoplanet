from jaxoplanet.starry.multiprecision.solution import sT
import numpy as np
from tqdm import tqdm

l_max = snakemake.params.l_max
r = float(snakemake.wildcards.r)
bs = np.load(snakemake.input[0], allow_pickle=True)["bs"]
bs = np.abs(np.concatenate(bs))

result = {b: sT(l_max, b, r) for b in tqdm(bs)}

with open(snakemake.output[0], "wb") as f:
    pickle.dump(
        result,
        f,
    )
