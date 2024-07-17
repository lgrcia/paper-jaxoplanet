import numpy as np
import starry

starry.config.lazy = False


r = float(snakemake.wildcards.r)
bs = np.load(snakemake.input[0], allow_pickle=True)["bs"]
bs = np.abs(np.concatenate(bs))
l_max = snakemake.params.l_max
ys = snakemake.params.ys


map = starry.Map(ydeg=l_max)


S = []

for b in bs:
    try:
        val = map.ops.sT([b], [r])[0]
    except:
        val = np.nan * np.ones(map.Ny)

    S.append(val)

S = np.array(S)


flux = []
from tqdm import tqdm

for y in tqdm(ys):

    map[:, :] = y
    f = map.flux(xo=0.0, yo=bs, zo=10.0, ro=r)
    flux.append(f)

flux = np.array(flux).T

np.savez(snakemake.output[0], f=flux, s=S)
