import numpy as np
import starry

starry.config.lazy = False


r = float(snakemake.wildcards.r)
bs = np.load(snakemake.input[0], allow_pickle=True)["bs"]
bs = np.abs(np.concatenate(bs))
degree = snakemake.params.degree


map = starry.Map(ydeg=degree, udeg=degree)

flux = []
from tqdm import tqdm


def u(deg):
    u = np.zeros(degree)
    if degree == 0:
        return u
    else:
        u[deg - 1] = 1
        return u


for deg in tqdm(range(degree + 1)):
    map[1:] = u(deg)
    f = map.flux(xo=0.0, yo=bs, zo=10.0, ro=r)
    flux.append(f)

flux = np.array(flux)

np.savez(snakemake.output[0], f=flux)
