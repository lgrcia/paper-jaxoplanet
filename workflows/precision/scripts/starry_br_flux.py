from tqdm import tqdm
import starry
import numpy as np

starry.config.lazy = False

ms = starry.Map(ydeg=1, udeg=1)
u = (1.0,)
ms[1:] = u

b, r = np.load(snakemake.input[0]).values()

calculated = np.zeros((len(r), len(b)))

for i, _r in enumerate(tqdm(r)):
    expected = ms.flux(ro=_r, yo=b, xo=0.0, zo=10.0)
    calculated[i, :] = expected

np.savez(snakemake.output[0], flux=calculated)
