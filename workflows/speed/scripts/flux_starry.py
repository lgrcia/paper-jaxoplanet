from tqdm import tqdm
import starry
import numpy as np
import timeit


def timeit_f(strf, number=4, repeat=2):
    times = (
        np.array(
            timeit.repeat(f"{strf}", repeat=repeat, number=number, globals=globals())
        )
        / number
    )
    return np.median(times)


starry.config.lazy = False

n = int(snakemake.wildcards.N)

ms = starry.Map(ydeg=1, udeg=1)
u = (1.0,)
ms[1:] = u

r = np.linspace(0, 2, n)
b = np.linspace(0, 2, n)

ro = 0.5

calculated = np.zeros((n, n))

for i, _r in enumerate(tqdm(r)):
    expected = ms.flux(ro=_r, yo=b, xo=0.0, zo=10.0)
    calculated[i, :] = expected


reference_time = timeit_f(f"ms.flux(ro={ro}, yo=b, xo=0.0, zo=10.0)")

np.savez(
    snakemake.output[0], r=r, b=b, value=calculated, ro=ro, u=u, time=reference_time
)
