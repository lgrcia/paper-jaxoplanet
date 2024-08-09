from tqdm import tqdm
import starry
import numpy as np
import timeit

radius = snakemake.params.radius
u = snakemake.params.u
N = int(snakemake.wildcards.N)
assert radius < 1
b = np.linspace(0, 1 + radius, N)


def timeit_f(strf, n=10):
    times = np.array(timeit.repeat(f"{strf}", number=n, globals=globals()))[1:] / (
        n - 1
    )
    return times


starry.config.lazy = False

if u is not None:
    ms = starry.Map(ydeg=len(u), udeg=len(u))
    ms[1:] = u
else:
    l_max = 20
    ms = starry.Map(ydeg=l_max)
    y = np.ones((l_max + 1) ** 2)
    ms[:, :] = y


times = timeit_f(f"ms.flux(ro={radius}, yo=b, xo=0.0, zo=10.0)")

np.savez(snakemake.output[0], time=times, N=N, b=b)
