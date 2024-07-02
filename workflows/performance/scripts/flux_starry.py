import starry
import numpy as np
import timeit

starry.config.lazy = False

n = int(snakemake.wildcards.N)

ms = starry.Map(ydeg=1, udeg=1)
u = (1.0,)
ms[1:] = u

r = np.linspace(0, 2, n)
b = np.linspace(0, 2, n)

ro = 0.5

calculated = np.zeros((n, n))

for i, _r in enumerate(r):
    expected = ms.flux(ro=_r, yo=b, xo=0.0, zo=10.0)
    calculated[i, :] = expected


def timeit_f(strf, number=100):
    times = (
        np.array(timeit.repeat(f"{strf}", number=number, globals=globals()))[1:]
        / number
    )
    return np.median(times), times.std()


reference_time = timeit_f(f"ms.flux(ro={ro}, yo=b, xo=0.0, zo=10.0)")[0]

np.savez(
    snakemake.output[0], r=r, b=b, value=calculated, ro=ro, u=u, time=reference_time
)
