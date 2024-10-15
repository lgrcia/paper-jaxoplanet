import starry
import numpy as np
from time import time
import theano
import theano.tensor as tt

starry.config.lazy = True


radius = float(snakemake.wildcards.radius)
u = snakemake.params.u
N = int(snakemake.wildcards.N)
assert radius < 1
b = np.linspace(0, 1 + radius, N)
lmax = snakemake.wildcards.lmax
lmax = None if lmax == "None" else int(lmax)

if lmax is None:
    ms = starry.Map(ydeg=len(u), udeg=len(u))
    ms[1:] = u
else:
    l_max = 20
    ms = starry.Map(ydeg=l_max)
    y = np.ones((l_max + 1) ** 2)
    ms[:, :] = y


r_ = tt.dscalar()
b_ = tt.dvector()
xo_ = tt.dscalar()
zo_ = tt.dscalar()
theta_ = tt.dscalar()
lc_func = theano.function([xo_, b_, zo_, r_], ms.flux(xo=xo_, yo=b_, zo=zo_, ro=r_))

times = []

for _ in range(20):
    t0 = time()
    lc_func(0.0, b, 10.0, radius)
    times.append(time() - t0)

np.savez(snakemake.output[0], time=times, N=N, b=b)
