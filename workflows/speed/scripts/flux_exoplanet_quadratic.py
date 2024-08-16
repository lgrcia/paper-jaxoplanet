from exoplanet.light_curves import LimbDarkLightCurve
import numpy as np
from time import time

radius = snakemake.params.radius
u = snakemake.params.u
N = int(snakemake.wildcards.N)
assert radius < 1
b = np.linspace(0, 1 + radius, N)

import theano
import theano.tensor as tt

r_ = tt.dvector()
b_ = tt.dvector()
u1_ = tt.dscalar()
u2_ = tt.dscalar()
lc = LimbDarkLightCurve(u1_, u2_)._compute_light_curve(b_, r_)
lc_func = theano.function([u1_, u2_, b_, r_], lc)  # compiling happens here
radii = np.ones_like(b) * radius


times = []

for _ in range(20):
    t0 = time()
    lc_func(*u, b, radii)
    times.append(time() - t0)

np.savez(snakemake.output[0], time=times, N=N, b=b)
