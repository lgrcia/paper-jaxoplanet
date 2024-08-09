from exoplanet.light_curves import LimbDarkLightCurve
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


lc_object = LimbDarkLightCurve(*u)
radii = np.ones_like(b) * radius
times = timeit_f(f"lc_object._compute_light_curve(b, radii)")

np.savez(snakemake.output[0], time=times, N=N, b=b)
