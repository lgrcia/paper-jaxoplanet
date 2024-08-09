from exoplanet.light_curves import LimbDarkLightCurve
import numpy as np
from tqdm import tqdm

r = float(snakemake.wildcards.r)
degree = snakemake.params.degree
bs = np.load(snakemake.input[0], allow_pickle=True)["bs"]
b = np.abs(np.concatenate(bs))


def flux(u):
    return (
        LimbDarkLightCurve(*u)._compute_light_curve(b, np.ones_like(b) * r).eval()
    ) + 1.0


fluxes = np.array([flux(u) for u in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]])
np.savez(snakemake.output[0], f=fluxes)
