import numpy as np

N = int(snakemake.wildcards.N)

r = np.linspace(0, 2, N)
b = np.linspace(0, 2, N)

np.savez(snakemake.output[0], r=r, b=b)
