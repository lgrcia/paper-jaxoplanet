import numpy as np

r = float(snakemake.wildcards.r)

if r > 1:
    b = np.linspace(r - 1 - 1e-6, r, 10)
else:
    b = np.linspace(1e-6, 1, 10)

orders = np.array([20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]).astype(int)

np.savez(snakemake.output[0], bs=[b], orders=orders)
