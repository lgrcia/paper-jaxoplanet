import numpy as np
import matplotlib.pyplot as plt
from jaxoplanet.experimental.starry.mpcore import utils, mp
from tqdm import tqdm

rs = snakemake.params.rs
orders = snakemake.params.orders

all_errors_s = []
all_errors_f = []


def errors(M_mp, M_np):
    if isinstance(M_mp, mp.matrix):
        d = utils.diff_mp(M_mp, M_np)
    else:
        d = M_mp - M_np

    rel = np.abs(d)
    frac = np.abs(
        d
        / max(1e-9, np.nanmax(np.min([np.abs(utils.to_numpy(M_mp)), np.abs(M_np)], 0)))
    )
    frac[rel < 1e-16] = 1e-16
    return frac


for i in tqdm(range(len(rs))):
    num_data = np.load(snakemake.input.num[i], allow_pickle=True)
    all_errors_f.append(
        [
            np.abs(
                errors(
                    num_data["f"],
                    np.load(snakemake.input[f"jax_{order}"][i], allow_pickle=True)["f"],
                )
            )
            for order in orders
        ]
    )
    all_errors_s.append(
        [
            np.abs(
                errors(
                    num_data["s"],
                    np.load(snakemake.input[f"jax_{order}"][i], allow_pickle=True)["s"],
                )
            )
            for order in orders
        ]
    )

np.savez(
    snakemake.output[0],
    errors_s=all_errors_s,
    errors_f=all_errors_f,
    orders=orders,
    rs=rs,
)
