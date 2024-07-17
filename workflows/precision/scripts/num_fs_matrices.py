l_max = snakemake.params.l_max

from jaxoplanet.experimental.starry.mpcore import mp
from jaxoplanet.experimental.starry.mpcore.basis import A1, A2_inv
from jaxoplanet.experimental.starry.mpcore.rotation import R


inc = mp.pi / 2
obl = 0.0

_A1 = A1(l_max)
_A2 = A2_inv(l_max) ** -1

R_px = R(l_max, (1.0, 0.0, 0.0), -0.5 * mp.pi)
R_mx = R(l_max, (1.0, 0.0, 0.0), 0.5 * mp.pi)
R_obl = R(l_max, (0.0, 0.0, 1.0), -obl)
R_inc = R(l_max, (-mp.cos(obl), -mp.sin(obl), 0.0), (0.5 * mp.pi - inc))

import numpy as np
import pickle

with open(snakemake.output[0], "wb") as f:
    pickle.dump(
        {
            "A1": _A1,
            "A2": _A2,
            "R_px": R_px,
            "R_mx": R_mx,
            "R_obl": R_obl,
            "R_inc": R_inc,
        },
        f,
    )
