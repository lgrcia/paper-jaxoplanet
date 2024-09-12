l_max = snakemake.params.l_max

from jaxoplanet.experimental.starry.multiprecision import mp
from jaxoplanet.experimental.starry.multiprecision.basis import A1, A2
from jaxoplanet.experimental.starry.multiprecision.rotation import get_R
from collections import defaultdict

inc = mp.pi / 2
obl = 0.0

matrices = defaultdict(
    lambda: {
        "R_obl": {},
        "R_inc": {},
        "sT": {},
    }
)


_ = A1(l_max, cache=matrices)
_ = A2(l_max, cache=matrices)
_ = get_R("R_px", l_max, cache=matrices)
_ = get_R("R_mx", l_max, cache=matrices)
_ = get_R("R_inc", l_max, inc=inc, obl=obl, cache=matrices)
_ = get_R("R_obl", l_max, inc=inc, obl=obl, cache=matrices)

import pickle

with open(snakemake.output[0], "wb") as f:
    pickle.dump(
        dict(matrices),
        f,
    )
