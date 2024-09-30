import jax

jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

from exoplanet.light_curves import LimbDarkLightCurve
from jaxoplanet.core.limb_dark import light_curve
from jaxoplanet.experimental.starry.light_curves import surface_light_curve
from jaxoplanet.experimental.starry import Surface, Ylm
import numpy as np
import starry
from tqdm import tqdm
import pickle

starry.config.lazy = False

radius = snakemake.params.radius
u = snakemake.params.u
b = 1 - radius
orders = snakemake.params.orders
y_20 = np.ones(21**2)

from jaxoplanet.experimental.starry.multiprecision.flux import flux_function
from jaxoplanet.experimental.starry.multiprecision import mp, utils
from jaxoplanet.experimental.starry.basis import U

f_q = flux_function(3**2, mp.pi / 2, 0.0)
y_q = mp.matrix([[1.0, 0.1, 0.1]]) @ utils.to_mp(U(2))
f_num = {"quadratic": f_q(y_q, b, radius, 0.0)}

f_20 = flux_function(20, mp.pi / 2, 0.0)

f_exoplanet = (
    LimbDarkLightCurve(*u)._compute_light_curve(b, np.ones_like(b) * radius).eval()
    + 1.0
)

f_starry = {}
ms = starry.Map(ydeg=len(u), udeg=len(u))
ms[1:] = u

f_starry["quadratic"] = ms.flux(xo=0.0, yo=b, zo=10.0, ro=radius) - 1
ms = starry.Map(ydeg=20)
ms[:, :] = y_20
f_starry["20"] = ms.flux(xo=0.0, yo=b, zo=10.0, ro=radius)

surface_20 = Surface(period=None, y=Ylm.from_dense(y_20))

f_jaxoplanet = {
    "quadratic": {
        order: light_curve(u, b, radius, order=order) for order in tqdm(orders)
    },
    "20": {
        order: surface_light_curve(surface_20, r=radius, y=b, z=10.0, order=order)
        for order in tqdm(orders)
    },
}

f_num["20"] = f_20(mp.matrix(y_20.tolist()), b, radius, 0.0)


pickle.dump(
    {
        "starry": f_starry,
        "exoplanet": f_exoplanet,
        "jaxoplanet": f_jaxoplanet,
        "multiprecision": f_num,
    },
    open(snakemake.output[0], "wb"),
)
