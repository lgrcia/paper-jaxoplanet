import numpy as np


def b_range(r, logdelta=-3, logeps=-6, res=2):
    delta = 10**logdelta
    eps = 10**logeps
    if r > 1:
        bs = [
            np.linspace(r - 1 + 1e-8, r - 1 + eps, res),
            np.linspace(r - 1 + eps, r - 1 + delta, res),
            np.linspace(r - 1 + delta, r - delta, 3 * res),
            np.linspace(r - delta, r - eps, res),
            np.linspace(r - eps, r + eps, res),
            np.linspace(r + eps, r + delta, res),
            np.linspace(r + delta, r + 1 - delta, 3 * res),
            np.linspace(r + 1 - delta, r + 1 - eps, res),
            np.linspace(r + 1 - eps, r + 1 - 1e-8, res),
        ]
        labels = [
            r"$r - 1$",
            r"$r - 1 + 10^{%d}$" % logeps,
            r"$r - 1 + 10^{%d}$" % logdelta,
            r"$r - 10^{%d}$" % logdelta,
            r"$r - 10^{%d}$" % logeps,
            r"$r + 10^{%d}$" % logeps,
            r"$r + 10^{%d}$" % logdelta,
            r"$r + 1 - 10^{%d}$" % logdelta,
            r"$r + 1 - 10^{%d}$" % logeps,
            r"$r + 1$",
        ]
    else:
        bs = [
            np.linspace(1e-8, eps, res),
            np.linspace(eps, delta, res),
            np.linspace(delta, r - delta, 3 * res),
            np.linspace(r - delta, r - eps, res),
            np.linspace(r - eps, r + eps, res),
            np.linspace(r + eps, r + delta, res),
            np.linspace(r + delta, 1 - r - delta, 3 * res),
            np.linspace(1 - r - delta, 1 - r - eps, res),
            np.linspace(1 - r - eps, 1 - r + eps, res),
            np.linspace(1 - r + eps, 1 - r + delta, res),
            np.linspace(1 - r + delta, 1 - delta, 3 * res),
            np.linspace(1 - delta, 1 - eps, res),
            np.linspace(1 - eps, 1 + eps, res),
            np.linspace(1 + eps, 1 + delta, res),
            np.linspace(1 + delta, r + 1 - delta, 3 * res),
            np.linspace(r + 1 - delta, r + 1 - eps, res),
            np.linspace(r + 1 - eps, r + 1 - 1e-8, res),
        ]
        labels = [
            r"$0$",
            r"$10^{%d}$" % logeps,
            r"$10^{%d}$" % logdelta,
            r"$r - 10^{%d}$" % logdelta,
            r"$r - 10^{%d}$" % logeps,
            r"$r + 10^{%d}$" % logeps,
            r"$r + 10^{%d}$" % logdelta,
            r"$1 - r - 10^{%d}$" % logdelta,
            r"$1 - r - 10^{%d}$" % logeps,
            r"$1 - r + 10^{%d}$" % logeps,
            r"$1 - r + 10^{%d}$" % logdelta,
            r"$1 - 10^{%d}$" % logdelta,
            r"$1 - 10^{%d}$" % logeps,
            r"$1 + 10^{%d}$" % logeps,
            r"$1 + 10^{%d}$" % logdelta,
            r"$r + 1 - 10^{%d}$" % logdelta,
            r"$r + 1 - 10^{%d}$" % logeps,
            r"$r + 1$",
        ]
    return bs, labels


r = float(snakemake.wildcards.r)

bs, labels = b_range(r)

np.savez(
    snakemake.output[0],
    bs=bs,
    labels=labels,
    r=r,
)
