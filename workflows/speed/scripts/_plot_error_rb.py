import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np

orders = snakemake.params.orders

data = {
    "starry": np.load(snakemake.input["starry"]),
    "jaxoplanet": {
        orders[i]: np.load(snakemake.input["jaxoplanet"][i]) for i in range(len(orders))
    },
}

r = data["jaxoplanet"][orders[0]]["r"]
b = data["jaxoplanet"][orders[0]]["b"]
R, B = np.meshgrid(r, b)


def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt

    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar


fig, axes = plt.subplots(1, len(orders), figsize=(8.5, 2.5))
# norm = colors.LogNorm(vmin=1e-15, vmax=1e-8)
for i, order in enumerate(orders):
    ax = axes[i]
    ax.set_aspect("equal", "box")
    errors = np.abs(data["starry"]["value"] - data["jaxoplanet"][order]["value"])
    errors[errors == 0] = 1e-20
    errors[np.logical_not(np.isfinite(errors))] = 1e-20

    norm = colors.LogNorm(vmin=1e-15, vmax=np.max(np.abs(errors)))
    pcm = ax.pcolor(R, B, errors.T, norm=norm, cmap="PuBu_r", shading="auto")
    k = dict(color="k", linestyle="-", linewidth=1)
    ax.axline((0, 0), slope=1, **k)
    ax.axline((1, 0), slope=1, **k)
    ax.axline((0, 1), slope=1, **k)
    ax.axline((0, 1), slope=-1, **k)
    if i > 0:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel("b")
    ax.set_xlabel("r")
    ax.set_title(rf"$n$ = {order}")
    colorbar(pcm)

# cm_ax = fig.add_axes([1.5, 0.1, 0.02, 0.8])
# place colorbar at cm_ax

plt.tight_layout()
plt.savefig(snakemake.output[0], dpi=250)
