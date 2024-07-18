import starry

starry_version = starry.__version__

with open(snakemake.output[0], "w") as f:
    f.write(starry_version)
