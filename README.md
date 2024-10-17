# jaxoplanet paper
This repository contains the code and data for the technical paper of [jaxoplanet](https://github.com/exoplanet-dev/jaxoplanet).

## To reproduce this paper
### Install dependencies

### Testing the workflow
This paper takes couple hours to reproduce from scratch. First, I recommend running the paper for less intense datasets. This can be done opening the `workflows/Snakefile` and setting

```raw
l_max = 2
b_sub_resolution = 2
static_gpu = "data" # this is to avoid running the GPU tests
```

If the next steps run correctly, revert these changes and run the workflow with 

```raw
l_max = 20
b_sub_resolution = 50
```

### Running the workflows

The paper rely on two main workflows that needs to be run separately with snakemake. In the conda environment, run 

```bash
snakemake -c12 precision_figures
snakeamke -c1 speed_figures
```

The `-c1` flag for the speed benchmark ensures that one job is run at a time, which matters as parallel jobs bias the benchmark.

### Building the latex paper

Once ran, figures are available in the `figures` folder and the paper can be compiled using your favorite latex compiler.