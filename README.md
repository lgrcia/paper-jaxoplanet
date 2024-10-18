# jaxoplanet paper
This repository contains the code and data for the technical paper of [jaxoplanet](https://github.com/exoplanet-dev/jaxoplanet).

> [!IMPORTANT] 
> If you run into errors while trying to reproduce the paper, please do not hesitate to open an issue!

## To reproduce this paper
### Install dependencies

Create a new conda environment 

```bash
conda create -n jaxoplanet-paper python=3.10 clangxx
conda activate jaxoplanet-paper
```
> [!NOTE] 
> Python 3.10 and `clangxx` are necessary to compile and use the `starry` package employed in our comparisons.

Then clone the repository and navigate to the cloned folder

```bash
git clone -b clean_workflow https://github.com/lgrcia/paper-jaxoplanet.git
cd paper-jaxoplanet
```

you can now install the dependencies with

```bash
pip install "jaxoplanet[test,test-math,comparison] @ git+https://github.com/exoplanet-dev/jaxoplanet.git@feat-starry-out-of-experimental"
pip install -r workflows/requirements.txt
```



### Testing the workflow
This paper takes couple hours to reproduce from scratch. First, I recommend running the paper for less intense datasets. This can be done by editing `workflows/Snakefile` to

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

The paper relies on two main workflows that needs to be run separately with snakemake. In the conda environment, run 

```bash
cd workflows
snakemake -c12 precision_figures
```

and then 

```bash
snakemake -c1 speed_figures
```

The `-c1` flag for the speed benchmark ensures that one job is run at a time, which matters as parallel jobs bias the benchmark.

### Building the latex paper

Once ran, figures are available in the `figures` folder and the paper can be compiled using your favorite latex compiler.


### GPU benchmarks

For now the GPU data must be produces separately on a different hardware (and modifying some scripts to use GPUS). This part has yet to be documented...