# Experiments with Lightning-UQ-Box


## Installation

We expect a minimum Python version of 3.10. You can set up a conda environment and install the necessary packages inside.

```code
conda create -n myEnvName python=3.10
conda activate myEnvName

pip install git+https://github.com/microsoft/torchgeo.git
pip install git+https://github.com/lightning-uq-box/lightning-uq-box.git
```

We also use WandB for experiment logging.
```code
pip install wandb
```

## Experiments

The experiments for each dataset can be found in their respective directory:

- [Tropical Cyclone](./tropical_cyclone/)
- [Digital Typhoon](./digital_typhoon/)
- [SKIPPD](./skippd/)

The contain a `run_script.py` which will execute and experiment and store evaluated results for all datasplits. Experiments are configured based on yaml files that are stored in a `configs` subdiretory of each dataset experiment directory. We have three "types" of config files:

- `dataset.yaml` configuration specific to the dataset and datamodule
- `trainer.yaml` configuration of experiment naming and Lightning Trainer
- `model_name.yaml` configuration of specific hyperparameters to a model

To run experiments on your machine, you need to make the following changes:

In `dataset.yaml` files, adapt the `root` argument to your preferred local directory. Dataset will automatically be downloaded when running the script for the first time.
```yaml
datamodule:
  root: "Change the root directory to one where data should be downloaded to"
```

In `trainer.yaml` files, 

```yaml
experiment:
  experiment_name: "name of experiment will be included in experiment run name"
  exp_dir: "Directory where to store experiment output"
wandb: # configure wandb here
  project: "name of project"
  entity: "name of user or entity"
  mode: "mode to run wandb in"
```

## Analysis of Experiments

Analysis of experiments was conducted via jupyter notebooks. They expect an experiment directory holding different model runs and will scrape over them to collect all experiment outputs so that they are available for whatevery analysis one might desire.

To run the notebooks with the above environmennt, you also need to install:

```code
pip install ipykernel
pip install ipywidgets
```

