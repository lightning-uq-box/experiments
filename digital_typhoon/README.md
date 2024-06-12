# Digitial Typhoon Dataset Experiments

## Download Data

To download data exectute the [download.py](./data/download.py) file. And then adapt the root path in the [dataset.yaml](./configs/classification/dataset.yaml) and [dataset.yaml](./configs/regression/dataset.yaml).

## Experiments

Classification Experiment:
```code
python run_script.py model_config=configs/classification/mc_dropout.yaml data_config=configs/classification/dataset.yaml trainer_config=configs/classification/trainer.yaml experiment.seed=10 trainer.devices=[0]
```

Regression Experiment:
```code
python run_script.py model_config=configs/regression/bnn_elbo.yaml data_config=configs/regression/dataset.yaml trainer_config=configs/regression/trainer.yaml experiment.seed=10 trainer.devices=[0]
```