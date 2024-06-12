# SKy Images and Photovoltaic Power Generation Dataset (SKIPP'D) Experiments

## Download data

Data will automatically be downloaded when running the experiment script below for the first time.

## Experiments

Regression Experiment:
```code
python run_script.py model_config=configs/dkl.yaml data_config=configs/dataset.yaml trainer_config=configs/trainer.yaml experiment.seed=10 trainer.devices=[0]
```