import pandas as pd
import tensorboard as tb
import pickle
from pathlib import Path


config_exp_ids = ([
    (8894521375383121754, "GWj9ZFB3TXKC8K2AgrC3fQ"),
    (8698596633345978445, "6qRn9r0CQ9OMwQwVaQ0How")
])


def get_data_from_experiment(experiment_id):
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()
    return df


def save_data(df, config_id):
    Path("data_runs_friends").mkdir(parents=True, exist_ok=True)
    file_name = "data_runs_friends/data_config_resnet_{}".format(config_id)
    with open(file_name, "wb") as f:
        pickle.dump(df, f)


def gather_data_from_runs():
    for config_id, experiment_id in config_exp_ids:
        df = get_data_from_experiment(experiment_id)
        print(config_id, "\t", max(df['step'].tolist()) + 1)
        save_data(df, config_id)


if __name__ == "__main__":
    gather_data_from_runs()
