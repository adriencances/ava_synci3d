import pandas as pd
import tensorboard as tb
import pickle
from pathlib import Path


config_exp_ids = ([
    (8874451247213503487, "PDexVPR3RWCOQjJqWiQwoA"),
    (7989562130473339989, "jkr3BFpPQAGApVzKykYm4w"),
    (8944460950739749069, "ZLqKTPYFQEmPcqZBPzGSNQ"),
    (899180313416084012, "0xwTQ0y9QqyyxAvAvoj07Q"),
    (8744896203684980137, "eR0PIoeNRyq2yoaRVFpzZw"),
    (2913721369671654269, "hEKOf1ZAR3C5U98ybjl9cw"),
    (4260299867352498202, "4Ynem0jPSKOs87VrA6XgWA"),
    (1279444429412012325, "Yuv22UgIQW2Oek5FpajmWA")
])


def get_data_from_experiment(experiment_id):
    experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
    df = experiment.get_scalars()
    return df


def save_data(df, config_id):
    Path("data_runs_friends").mkdir(parents=True, exist_ok=True)
    file_name = "data_runs_friends/data_config_{}".format(config_id)
    with open(file_name, "wb") as f:
        pickle.dump(df, f)


def gather_data_from_runs():
    for config_id, experiment_id in config_exp_ids:
        df = get_data_from_experiment(experiment_id)
        print(config_id, "\t", max(df['step'].tolist()) + 1)
        save_data(df, config_id)


if __name__ == "__main__":
    gather_data_from_runs()
