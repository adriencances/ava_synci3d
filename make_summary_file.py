import pandas as pd
import tensorboard as tb
import pickle
from pathlib import Path
import numpy as numpy
import matplotlib.pyplot as plt
import glob


summary_file = "summary_runs_friends_synci3d.csv"


config_ids = ([
    "8874451247213503487",
    "2913721369671654269",
    "7989562130473339989",
    "899180313416084012",
    "8744896203684980137",
    "8944460950739749069",
    "resnet_8894521375383121754",
    "resnet_8698596633345978445"
])


def get_data_for_config(config_id):
    file_name = "data_runs_friends/data_config_{}".format(config_id)
    df = pickle.load(open(file_name, "rb"))
    return df


def shorten_dictionary(d, max_epoch=8):
    for k in list(d.keys()):
        if k > max_epoch:
            del d[k]


def gather_data(config_id):
    df = get_data_for_config(config_id)
    dat = df.values

    categories = ["training", "validation", "test"]

    loss = {}
    pos_accuracy = {}
    neg_accuracy = {}
    mean_accuracy = {}
    for cat in categories:
        losses = dict([(dat[i, -2], dat[i, -1]) for i in range(len(dat)) if dat[i, 1] == "{}_loss".format(cat)])
        pos_accs = dict([(dat[i, -2], dat[i, -1]) for i in range(len(dat)) if dat[i, 0] == "{}_class_accuracies_positive".format(cat)])
        neg_accs = dict([(dat[i, -2], dat[i, -1]) for i in range(len(dat)) if dat[i, 0] == "{}_class_accuracies_negative".format(cat)])
        mean_accs = dict([(dat[i, -2], dat[i, -1]) for i in range(len(dat)) if dat[i, 1] == "{}_accuracy".format(cat)])

        shorten_dictionary(losses)
        shorten_dictionary(pos_accs)
        shorten_dictionary(neg_accs)
        shorten_dictionary(mean_accs)

        relevant_epoch = 8 if cat == "training" else max(mean_accs, key=lambda k: mean_accs[k])
        
        loss[cat] = round(losses[relevant_epoch], 3)
        pos_accuracy[cat] = round(pos_accs[relevant_epoch] * 100, 1)
        neg_accuracy[cat] = round(neg_accs[relevant_epoch] * 100, 1)
        mean_accuracy[cat] = round(mean_accs[relevant_epoch] * 100, 1)

    return loss, mean_accuracy, pos_accuracy, neg_accuracy


def make_summary_file():
    datas = []
    categories = ["training", "validation", "test"]
    for config_id in config_ids:
        datas.append((config_id, gather_data(config_id)))
    with open(summary_file, "w") as f:
        names = (["config id",
                    "train loss", "train mean acc", "train pos acc", "train neg acc",
                    "val loss", "val mean acc", "val pos acc", "val neg acc",
                    "test loss", "test mean acc", "test pos acc", "test neg acc"])
        f.write(",".join(names) + "\n")
        for config_id, data in datas:
            entries = [config_id] + [values[cat] for cat in categories for values in data]
            f.write(",".join(map(str, entries)) + "\n")


if __name__ == "__main__":
    make_summary_file()
