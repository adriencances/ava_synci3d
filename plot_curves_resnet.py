import pandas as pd
import tensorboard as tb
import pickle
from pathlib import Path
import numpy as numpy
import matplotlib.pyplot as plt
import glob


def get_data_for_config(config_id):
    file_name = "data_runs_friends/data_config_resnet_{}".format(config_id)
    df = pickle.load(open(file_name, "rb"))
    return df


def shorten_dictionary(d, max_epoch=8):
    for k in list(d.keys()):
        if k > max_epoch:
            del d[k]


def plot_curves(config_id):
    df = get_data_for_config(config_id)
    dat = df.values

    subdir = "curves_friends/config_resnet_{}".format(config_id)
    Path(subdir).mkdir(parents=True, exist_ok=True)

    nb_epochs = len([1 for i in range(len(dat)) if dat[i, 1] == "training_loss"])
    print(config_id, "\t", nb_epochs)
    for cat in ["training", "validation", "test"]:
        losses = dict([(dat[i, -2], dat[i, -1]) for i in range(len(dat)) if dat[i, 1] == "{}_loss".format(cat)])
        pos_accs = dict([(dat[i, -2], dat[i, -1]) for i in range(len(dat)) if dat[i, 0] == "{}_class_accuracies_positive".format(cat)])
        neg_accs = dict([(dat[i, -2], dat[i, -1]) for i in range(len(dat)) if dat[i, 0] == "{}_class_accuracies_negative".format(cat)])
        mean_accs = dict([(dat[i, -2], dat[i, -1]) for i in range(len(dat)) if dat[i, 1] == "{}_accuracy".format(cat)])
        
        shorten_dictionary(losses)
        shorten_dictionary(pos_accs)
        shorten_dictionary(neg_accs)
        shorten_dictionary(mean_accs)
    
        plt.figure()
        plt.title("{} loss".format(cat.title()))
        plt.hlines(0, min(list(losses.keys())), max(list(losses.keys())), colors="lightgray")
        plt.plot(list(losses.keys()), list(losses.values()))
        plt.savefig("{}/{}_loss.png".format(subdir, cat))
        plt.close()

        plt.figure()
        plt.title("{} class accuracies".format(cat.title()))
        plt.hlines([0, 1], min(list(pos_accs.keys())), max(list(pos_accs.keys())), colors="lightgray")
        plt.plot(list(pos_accs.keys()), list(pos_accs.values()))
        plt.plot(list(neg_accs.keys()), list(neg_accs.values()))
        plt.legend(["positive", "negative"], loc="lower left")
        plt.ylim(-0.1, 1.1)
        plt.savefig("{}/{}_class_accs.png".format(subdir, cat))
        plt.close()

        plt.figure()
        plt.title("{} mean accuracy".format(cat.title()))
        plt.hlines([0, 1], min(list(pos_accs.keys())), max(list(pos_accs.keys())), colors="lightgray")
        plt.plot(list(mean_accs.keys()), list(mean_accs.values()))
        plt.ylim(-0.1, 1.1)
        plt.savefig("{}/{}_mean_accs.png".format(subdir, cat))
        plt.close()


def plot_all_curves():
    config_ids = [int(e.split("_")[-1]) for e in glob.glob("data_runs_friends/data_config_resnet_*")]
    for config_id in config_ids:
        plot_curves(config_id)


if __name__ == "__main__":
    plot_all_curves()
