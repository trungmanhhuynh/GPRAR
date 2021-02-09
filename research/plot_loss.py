'''
generate_inputs.py : plot train and vidation loss from log files.

Author: Manh Huynh
Last Update: 07/17/2020
'''

import json
import matplotlib.pyplot as plt
import numpy as np

from scipy import ndimage


def plot_multiple_loss():
    """
        plot train/validation loss
        save image to current directory.
    """

    # log file is generarted by train a model,
    # specify LOG_FILE correctly - must be json file
    LOGFILE_1 = "/home/manhh/work_dir/reconstruction/jaad_rand_init/loss.json"
    LOGFILE_2 = "/home/manhh/work_dir/reconstruction/jaad/loss.json"

    LOG_FIGURE = "jaad_reconstruction_loss.png"

    with open(LOGFILE_1, "r") as f:
        data_1 = json.load(f)
    with open(LOGFILE_2, "r") as f:
        data_2 = json.load(f)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    epoch = range(0,len(data_1['loss_rec']))
    ax.plot(epoch, data_1['loss_rec'], '-b', label='random weights', linewidth=4)
    ax.plot(epoch, data_2['loss_rec'], '-r', label='pre-trained weights', linewidth=4)

    ax.set_ylim([0, 0.04])
    # ax.plot(log_data['epoch'], log_data['val_loss'], 'b', label='validation')
    plt.title("JAAD", fontsize=20)
    plt.xlabel("#epoch",fontsize= 20)
    plt.ylabel("loss", fontsize= 20)
    ax.legend(prop={'size': 18})
    plt.xticks(np.arange(0, 60, step=10))
    plt.yticks([0, 0.01, 0.02, 0.03, 0.04])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    fig.savefig(LOG_FIGURE)
    plt.close()

if __name__ == '__main__':

    plot_multiple_loss()
