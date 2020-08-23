'''
generate_inputs.py : plot train and vidation loss from log files.

Author: Manh Huynh
Last Update: 07/17/2020
'''

import json
import matplotlib.pyplot as plt
import numpy as np


def plot_multiple_loss():
    """
        plot train/validation loss
        save image to current directory.
    """

    # log file is generarted by train a model,
    # specify LOG_FILE correctly - must be json file
    #TCNN_LOG_FILE = "save/tcnn/small_size/log/log.json"
    LOGFILE_1 = "save/model2/log/log.json"
    LOGFILE_2 = "save/model4/prediction/log/log.json"

    LOG_FIGURE = "train_loss.png"

    # with open(TCNN_LOG_FILE, "r") as f:
    #     tcnn_data = json.load(f)

    with open(LOGFILE_1, "r") as f:
        data_1 = json.load(f)

    with open(LOGFILE_2, "r") as f:
        data_2 = json.load(f)

    fig, ax = plt.subplots(nrows=1, ncols=1)

    #ax.plot(tcnn_data['epoch'], tcnn_data['train_loss'], 'r',  label='tcnn')
    ax.plot(data_1['epoch'], 5 + np.log(data_1['train_loss']), 'b', label='model_2')
    ax.plot(data_2['epoch'], 5 + np.log(data_2['train_loss']), 'k', label='model_3')

    # ax.plot(log_data['epoch'], log_data['val_loss'], 'b', label='validation')
    plt.title("train loss")
    plt.xlabel("#epoch")
    plt.ylabel("loss")
    ax.legend()

    fig.savefig(LOG_FIGURE)
    plt.close()


def plot_loss():
    """
        plot train/validation loss
        save image to current directory.
    """

    # log file is generarted by train a model,
    # specify LOG_FILE correctly - must be json file
    LOG_FILE = "save/tcnn/small_size/log/log.json"
    LOG_FIGURE = "train_loss.png"

    with open(LOG_FILE, "r") as f:
        log_data = json.load(f)

    fig, ax = plt.subplots(nrows=1, ncols=1)

    ax.plot(log_data['epoch'], log(log_data['train_loss']), 'r', label='train')
    #ax.plot(log_data['epoch'], log_data['val_loss'], 'b', label='validation')
    plt.xlabel("#epoch")
    plt.ylabel("loss")
    ax.legend()

    fig.savefig(LOG_FIGURE)
    plt.close()


def plot_1():

    # occlusion_rate = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]
    # x = range(len(occlusion_rate))
    # LSTM  = [34.48,  34.48 ,  34.48 , 34.48 , 34.48 ,  34.48 ,  34.48, \
    #          34.48,   34.48,   34.48,   71.08,   173.63,  261.34,  401.98,  401.98]
    # TCNN_POSE =[15.18,   25.7,    63.56,   101.38,  686.48,  450.77,  453.07,  472.72,  698.32,  \
    #             698.32, 680.62,  715.53,  826.58,  826.58,  826.58]

    occlusion_rate = [0, 1, 3, 5, 7, 10, 20, 40, 50]
    LSTM = [34.48, 34.48, 34.48, 34.48, 34.48, 71.08, 173.63, 401.98, 401.98]
    TCNN_POSE = [15.18, 25.7, 101.38, 450.77, 472.72, 680.62, 715.53, 826.58, 826.58]

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(range(len(occlusion_rate)), LSTM, '-or', label='LSTM')
    ax.plot(range(len(occlusion_rate)), TCNN_POSE, '-ob', label='TCNN_POSE')

    plt.xlabel("% occlusion")
    plt.ylabel("ADE (pixels)")
    ax.legend()

    ax.set_xticks(range(len(occlusion_rate)))
    # ax.set_yticks(ade)

    ax.set_xticklabels(occlusion_rate)
    # ax.set_yticklabels(ade)

    fig.savefig("plot_1")
    plt.close()


if __name__ == '__main__':

    # plot_loss()
    plot_multiple_loss()
    # plot_1()
