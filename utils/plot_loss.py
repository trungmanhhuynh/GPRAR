'''
generate_inputs.py : plot train and vidation loss from log files.

Author: Manh Huynh
Last Update: 07/17/2020
'''

import os 
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
    TCNN_POSE_LOG_FILE = "sota/tcnn/save/tcnn_pose/small_size/full_pose/log/log.json" 
    TCNN_POSE_KP1_LOG_FILE = "sota/tcnn/save/tcnn_pose/small_size/missing_1/log/log.json" 


    LOG_FIGURE= "train_loss.png" 



    # with open(TCNN_LOG_FILE, "r") as f:
    #     tcnn_data = json.load(f)

    with open(TCNN_POSE_LOG_FILE, "r") as f:
        tcnn_pose_data = json.load(f)

    with open(TCNN_POSE_KP1_LOG_FILE, "r") as f:
        tcnn_pose_ms1_data = json.load(f)


    fig, ax = plt.subplots(nrows=1, ncols=1)

    #ax.plot(tcnn_data['epoch'], tcnn_data['train_loss'], 'r',  label='tcnn')
    ax.plot(tcnn_pose_data['epoch'], np.log10(tcnn_pose_data['train_loss']), 'b',  label='tcnn_pose')
    ax.plot(tcnn_pose_ms1_data['epoch'], np.log10(tcnn_pose_ms1_data['train_loss']), 'k',  label= 'tcnn_pose_ms1')

    #ax.plot(log_data['epoch'], log_data['val_loss'], 'b', label='validation')
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
    LOG_FIGURE= "train_loss.png" 



    with open(LOG_FILE, "r") as f:
        log_data = json.load(f)


    fig, ax = plt.subplots(nrows=1, ncols=1)


    ax.plot(log_data['epoch'], log(log_data['train_loss']), 'r',  label='train')
    #ax.plot(log_data['epoch'], log_data['val_loss'], 'b', label='validation')
    plt.xlabel("#epoch")
    plt.ylabel("loss")
    ax.legend()

    fig.savefig(LOG_FIGURE)
    plt.close()


if __name__ == '__main__':

    
    #plot_loss() 
    plot_multiple_loss()
