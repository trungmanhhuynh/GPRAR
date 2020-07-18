'''
generate_inputs.py : plot train and vidation loss from log files.

Author: Manh Huynh
Last Update: 07/17/2020
'''

import os 
import json
import matplotlib.pyplot as plt

# log file is generarted by train a model, 
# specify LOG_FILE correctly - must be json file
LOG_FILE = "/home/manhh/github/Traj-STGCNN/save/log/log.json" 
LOG_FIGURE= "/home/manhh/github/Traj-STGCNN/save/log/log_image.png" 




def plot_loss():
    """
        plot train/validation loss
        save image to current directory.
    """
    with open(LOG_FILE, "r") as f:
        log_data = json.load(f)


    fig, ax = plt.subplots(nrows=1, ncols=1)


    ax.plot(log_data['epoch'], log_data['train_loss'], 'r',  label='train')
    ax.plot(log_data['epoch'], log_data['val_loss'], 'b', label='validation')
    plt.xlabel("#epoch")
    plt.ylabel("loss")
    ax.legend()

    fig.savefig(LOG_FIGURE)
    plt.close()


if __name__ == '__main__':

    
    plot_loss() 
