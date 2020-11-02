# sys
import os
import sys
import numpy as np
import random
import pickle

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

# visualization
import time

# operation
from . import tools


class Feeder(torch.utils.data.Dataset):
    """ Feeder for skeleton-based action recognition
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
        label_path: the path to label
        random_choose: If true, randomly choose a portion of the input sequence
        random_shift: If true, randomly pad zeros at the begining or end of sequence
        window_size: The length of the output sequence
        normalization: If true, normalize input sequence
        debug: If true, only use the first 100 samples
    """

    def __init__(self,
                 data_path,
                 label_path,
                 obs_len=10,
                 pred_len=10,
                 random_noise=False,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 mmap=False):

        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.random_noise = random_noise
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.load_data(mmap)

    def load_data(self, mmap):
        '''
            + poses (N, C, T, V, 1)
        '''

        # load label
        with open(self.label_path, 'rb') as f:
            self.video_names, self.image_names, self.action_labels, self.action_indexes = pickle.load(f)

        # load data
        with open(self.data_path, 'rb') as f:
            self.locations, self.poses, self.gridflow = pickle.load(f)

        if self.debug:
            self.locations = self.locations[0:10000]
            self.poses = self.poses[0:10000]
            self.gridflow = self.gridflow[0:10000]
            self.video_names = self.video_names[0:10000]
            self.image_names = self.image_names[0:10000]
            self.action_labels = self.action_labels[0:10000]
            self.action_indexes = self.action_indexes[0:10000]

        self.N, self.C, self.T, self.V, self.M = self.poses.shape

        print("pose data shape:", self.poses.shape)
        print("location data shape:", self.locations.shape)
        print("gridflow data shape:", self.gridflow.shape)

    def __len__(self):
        return self.N

    def __getitem__(self, index):

        # get data
        obs_pose = self.poses[index, :, 0:self.obs_len, :, :]  # (1, C, obs_len, V, M)
        obs_location = self.locations[index, 0:self.obs_len, :]  # (1, obs_len, 2)
        obs_gridflow = self.gridflow[index, 0:self.obs_len, :]    # (1, obs_len, 24)
        gt_location = self.locations[index, -self.pred_len:, :]     # (1, pred_len, 2)

        # processing
        # if self.random_choose:
        #     data_numpy = tools.random_choose(data_numpy, self.window_size)
        # elif self.window_size > 0:
        #     data_numpy = tools.auto_pading(data_numpy, self.window_size)
        # if self.random_move:
        #     data_numpy = tools.random_move(data_numpy)
        # noisy_data = np.copy(data_numpy)
        # if self.random_noise:
        #     noisy_data = tools.random_noise(noisy_data)

        return obs_location, obs_pose, obs_gridflow, gt_location
