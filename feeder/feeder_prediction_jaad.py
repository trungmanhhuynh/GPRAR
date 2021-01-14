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
            + poses (N, C, T, V)
            + bbox: (N, 4, T)
        '''

        # load label
        with open(self.label_path, 'rb') as f:
            bbox, video_names, image_names, action_labels, action_indexes = pickle.load(f)

        # load data
        with open(self.data_path, 'rb') as f:
            gt_locations, poses, gridflow = pickle.load(f)

        if self.debug:
            gt_locations = gt_locations[0:10000]
            poses = poses[0:10000]
            gridflow = gridflow[0:10000]
            video_names = video_names[0:10000]
            image_names = image_names[0:10000]
            action_labels = action_labels[0:10000]
            action_indexes = action_indexes[0:10000]

        bbox = np.expand_dims(bbox, axis=3)    # (N, 4, T, 1)
        bbox = np.repeat(bbox, 18, axis=3)   # (N, 4, T, V)
        poses[:, 0] = (poses[:, 0] - bbox[:, 0]) / (bbox[:, 2] - bbox[:, 0]) - 0.5
        poses[:, 1] = (poses[:, 1] - bbox[:, 1]) / (bbox[:, 3] - bbox[:, 1]) - 0.5
        poses[:, 0][poses[:, 2] == 0] = 0
        poses[:, 1][poses[:, 2] == 0] = 0

        self.poses = poses
        self.gridflow = gridflow
        self.bbox = bbox
        self.gt_locations = gt_locations
        self.video_names = video_names
        self.image_names = image_names
        self.action_indexes = action_indexes
        self.action_labels = action_labels

        print("pose data shape:", self.poses.shape)
        print("location data shape:", self.gt_locations.shape)
        print("gridflow data shape:", self.gridflow.shape)

    def __len__(self):
        return self.poses.shape[0]

    def __getitem__(self, index):

        # get data
        obs_pose = self.poses[index, :, 0:self.obs_len, :]  # (1, C, obs_len, V)
        obs_gridflow = self.gridflow[index, 0:self.obs_len, :]    # (1, obs_len, 24)
        gt_location = self.gt_locations[index, -self.pred_len:, :]     # (1, pred_len, 2)
        bbox = self.bbox[index, :, 0:self.obs_len, :]
        video_name = self.video_names[index]
        image_name = self.image_names[index]
        label = self.action_labels[index]

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

        return obs_pose, obs_gridflow, gt_location, bbox, video_name, image_name, label
