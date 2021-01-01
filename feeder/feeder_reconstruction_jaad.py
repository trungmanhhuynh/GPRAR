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
                 random_noise=False,
                 random_choose=False,
                 random_move=False,
                 window_size=-1,
                 debug=False,
                 mmap=True):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size
        self.random_noise = random_noise

        self.load_data(mmap)

    def load_data(self, mmap):
        """
            Args:
                data: N C V T M
                bbox: (N, T, 4)

        """

        # load label
        with open(self.label_path, 'rb') as f:
            bbox, video_names, image_names, action_labels, action_indexes = pickle.load(f)

        # load data
        data = np.load(self.data_path)

        if self.debug:
            data = data[0:10000]
            video_names = video_names[0:10000]
            image_names = image_names[0:10000]
            action_labels = action_labels[0:10000]
            action_indexes = action_indexes[0:10000]

        # scale pose using bounding box. We do this because
        # the input pose data (for reconstruction task) is highly dependent on the scale (small vs large)
        # Thus we apply normalization to make all the pose has the same scale.
        N, C, T ,V, M = data.shape
        bbox = np.expand_dims(bbox, axis=[3, 4])  # (N, 4, T, 1, 1)
        bbox = np.repeat(bbox, V, axis=3)  # (N, 4, T, V, 1)
        print(bbox.shape)
        data[:, 0] = (data[:, 0] - bbox[:, 0]) / (bbox[:, 2] - bbox[:, 0]) - 0.5
        data[:, 1] = (data[:, 1] - bbox[:, 1]) / (bbox[:, 3] - bbox[:, 1]) - 0.5

        data[:, 0][data[:, 2] == 0] = 0
        data[:, 1][data[:, 2] == 0] = 0

        self.data = data
        self.bbox = bbox
        self.video_names = video_names
        self.image_names = image_names
        self.action_labels = action_labels
        self.action_indexes = action_indexes

        print("pose shape:", self.data.shape)

    def __len__(self):
        return len(self.action_indexes)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.action_indexes[index]
        video_name = self.video_names[index]
        image_name = self.image_names[index]
        bbox = np.array(self.bbox[index])

        # processing
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        noisy_data = np.copy(data_numpy)
        if self.random_noise:
            noisy_data = tools.random_noise(noisy_data)

        return noisy_data, data_numpy, label, video_name, image_name, bbox
