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
        # data: N C V T M
        #  self.bbox (N , T , 4)

        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label, self.bbox = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        if self.debug:
            self.label = self.label[0:10000]
            self.data = self.data[0:10000]
            self.sample_name = self.sample_name[0:10000]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

        print("N={} C={} T={} V={} M={}".format(self.N, self.C, self.T, self.V, self.M))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        bbox = np.array(self.bbox[index])

        # processing
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)

        noisy_data = np.copy(data_numpy)
        if self.random_noise:
            noisy_data = tools.random_noise(noisy_data)

        return noisy_data, data_numpy, label, "None", "None", bbox
