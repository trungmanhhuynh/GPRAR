"""
    reconstructor model
"""


import torch
import torch.nn as nn
from net.graph import Graph
from net.module import st_gcn

class Reconstructor(nn.Module):

    # reference : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""Spatial temporal graph convolutional networks.

    """

    def __init__(self,
                 in_channels=3,   # [x,y,c]
                 out_channels=3,
                 obs_len=10,
                 pred_len=10,
                 num_keypoints=25,
                 edge_importance_weighting=False):

        super().__init__()

        # init variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.num_keypoints = num_keypoints
        self.edge_importance_weighting = edge_importance_weighting

        # load graph
        self.graph = Graph(layout='openpose_25',
                           strategy='spatial',
                           max_hop=1,
                           dilation=1)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)    # A ~ (1,25,25)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        self.data_bn = nn.BatchNorm1d(self.in_channels * A.size(1))

        self.st_gcn_networks = nn.ModuleList((
            st_gcn(self.in_channels, 32, (temporal_kernel_size, spatial_kernel_size), 1, residual=False, dropout=0.5),
            st_gcn(32, 64, (temporal_kernel_size, spatial_kernel_size), 1),
            st_gcn(64, 128, (temporal_kernel_size, spatial_kernel_size), 1),
            st_gcn(128, 256, (temporal_kernel_size, spatial_kernel_size), 1),
            st_gcn(256, 256, (temporal_kernel_size, spatial_kernel_size), 1),
            st_gcn(256, 128, (temporal_kernel_size, spatial_kernel_size), 1),
            st_gcn(128, 64, (temporal_kernel_size, spatial_kernel_size), 1),
            st_gcn(64, 32, (temporal_kernel_size, spatial_kernel_size), 1),
            st_gcn(32, self.out_channels, (temporal_kernel_size, spatial_kernel_size), 1)
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

    def forward(self, pose_in):

        # pose_in ~ (batch_size, obs_len, num_keypoints*in_channels)
        assert self.obs_len == pose_in.shape[1]
        assert self.num_keypoints * self.in_channels == pose_in.shape[2]
        batch_size = pose_in.shape[0]

        # re-shape pose_in to shape (batch_size, in_channels, obs_len, num_keypoints)
        pose_in = pose_in.view(batch_size, self.obs_len, self.num_keypoints, self.in_channels)
        pose_in = pose_in.permute(0, 3, 1, 2).contiguous()   # (batch_size, in_channels, obs_len, num_keypoints)

        # re-construct missing keypoints.
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            pose_in, _ = gcn(pose_in, self.A * importance)  # ~ (batch_size, out_channels, pred_len, num_keypoints)

        # reshape output
        output = pose_in.permute(0, 2, 3, 1).contiguous()   # (batch_size, pred_len, num_keypoints, out_channels)
        output = output.view(batch_size, self.pred_len, self.num_keypoints * self.out_channels)

        return output
