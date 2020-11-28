import torch
import torch.nn as nn
import torch.nn.functional as F
from net.utils.graph import Graph
from net.utils.rst_gcn import rst_gcn
from net.utils.st_gcn import st_gcn


class RegGenNet(nn.Module):
    """Spatial temporal graph convolutional networks for action recognition
        and pose generation

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # build encoder networks
        self.enc_graph = Graph(**graph_args)
        enc_A = torch.tensor(self.enc_graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('enc_A', enc_A)
        spatial_kernel_size = enc_A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.enc_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 1, **kwargs)
        ))

        # build recognition networks
        self.reg_graph = Graph(**graph_args)
        reg_A = torch.tensor(self.reg_graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('reg_A', reg_A)
        spatial_kernel_size = reg_A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.reg_networks = nn.ModuleList((
            rst_gcn(256, 128, kernel_size, 1, residual=False, **kwargs0),
            rst_gcn(128, 64, kernel_size, 1, **kwargs),
            rst_gcn(64, 32, kernel_size, 1, **kwargs)
        ))
        self.fcn = nn.Conv2d(32, num_class, kernel_size=1)

        # add reconstruction branch
        self.rec_graph = Graph(**graph_args)
        rec_A = torch.tensor(self.rec_graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('rec_A', rec_A)
        spatial_kernel_size = rec_A.size(0)
        temporal_kernel_size = 9
        self.rec_networks = nn.ModuleList((
            rst_gcn(256, 128, kernel_size, 1, residual=False, **kwargs),
            rst_gcn(128, 64, kernel_size, 1, residual=True, **kwargs),
            rst_gcn(64, 32, kernel_size, 1, residual=True, **kwargs),
            rst_gcn(32, in_channels, kernel_size, 1, residual=True, **kwargs)
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.enc_edge_imp = nn.ParameterList(
                [nn.Parameter(torch.ones(self.enc_A.size())) for i in self.enc_networks])
            self.reg_edge_imp = nn.ParameterList(
                [nn.Parameter(torch.ones(self.reg_A.size())) for i in self.reg_networks])
            self.rec_edge_imp = nn.ParameterList(
                [nn.Parameter(torch.ones(self.rec_A.size())) for i in self.rec_networks])
        else:
            self.enc_edge_imp = [1] * len(self.enc_networks)
            self.reg_edge_imp = [1] * len(self.reg_networks)
            self.rec_edge_imp = [1] * len(self.rec_networks)

        self.data_bn = nn.BatchNorm1d(in_channels * enc_A.size(1))

    def forward(self, x):
        """
        Args:
            inputs (tuple): input data consisting of features:
            obs_loc (tensor): observed location.
            obs_pose (tensor): observed pose.
            obs_gridflow (tensor): observed grid flow. Shape: (N, T, G)
        Shapes:
            obs_loc: (N, To, L)
            obs_pose:  (N, P, T, V, M)
            obs_gridflow: (N, T, G)
            Where:
                N: batch size, To: observed time, Tp: predicted time, V: number of human keypoints
                M: number of pedestrian each batch, P: number of pose features, L: number of location features
                G: number of gridflow features
        Returns:
            + pred_loc (tensor): predicted locations. Shape (N, Tp, L)
        """

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # N, M, V, C, T
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)  # ~ (N * M, C, T, V) (64, 3, 10, 18)

        # encode
        for gcn, importance in zip(self.enc_networks, self.enc_edge_imp):
            x, _ = gcn(x, self.enc_A * importance)  # x ~ (N * M, C', T, V)

        # action recognition
        x1 = x
        for gcn, importance in zip(self.reg_networks, self.reg_edge_imp):
            x1, _ = gcn(x1, self.reg_A * importance)  # x ~ (N * M, C', T, V)

        x1 = F.avg_pool2d(x1, x1.size()[2:])
        x1 = x1.view(N, M, -1, 1, 1).mean(dim=1)
        x1 = self.fcn(x1)
        x1 = x1.view(x1.size(0), -1)

        # add a branch for reconstruction
        x2 = x
        for gcn, importance in zip(self.rec_networks, self.rec_edge_imp):
            x2, _ = gcn(x2, self.rec_A * importance)                     # (N * M, C', T, V)
        x2 = x2.view(N, M, C, T, V)                                       # (N, M, C, T, V)
        x2 = x2.permute(0, 2, 3, 4, 1)                                   # (N, C, T, V, M)

        return x1, x2

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # N, M, V, C, T
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)  # ~ ([256, 3, 150, 18])

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A1 * importance)  # x ~ (N * M, C', T, V) ([256, 256, 75, 18])

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x1 = self.fcn(x)
        x1 = x1.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        # add a branch for reconstruction
        for rst_gcn, importance in zip(self.reconstructor, self.edge_importance_reconstruction):
            x, _ = rst_gcn(x, self.A1 * importance)  # x ~ (N * M, C', T, V) (64, 3, 10, 18)

        # x2 = self.last_tcn(x2)
        x2 = x.view(N, M, C, T, V)  # x ~ (N , M, C, T, V) (64, 1, 3, 10, 18)
        x2 = x2.permute(0, 2, 3, 4, 1)  # x ~ (N , C, T, V, M) (64, 3, 10, 18, 1)

        return x1, x2, feature
