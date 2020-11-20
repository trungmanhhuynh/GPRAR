import torch
import torch.nn as nn
import torch.nn.functional as F
from net.utils.graph import Graph
from net.utils.rst_gcn import rst_gcn
from net.utils.st_gcn import st_gcn

class RecPoseNet(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, **kwargs):
        super().__init__()

        # load graph

        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}

        # print("kwargs:", kwargs)
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 1, **kwargs),  # stride =2 originally
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 128, kernel_size, 1, **kwargs),
            st_gcn(128, 256, kernel_size, 1, **kwargs),  # stride =2 originally
            st_gcn(256, 256, kernel_size, 1, **kwargs),
            st_gcn(256, 256, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

        # add reconstruction branch
        self.reconstructor_graph = Graph(**graph_args)
        rA = torch.tensor(self.reconstructor_graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('rA', rA)
        self.reconstructor = nn.ModuleList((
            rst_gcn(256, 128, kernel_size, 1, residual=False, **kwargs),
            rst_gcn(128, 64, kernel_size, 1, residual=True, **kwargs),
            rst_gcn(64, 64, kernel_size, 1, residual=True, **kwargs),
            rst_gcn(64, in_channels, kernel_size, 1, residual=True, **kwargs)
        ))
        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.redge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.rA.size()))
                for i in self.reconstructor
            ])
        else:
            self.redge_importance = [1] * len(self.reconstructor)


    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # N, M, V, C, T
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)  # ~ (N * M, C, T, V) (64, 3, 10, 18)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)  # x ~ (N * M, C', T, V) (64, 256, 10, 18)

        # global pooling
        x1 = F.avg_pool2d(x, x.size()[2:])
        x1 = x1.view(N, M, -1, 1, 1).mean(dim=1)

        # prediction
        x1 = self.fcn(x1)
        x1 = x1.view(x1.size(0), -1)

        # add a branch for reconstruction
        for rst_gcn, importance in zip(self.reconstructor, self.redge_importance):
            x, _ = rst_gcn(x, self.rA * importance)  # x ~ (N * M, C', T, V) (64, 3, 10, 18)

        # x2 = self.last_tcn(x2)
        x2 = x.view(N, M, C, T, V)  # x ~ (N , M, C, T, V) (64, 1, 3, 10, 18)
        x2 = x2.permute(0, 2, 3, 4, 1)  # x ~ (N , C, T, V, M) (64, 3, 10, 18, 1)

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
