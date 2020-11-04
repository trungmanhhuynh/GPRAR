import torch
import torch.nn as nn
import torch.nn.functional as F

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph
from net.utils.conv2d import conv2d
from net.utils.deconv2d import deconv2d


class Model(nn.Module):
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

    def __init__(self, obs_len, pred_len, loc_feats, pose_feats, gridflow_feats, num_keypoints,
                 graph_args, edge_importance_weighting, **kwargs):
        super().__init__()

        # init parameters
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.loc_feats = loc_feats
        self.pose_feats = pose_feats
        self.gridflow_feats = gridflow_feats
        self.num_keypoints = num_keypoints

        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        self.graph1 = Graph(**graph_args)
        A1 = torch.tensor(self.graph1.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A1', A1)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(pose_feats * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            st_gcn(self.pose_feats, 64, kernel_size, 1, residual=False, **kwargs0),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 64, kernel_size, 1, **kwargs),
            st_gcn(64, 128, kernel_size, 1, **kwargs),      # stride =2 originally
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

        # add reconstruction branch
        self.reconstructor = nn.ModuleList((
            rst_gcn(256, 128, kernel_size, 1, residual=False, **kwargs),
            rst_gcn(128, 64, kernel_size, 1, residual=True, **kwargs),
            rst_gcn(64, 64, kernel_size, 1, residual=True, **kwargs),
            rst_gcn(64, self.pose_feats, kernel_size, 1, residual=True, **kwargs)
        ))
        self.edge_importance_reconstruction = [1] * len(self.reconstructor)

        # predictor
        self.predictor = Predictor(obs_len=obs_len,
                                   pred_len=pred_len,
                                   loc_feats=loc_feats,
                                   pose_feats=pose_feats,
                                   gridflow_feats=gridflow_feats,
                                   num_keypoints=num_keypoints)
    # def forward(self, inputs):

    #     obs_loc, x, flow_in  = inputs
    #     # data normalization
    #     N, C, T, V, M = x.size()
    #     x = x.permute(0, 4, 3, 1, 2).contiguous()  # N, M, V, C, T
    #     x = x.view(N * M, V * C, T)
    #     x = self.data_bn(x)
    #     x = x.view(N, M, V, C, T)
    #     x = x.permute(0, 1, 3, 4, 2).contiguous()
    #     x = x.view(N * M, C, T, V)                 # ~ (N * M, C, T, V)

    #     # forwad
    #     for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
    #         x, _ = gcn(x, self.A * importance)          # x ~ (N * M, C', T, V)

    #     _, c, t, v = x.size()
    #     action_in = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

    #     # global pooling
    #     x1 = F.avg_pool2d(x, x.size()[2:])
    #     x1 = x1.view(N, M, -1, 1, 1).mean(dim=1)

    #     # reconstruction branch
    #     for rst_gcn, importance in zip(self.reconstructor, self.edge_importance_reconstruction):
    #         x, _ = rst_gcn(x, self.A1 * importance)        # x ~ (N * M, C', T, V)

    #     x2 = x.view(N, M, C, T, V)                # x ~ (N , M, C, T, V)
    #     x2 = x2.permute(0, 2, 3, 4, 1).contiguous()            # x ~ (N , C, T, V, M)

    #     # extract observed locations
    #     left_hip = x2[:, :, :, 8, :]
    #     right_hip = x2[:, :, :, 11, :]
    #     traj_in = 0.5 * (left_hip + right_hip)

    #     pred_locs = self.predictor(pose_in=x2, traj_in=traj_in, flow_in=flow_in, action_in=action_in)

    #     return pred_locs

    def forward(self, inputs):
        '''
            Inputs:
                + inputs (tuple) (obs_loc, obs_pose, obs_gridflow)
                + obs_loc (batch_size, obs_len, loc_feats)
                + obs_pose (batch_size, pose_feats, obs_len, num_keypoints, 1)
                + obs_gridflow (batch_size, obs_len, gridflow_feats)
            Outputs:
                + pred_loc (batch_size, pred_len, loc_feats)
        '''

        obs_loc, obs_pose, obs_gridflow = inputs
        batch_size = obs_pose.shape[0]

        # Reconstructing pose
        # data normalization
        x = obs_pose
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()                   # N, M, V, C, T
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)                       # (N * M, C', T, V)

        # reconstruction branch
        for rst_gcn, importance in zip(self.reconstructor, self.edge_importance_reconstruction):
            x, _ = rst_gcn(x, self.A1 * importance)                  # (N * M, C', T, V)
        rpose = x.view(N, M, C, T, V)                                # (N , M, C, T, V)
        rpose = rpose.permute(0, 2, 3, 4, 1).contiguous()            # (N , C, T, V, M)

        # reshape
        obs_loc = obs_loc.permute(0, 2, 1).unsqueeze(3)               # (batch_size, loc_feats, obs_len, 1)
        obs_gridflow = obs_gridflow.permute(0, 2, 1).unsqueeze(3)     # (batch_size, gridflow_feats, obs_len, 1)
        rpose = rpose.permute(0, 3, 1, 2, 4)                          # (batch_size, num_keypoints, pose_feats, obs_len, 1)
        rpose = rpose.reshape(batch_size, self.num_keypoints * self.pose_feats, self.obs_len, 1)

        # predict
        pred_loc = self.predictor(obs_loc, rpose, obs_gridflow)

        return pred_loc

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # N, M, V, C, T
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)                 # ~ ([256, 3, 150, 18])

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A1 * importance)          # x ~ (N * M, C', T, V) ([256, 256, 75, 18])

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x1 = self.fcn(x)
        x1 = x1.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        # add a branch for reconstruction
        for rst_gcn, importance in zip(self.reconstructor, self.edge_importance_reconstruction):
            x, _ = rst_gcn(x, self.A1 * importance)         # x ~ (N * M, C', T, V) (64, 3, 10, 18)

        # x2 = self.last_tcn(x2)
        x2 = x.view(N, M, C, T, V)                # x ~ (N , M, C, T, V) (64, 1, 3, 10, 18)
        x2 = x2.permute(0, 2, 3, 4, 1)  # x ~ (N , C, T, V, M) (64, 3, 10, 18, 1)

        return x1, x2, feature


class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True,
                 window_size=None):
        super().__init__()

        assert len(kernel_size) == 2                    # kernel_size = (9,3)
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True)
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)           # A.shape (3,18,18)
        x = self.tcn(x) + res

        return x, A


class rst_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True,
                 window_size=None):

        super().__init__()

        assert len(kernel_size) == 2                    # kernel_size = (9,3)
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        # padding = (0, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                ),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + res

        return x, A


class Predictor(nn.Module):
    def __init__(self, obs_len, pred_len, loc_feats, pose_feats, gridflow_feats, num_keypoints):
        super().__init__()

        # init variables
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.loc_feats = loc_feats
        self.pose_feats = pose_feats
        self.gridflow_feats = gridflow_feats
        self.num_keypoints = num_keypoints

        # TCN for prediction
        self.encoder_location = nn.Sequential(
            conv2d(in_channels=self.loc_feats, out_channels=32, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
        )

        self.encoder_pose = nn.Sequential(
            conv2d(in_channels=self.pose_feats * self.num_keypoints, out_channels=32, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
        )

        self.encoder_gridflow = nn.Sequential(
            conv2d(in_channels=self.gridflow_feats, out_channels=32, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
        )

        self.intermediate = conv2d(in_channels=128 * 3, out_channels=128, kernel_size=1, stride=1, padding=0)

        self.decoder = nn.Sequential(
            conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0),
            conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            deconv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0),
            deconv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0),
            deconv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            deconv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, obs_loc, obs_pose, obs_gridflow):
        '''
            Inputs:
                + obs_loc (batch_size, loc_feats, obs_len, 1)
                + obs_pose (batch_size, pose_feats*num_keypoints, obs_len, 1)
                + obs_gridflow (batch_size, gridflow_feats, obs_len, 1)
            Outputs:
                + pred_loc (batch_size, pred_len, loc_feats)

        '''
        # encode
        inter_loc = self.encoder_location(obs_loc)
        inter_pose = self.encoder_pose(obs_pose)
        inter_gridflow = self.encoder_gridflow(obs_gridflow)

        # intermediate
        f = torch.cat((inter_loc, inter_pose, inter_gridflow), dim=1)
        f = self.intermediate(f)

        # decode
        pred_loc = self.decoder(f)                                   # (batch_size, loc_feats, pred_len , 1)
        pred_loc = pred_loc.squeeze(3)
        pred_loc = pred_loc.permute(0, 2, 1).contiguous()            # (batch_size, pred_len, loc_feats)

        return pred_loc
