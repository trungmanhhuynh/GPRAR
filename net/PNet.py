import torch
import torch.nn as nn
from net.utils.graph import Graph
from net.utils.conv2d import conv2d
from net.utils.deconv2d import deconv2d
from net.utils.st_gcn import st_gcn
from net.utils.rst_gcn import rst_gcn


class PNet(nn.Module):
    """Trajectory Prediction Network. 

    Args:
        obs_len (int): Number of channels in the input data
        pred_len (int): Number of classes for the classification task
        loc_feats (int) : Number of location features
        pose_feats (int) : Number of pose features
        gridflow_feats (int) : Number of grid flow features
        num_keypoints (int): Number of human joints
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units
    """

    def __init__(self, num_class, obs_len, pred_len, loc_feats, pose_feats, gridflow_feats, action_feats, num_keypoints,
                 graph_args, edge_importance_weighting, **kwargs):
        super().__init__()

        # init variables
        self.loc_feats = loc_feats
        self.pose_feats = pose_feats
        self.gridflow_feats = gridflow_feats
        self.action_feats = action_feats
        self.obs_len = obs_len
        self.pre_len = pred_len
        self.num_keypoints = num_keypoints

        # build encoder networks
        self.enc_graph = Graph(**graph_args)
        enc_A = torch.tensor(self.enc_graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('enc_A', enc_A)
        spatial_kernel_size = enc_A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.enc_networks = nn.ModuleList((
            st_gcn(pose_feats, 64, kernel_size, 1, residual=False, **kwargs0),
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
            rst_gcn(64, 32, kernel_size, 1, **kwargs),
            rst_gcn(32, 3, kernel_size, 1, **kwargs)
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
            rst_gcn(32, pose_feats, kernel_size, 1, residual=True, **kwargs)
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

        self.data_bn = nn.BatchNorm1d(pose_feats * enc_A.size(1))

        # predictor
        self.predictor = Predictor(obs_len=obs_len, pred_len=pred_len, loc_feats=loc_feats,
                                   pose_feats=pose_feats, gridflow_feats=gridflow_feats,
                                   action_feats=action_feats, num_keypoints=num_keypoints)

        #
        # for p in self.reg_networks.parameters():
        #     p.requires_grad = False
        # for p in self.rec_networks.parameters():
        #     p.requires_grad = False
        # for p in self.enc_networks.parameters():
        #     p.requires_grad = False
        # for p in self.data_bn.parameters():
        #     p.requires_grad = False
        # for p in self.fcn.parameters():
        #     p.requires_grad = False
        # for p in self.enc_edge_imp.parameters():
        #     p.requires_grad = False
        # for p in self.reg_edge_imp.parameters():
        #     p.requires_grad = False
        # for p in self.rec_edge_imp.parameters():
        #     p.requires_grad = False

    def forward(self, inputs):
        """
            Args:
                inputs (tuple): input data consisting of features:
                obs_loc (tensor): observed location.
                obs_pose (tensor): observed pose.
                obs_gridflow (tensor): observed grid flow. Shape: (N, T, G)
            Shapes:
                obs_loc: (N, To, L)
                obs_pose:  (N, P, T, V)
                obs_gridflow: (N, T, G)
                Where:
                N: batch size, To: observed time, Tp: predicted time, V: number of human keypoints
                M: number of pedestrian each batch, P: number of pose features, L: number of location features
                G: number of gridflow features
            Returns:
                + pred_loc (tensor): predicted locations. Shape (N, Tp, L)
        """

        # extract features
        obs_pose, obs_gridflow, bbox = inputs
        N, C, T, V = obs_pose.size()
        _, _, G = obs_gridflow.size()
        assert self.gridflow_feats == G
        assert self.pose_feats == C
        A = self.action_feats

        # data normalization
        x = obs_pose
        x = x.permute(0, 3, 1, 2).contiguous()  # N, V, C, T
        x = x.view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T)
        x = x.permute(0, 2, 3, 1).contiguous()  # (N, C, T, V)

        # encode
        for gcn, importance in zip(self.enc_networks, self.enc_edge_imp):
            x, _ = gcn(x, self.enc_A * importance)  # x ~ (N , C', T, V)

        # action recognition
        x1 = x
        for gcn, importance in zip(self.reg_networks, self.reg_edge_imp):
            x1, _ = gcn(x1, self.reg_A * importance)  # x ~ (N, A, T, V)

        # pose reconstruction
        x2 = x
        for gcn, importance in zip(self.rec_networks, self.rec_edge_imp):
            x2, _ = gcn(x2, self.rec_A * importance)  # (N, C, T, V)

        rec_pose = x2
        act_feature = x1

        # replace the missing keypoints
        rec_pose[obs_pose != 0] = obs_pose[obs_pose != 0]
        out_pose = rec_pose.clone()

        # reshape for prediction
        rec_pose[:, 0] = ((rec_pose[:, 0] + 0.5) * (bbox[:, 2] - bbox[:, 0]) + bbox[:, 0])
        rec_pose[:, 1] = ((rec_pose[:, 1] + 0.5) * (bbox[:, 3] - bbox[:, 1]) + bbox[:, 1])

        obs_loc = 0.5 * (rec_pose[:, :2, :, 8] + rec_pose[:, :2, :, 11])  # (N, L, T)
        obs_loc = obs_loc.unsqueeze(3)  # (N, L, T, 1)

        rec_pose = rec_pose.permute(0, 3, 1, 2).contiguous()  # (N, V, C, T)
        rec_pose = rec_pose.view(N, V * C, T)
        rec_pose = rec_pose.unsqueeze(3)  # (N, V * C, T, 1)

        obs_gridflow = obs_gridflow.permute(0, 2, 1).unsqueeze(3)  # (N, G, To, 1)

        act_feature = act_feature.permute(0, 3, 1, 2).contiguous()  # (N, V, A, T)
        act_feature = act_feature.view(N, V * A, T)
        act_feature = act_feature.unsqueeze(3)  # (N, V * A, T, 1)

        # predict
        pred_loc = self.predictor(obs_loc, rec_pose, obs_gridflow, act_feature)  # (N, Tp, L)

        return pred_loc, out_pose

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
            x, _ = gcn(x, self.rA * importance)  # x ~ (N * M, C', T, V) ([256, 256, 75, 18])

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x1 = self.fcn(x)
        x1 = x1.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        # add a branch for reconstruction
        for rst_gcn, importance in zip(self.reconstructor, self.edge_importance_reconstruction):
            x, _ = rst_gcn(x, self.rA * importance)  # x ~ (N * M, C', T, V) (64, 3, 10, 18)

        # x2 = self.last_tcn(x2)
        x2 = x.view(N, M, C, T, V)  # x ~ (N , M, C, T, V)
        x2 = x2.permute(0, 2, 3, 4, 1)  # x ~ (N , C, T, V, M)

        return x1, x2, feature


class Predictor(nn.Module):
    """ Prediction Network consisting of encoders and a decoder. Encoder and
    decoder are built using temporal convolutional networks (TCN).
    Args:
        obs_len: number of observed frames
        pred_len: number of predicted frames
        loc_feats: number of location features, default is 2 (x,y)
        pose_feats: number of pose features, default is 3 (x,y,c)
        action_feats: number of action features
        num_keypoints: number of human joints.
        gridflow_features: number of grid optical flow features
    """

    def __init__(self, obs_len, pred_len, loc_feats, pose_feats, gridflow_feats, action_feats, num_keypoints):
        super().__init__()

        # init variables
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.loc_feats = loc_feats
        self.pose_feats = pose_feats
        self.gridflow_feats = gridflow_feats
        self.action_feats = action_feats
        self.num_keypoints = num_keypoints

        # location encoder network
        self.encoder_location = nn.Sequential(
            conv2d(in_channels=self.loc_feats, out_channels=32, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
        )
        # pose encoder network
        self.encoder_pose = nn.Sequential(
            conv2d(in_channels=self.pose_feats * self.num_keypoints, out_channels=32, kernel_size=3, stride=1,
                   padding=0),
            conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
        )
        # gridflow encoder network
        self.encoder_gridflow = nn.Sequential(
            conv2d(in_channels=self.gridflow_feats, out_channels=32, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
        )

        #  action network
        self.encoder_action = nn.Sequential(
            conv2d(in_channels=self.action_feats * self.num_keypoints, out_channels=32, kernel_size=3, stride=1,
                   padding=0),
            conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
        )

        # intermediate layers for concatenating features
        self.intermediate = conv2d(in_channels=128 * 4, out_channels=128, kernel_size=1, stride=1, padding=0)
        # decoder network
        self.decoder = nn.Sequential(
            conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0),
            conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            deconv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0),
            deconv2d(in_channels=256, out_channels=128, kernel_size=3, st ride=1, padding=0),
            deconv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            deconv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, obs_loc, obs_pose, obs_gridflow, obs_act):
        """
            Args:
                obs_loc (tensor): observed location feature/,.
                obs_pose (tensor): observed pose feature
                obs_gridflow (tensor): observed grid optical flow feature
            Shape:
               obs_loc: (N, L, To, 1)
                   obs_pose: (N, V * P, To, 1)
               obs_gridflow: (N, G, To, 1)
               obs_act: (N, A, To, 1)      # A = V * P
            Return:
                + pred_loc (N, Tp, L)
        """
        # encode feature
        inter_loc = self.encoder_location(obs_loc)
        inter_pose = self.encoder_pose(obs_pose)
        inter_gridflow = self.encoder_gridflow(obs_gridflow)
        inter_act = self.encoder_action(obs_act)

        # concatenate encoded features
        f = torch.cat((inter_loc, inter_pose, inter_gridflow, inter_act), dim=1)
        # f = torch.cat((inter_loc, inter_pose), dim=1)
        f = self.intermediate(f)

        # decode
        pred_loc = self.decoder(f)  # (N, L, Tp , 1)
        pred_loc = pred_loc.squeeze(3)  # (N, L, Tp)
        pred_loc = pred_loc.permute(0, 2, 1).contiguous()  # (N, Tp, L)

        return pred_loc
