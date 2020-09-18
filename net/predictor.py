"""
    Predictor
"""

import torch
import torch.nn as nn
from net.module import conv2d, deconv2d


class TCNN(nn.Module):
    def __init__(self, output_feats=2,
                 obs_len=10,
                 pred_len=10,
                 ):
        super().__init__()

        # init variables
        self.output_feats = output_feats
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.loc_features = 2

        # TCN for prediction
        self.encoder_location = nn.Sequential(
            conv2d(in_channels=self.loc_features, out_channels=32, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
        )

        self.intermediate = conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)

        self.decoder = nn.Sequential(
            conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0),
            conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            deconv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0),
            deconv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0),
            deconv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            deconv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1, padding=0, use_bn=False),
        )

    def forward(self, traj_in):

        # reshape input features ~ (batch_size, in_channels, obs_len, 1)
        traj_in = traj_in.permute(0, 2, 1).contiguous()
        traj_in = traj_in.unsqueeze(3)

        traj_y = self.encoder_location(traj_in)

        y = self.intermediate(traj_y)
        y = self.decoder(y)                          # y ~ (batch_size, out_channels, pred_len , 1)

        y = y.squeeze(3)
        y = y.permute(0, 2, 1).contiguous()          # y ~ (batch_size, pred_len, out_channels)

        return y


class TCNN_FLOW(nn.Module):
    def __init__(self, output_feats=2,
                 obs_len=10,
                 pred_len=10,
                 flow_feats=24
                 ):
        super().__init__()

        # init variables
        self.output_feats = output_feats
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.loc_features = 2
        self.flow_feats = flow_feats

        # TCN for prediction
        self.encoder_location = nn.Sequential(
            conv2d(in_channels=self.loc_features, out_channels=32, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
        )

        self.encoder_flow = nn.Sequential(
            conv2d(in_channels=self.flow_feats, out_channels=32, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
        )

        self.intermediate = conv2d(in_channels=128 * 2, out_channels=128, kernel_size=1, stride=1, padding=0)

        self.decoder = nn.Sequential(
            conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0),
            conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            deconv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0),
            deconv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0),
            deconv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            deconv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1, padding=0, use_bn=False),
        )

    def forward(self, traj_in, flow_in):

        # reshape input features ~ (batch_size, in_channels, obs_len, 1)
        traj_in = traj_in.permute(0, 2, 1).contiguous()
        traj_in = traj_in.unsqueeze(3)
        flow_in = flow_in.permute(0, 2, 1).contiguous()
        flow_in = flow_in.unsqueeze(3)

        traj_y = self.encoder_location(traj_in)
        flow_y = self.encoder_flow(flow_in)

        encoded_f = torch.cat((traj_y, flow_y), dim=1)

        y = self.intermediate(encoded_f)
        y = self.decoder(y)                          # y ~ (batch_size, out_channels, pred_len , 1)

        y = y.squeeze(3)
        y = y.permute(0, 2, 1).contiguous()          # y ~ (batch_size, pred_len, out_channels)

        return y


class TCNN_POSE(nn.Module):
    def __init__(self, output_feats=2,
                 obs_len=10,
                 pred_len=10,
                 pose_features=2,
                 num_keypoints=25
                 ):
        super().__init__()

        # init variables
        self.output_feats = output_feats
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.pose_features = pose_features
        self.loc_features = 2
        self.num_keypoints = num_keypoints

        # TCN for prediction
        self.encoder_location = nn.Sequential(
            conv2d(in_channels=self.loc_features, out_channels=32, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
        )

        self.encoder_pose = nn.Sequential(
            conv2d(in_channels=self.pose_features * self.num_keypoints, out_channels=32, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
        )

        self.intermediate = conv2d(in_channels=128 * 2, out_channels=128, kernel_size=1, stride=1, padding=0)

        self.decoder = nn.Sequential(
            conv2d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0),
            conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            deconv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0),
            deconv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=0),
            deconv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0),
            deconv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1, padding=0, use_bn=False),
        )

    def forward(self, pose_in, traj_in):

        # reshape input features ~ (batch_size, in_channels, obs_len, 1)
        traj_in = traj_in.permute(0, 2, 1).contiguous()
        traj_in = traj_in.unsqueeze(3)
        pose_in = pose_in.permute(0, 2, 1).contiguous()
        pose_in = pose_in.unsqueeze(3)

        traj_y = self.encoder_location(traj_in)
        pose_y = self.encoder_pose(pose_in)

        encoded_f = torch.cat((traj_y, pose_y), dim=1)

        y = self.intermediate(encoded_f)
        y = self.decoder(y)                          # y ~ (batch_size, out_channels, pred_len , 1)

        y = y.squeeze(3)
        y = y.permute(0, 2, 1).contiguous()          # y ~ (batch_size, pred_len, out_channels)

        return y


class TCNN_POSE_FLOW(nn.Module):
    def __init__(self, output_feats=2,
                 obs_len=10,
                 pred_len=10,
                 pose_features=2,
                 num_keypoints=25,
                 flow_feats=24,
                 use_fa=False
                 ):
        super().__init__()

        # init variables
        self.output_feats = output_feats
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.pose_features = pose_features
        self.loc_features = 2
        self.num_keypoints = num_keypoints
        self.flow_feats = flow_feats
        self.use_fa = use_fa

        # TCN for prediction
        self.encoder_location = nn.Sequential(
            conv2d(in_channels=self.loc_features, out_channels=32, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
        )

        self.encoder_pose = nn.Sequential(
            conv2d(in_channels=self.pose_features * self.num_keypoints, out_channels=32, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0)
        )

        self.encoder_flow = nn.Sequential(
            conv2d(in_channels=self.flow_feats, out_channels=32, kernel_size=3, stride=1, padding=0),
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
            conv2d(in_channels=32, out_channels=2, kernel_size=1, stride=1, padding=0, use_bn=False),
        )

        if (self.use_fa):
            self.fa = FeatureAttention(
                out_features=3,
                in_channels=128
            )

    def forward(self, pose_in, traj_in, flow_in):

        # reshape input features ~ (batch_size, in_channels, obs_len, 1)
        traj_in = traj_in.permute(0, 2, 1).contiguous()
        traj_in = traj_in.unsqueeze(3)
        pose_in = pose_in.permute(0, 2, 1).contiguous()
        pose_in = pose_in.unsqueeze(3)
        flow_in = flow_in.permute(0, 2, 1).contiguous()
        flow_in = flow_in.unsqueeze(3)

        traj_y = self.encoder_location(traj_in)
        pose_y = self.encoder_pose(pose_in)
        flow_y = self.encoder_flow(flow_in)

        if(self.use_fa):
            traj_y, pose_y, flow_y = self.fa(traj_y, pose_y, flow_y)

        encoded_f = torch.cat((traj_y, pose_y, flow_y), dim=1)

        y = self.intermediate(encoded_f)
        y = self.decoder(y)                          # y ~ (batch_size, out_channels, pred_len , 1)

        y = y.squeeze(3)
        y = y.permute(0, 2, 1).contiguous()          # y ~ (batch_size, pred_len, out_channels)

        return y


class FeatureAttention(nn.Module):
    def __init__(self,
                 out_features=3,
                 in_channels=128,
                 ):
        super().__init__()
        self.out_features = out_features
        self.in_channels = in_channels

        self.conv2d_1 = conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.conv2d_2 = conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=0)

        self.fcn = torch.nn.Linear(in_features=32 * 2, out_features=self.out_features)
        self.softmax = torch.nn.Softmax()

    def forward(self, location, pose, flow):

        # f shape (batch_size, channels, feature_size , 1) ~ (B,C,F,1) ~ (128, 384, 2, 1)
        # each feature has size (128, 128, 2, 1)

        f = torch.cat((location, pose, flow), dim=2)  # (128, 128, 6, 1)
        f = self.conv2d_1(f)    # (128, 64, 4, 1)
        f = self.conv2d_2(f)    # (128, 32, 2, 1)
        B, C, F, _ = f.shape
        f = f.view(B, C * F)     # (128, 64)
        s = self.softmax(self.fcn(f))       # (128,3)

        location_weights = s[:, 0].view(B, 1, 1, 1)
        pose_weights = s[:, 1].view(B, 1, 1, 1)
        flow_weights = s[:, 2].view(B, 1, 1, 1)

        location = location * location_weights
        pose = pose * pose_weights
        flow = flow * flow_weights

        return location, pose, flow

class Predictor(nn.Module):

    def __init__(self, output_feats=2,
                 obs_len=10,
                 pred_len=10,
                 pose_features=2,
                 num_keypoints=25
                 ):
        super().__init__()

        # init variables
        self.output_feats = output_feats
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.pose_features = pose_features
        self.loc_features = 2
        self.num_keypoints = num_keypoints

        self.tcnn = TCNN(
            output_feats=2,
            obs_len=10,
            pred_len=10,
        )

        # self.tcnn_flow = TCNN_FLOW(
        #     output_feats=2,
        #     obs_len=10,
        #     pred_len=10,
        #     flow_feats=24
        # )

        self.tcnn_pose = TCNN_POSE(
            output_feats=2,
            obs_len=10,
            pred_len=10,
            pose_features=2,
            num_keypoints=25
        )

        self.tcnn_pose_flow = TCNN_POSE_FLOW(
            output_feats=2,
            obs_len=10,
            pred_len=10,
            pose_features=2,
            num_keypoints=25,
            flow_feats=24,
            use_fa=True
        )

    def forward(self, pose_in, traj_in, flow_in):

        # prediction using pose + location + gridflow (camera motions)
        y = self.tcnn_pose_flow(pose_in, traj_in, flow_in)

        # prediction using pose + location
        # y = self.tcnn_pose(pose_in, traj_in)

        # prediction using location + flow
        # y = self.tcnn_flow(traj_in, flow_in)

        # prediction using location
        # y = self.tcnn(traj_in)

        return y
