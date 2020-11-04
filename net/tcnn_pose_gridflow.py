import torch
import torch.nn as nn
from net.utils.conv2d import conv2d
from net.utils.deconv2d import deconv2d


class Model(nn.Module):
    """
        Model: TCNN (use locations) for human trajectory prediction.
        Inputs:
            + obs_len (int) : observation len
            + pred_len (int) : prediction len
            + loc_feats (int) : number of location feature, default 2 (x,y)
            + pose_feats (int) : number of pose features, default 3 (x,y,z)
    """

    def __init__(self, obs_len, pred_len, loc_feats, pose_feats, gridflow_feats,
                 num_keypoints, gt_observation=False):
        super().__init__()

        # init parameters
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.loc_feats = loc_feats
        self.pose_feats = pose_feats
        self.gridflow_feats = gridflow_feats
        self.num_keypoints = num_keypoints
        assert num_keypoints == 18          # only support 18 keypoints

        # predictor
        self.predictor = Predictor(obs_len=obs_len,
                                   pred_len=pred_len,
                                   loc_feats=loc_feats,
                                   pose_feats=pose_feats,
                                   gridflow_feats=gridflow_feats,
                                   num_keypoints=num_keypoints)

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

        # reshape
        obs_loc = obs_loc.permute(0, 2, 1).unsqueeze(3)   # (batch_size, loc_feats, obs_len, 1)
        obs_gridflow = obs_gridflow.permute(0, 2, 1).unsqueeze(3)     # (batch_size, gridflow_feats, obs_len, 1)
        obs_pose = obs_pose.permute(0, 3, 1, 2, 4)  # (batch_size, num_keypoints, pose_feats, obs_len, 1)
        obs_pose = obs_pose.reshape(batch_size, self.num_keypoints * self.pose_feats, self.obs_len, 1)

        # predict
        pred_loc = self.predictor(obs_loc, obs_pose, obs_gridflow)

        return pred_loc


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
