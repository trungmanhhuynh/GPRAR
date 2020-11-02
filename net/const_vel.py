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

    def __init__(self, obs_len, pred_len, loc_feats, pose_feats, num_keypoints):
        super().__init__()

        # init parameters
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.loc_feats = loc_feats
        self.pose_feats = pose_feats
        self.num_keypoints = num_keypoints
        assert num_keypoints == 18          # only support 18 keypoints

        # predictor
        self.predictor = ConstVel(obs_len=obs_len,
                                  pred_len=pred_len,
                                  cuda=True)

    def forward(self, inputs):
        '''
            Inputs:
                + inputs (tuple) (obs_loc, obs_pose, obs_gridflow)
                + obs_loc (batch_size, obs_len, loc_feats)
                + obs_pose (batch_size, pose_feats, obs_len, num_keypoints, 1)
            Outputs:
                + pred_loc (batch_size, pred_len, loc_feats)
        '''

        obs_loc, obs_pose, _ = inputs

        # use noisy observation
        obs_loc = 0.5 * (obs_pose[:, 0:2, :, 8, :] + obs_pose[:, 0:2, :, 11, :])        # (batch_size, loc_feats, obs_len, 1)
        obs_loc = obs_loc.squeeze(3)
        obs_loc = obs_loc.permute(0, 2, 1)      # (batch_size, obs_len, 2)
        pred_loc = self.predictor(obs_loc)

        return pred_loc


class ConstVel():
    def __init__(self, obs_len=10, pred_len=10, cuda=True):

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.cuda = cuda

    def forward(self, traj_in):

        # traj_in.shape  ~ (batch_size, obs_len, 2)
        batch_size = traj_in.shape[0]

        # calcualte const vel from the last 2 positions.
        vel = traj_in[:, -1, :] - traj_in[:, -2, :]

        # predict using constant velocity
        predict_traj = torch.zeros([batch_size, self.pred_len, 2], dtype=torch.float)
        if self.cuda:
            predict_traj = predict_traj.cuda()

        predict_traj[:, 0, :] = traj_in[:, -1, :] + vel
        for t in range(1, self.pred_len):
            predict_traj[:, t, :] = predict_traj[:, t - 1, :] + vel

        return predict_traj

    __call__ = forward
