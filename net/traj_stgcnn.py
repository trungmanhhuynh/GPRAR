"""
    Predictor
"""

import torch.nn as nn
from net.predictor import Predictor
from net.reconstructor import Reconstructor


class Traj_STGCNN(nn.Module):
    def __init__(self, mode="reconstructor",
                 output_feats=2,
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
        self.mode = mode

        self.reconstructor = Reconstructor(in_channels=2,
                                           out_channels=2,
                                           obs_len=10,
                                           pred_len=10,
                                           num_keypoints=25,
                                           edge_importance_weighting=True)

        self.predictor = Predictor(output_feats=2,
                                   obs_len=10,
                                   pred_len=10,
                                   pose_features=pose_features,
                                   num_keypoints=25)

    def forward(self, pose_in, missing_keypoints=None, obs_locations=None):

        if(self.mode == "reconstructor"):

            reconstructed_pose = self.reconstructor(pose_in)
            output = reconstructed_pose                     # size ~ (batch_size, obs_len, pose_features*num_keypoints)

        elif (self.mode == "predictor"):

            # pred_poses = self.reconstructor(pose_in)

            # # not update reconstructor's weight while predicting
            # for p in self.reconstructor.parameters():
            #     p.requires_grad = False

            # pose_in[missing_keypoints] = pred_poses[missing_keypoints]
            traj_in = pose_in[:, :, 24:26]

            # 8th keypoint for locations. size ~ (batch_size, obs_len, 2)
            # traj_in = obs_locations
            pred_locations = self.predictor(pose_in, traj_in)
            output = pred_locations                # size ~ (batch_size, obs_len, 2)

        else:
            print("wrong mode, only support mode: reconstructor or predictor")
            exit(-1)

        return output
