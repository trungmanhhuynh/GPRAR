"""
    Predictor
"""

import torch.nn as nn
from net.predictor import Predictor
from net.reconstructor import Reconstructor


class TrajSTGCNN(nn.Module):

    def __init__(self, output_feats=2,
                 obs_len=10,
                 pred_len=10,
                 pose_features=3,
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

        self.reconstructor = Reconstructor(in_channels=3,
                                           out_channels=3,
                                           obs_len=10,
                                           pred_len=10,
                                           num_keypoints=25,
                                           edge_importance_weighting=False)

        self.predictor = Predictor(output_feats=2,
                                   obs_len=10,
                                   pred_len=10,
                                   pose_features=3,
                                   num_keypoints=25)

    def forward(self, pose_in):

        pose_in = self.reconstructor(pose_in)

        predicted_locations = self.predictor(pose_in)

        return predicted_locations
