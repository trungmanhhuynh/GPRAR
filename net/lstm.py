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
        self.predictor = LSTM(obs_len=obs_len,
                              pred_len=pred_len,
                              use_cuda=True)

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
        obs_loc = obs_loc.permute(0, 2, 1)
        pred_loc = self.predictor(obs_loc)

        return pred_loc


class LSTM(nn.Module):
    def __init__(self, obs_len=10, pred_len=10, dropout=0.1, use_cuda=True):
        super(LSTM, self).__init__()

        self.use_cuda = use_cuda
        self.obs_len = obs_len
        self.pred_len = pred_len

        self.rnn_size = 128                                      # rnn_size of ALL LSTMs
        self.input_size = 2
        self.output_size = 2

        self.lstm = nn.LSTM(self.input_size, self.rnn_size, num_layers=1)
        self.relu = nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)
        self.last = nn.Linear(self.rnn_size, self.output_size)

    def init_target_hidden_states(self, batch_size):
       # Initialize states for all targets in current batch
        h0 = torch.zeros(1, batch_size, self.rnn_size)
        c0 = torch.zeros(1, batch_size, self.rnn_size)
        if(self.use_cuda):
            h0, c0 = h0.cuda(), c0.cuda()
        return h0, c0

    def forward(self, traj_in):

        #  traj_in ~  # [batch_size, obs_len , 2]
        batch_size = traj_in.shape[0]
        h_t, c_t = self.init_target_hidden_states(batch_size)
        traj_in = traj_in.permute(1, 0, 2)  # [obs_len, batch_size , 2]

        # obseve duration - learn hidden states
        for t in range(self.obs_len):
            traj_t, h_t, c_t = self.forward_t(traj_in[t, :, :].unsqueeze(0), h_t, c_t)

        # predict
        traj_pred = []
        for t in range(self.pred_len):
            traj_pred.append(traj_t)
            traj_t, h_t, c_t = self.forward_t(traj_t, h_t, c_t)

        traj_pred = torch.stack(traj_pred).squeeze(1)  # [pred_len, batch_size, 2]
        traj_pred = traj_pred.permute(1, 0, 2)  # [batch_size, pred_len, 2]

        return traj_pred

    def forward_t(self, traj_t, h_t, c_t):

        # traj_t ~ [1, batch_size , input_size]
        output, (h_next, c_next) = self.lstm(traj_t, (h_t, c_t))  # [1, batch_size, rnn_size]
        output = self.relu(output)
        output = self.dropout(output)
        pred_t = self.last(output)  # [1, batch_size, output_size]

        return pred_t, h_next, c_next

    __call__ = forward
