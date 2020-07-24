import torch


class ConstVel():
    def __init__(self, obs_len = 10, pred_len = 10, cuda = True):

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.cuda = cuda

    def forward(self, traj_in):

        # traj_in.shape  ~ (batch_size, obs_len, 2)
        batch_size = traj_in.shape[0]

        # calcualte const vel from the last 2 positions.
        vel = traj_in[:,-1,:] -  traj_in[:,-2,:]

        # predict using constant velocity
        predict_traj = torch.zeros([batch_size, self.pred_len, 2], dtype=torch.float)
        if self.cuda:
            predict_traj = predict_traj.cuda()

        predict_traj[:,0,:]  = traj_in[:,-1,:] + vel  
        for t in range(1, self.pred_len):
            predict_traj[:,t,:] = predict_traj[:,t-1,:] + vel

        return predict_traj 

    __call__ = forward
 