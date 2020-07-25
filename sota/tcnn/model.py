import numpy as np
import chainer
import cupy
from chainer import Variable, cuda
import chainer.functions as F

from sota.tcnn.module import Conv_Module, Encoder, Decoder

class TCNN(chainer.Chain):
    """
    Baseline: location only
    """
    def __init__(self,channel_list, dc_channel_list, ksize_list,
                 dc_ksize_list, inter_list, last_list, pad_list):
        super(TCNN, self).__init__()
        if len(ksize_list) > 0 and len(dc_ksize_list) == 0:
            dc_ksize_list = ksize_list
        with self.init_scope():
            self.nb_inputs = 2  # [x,y]
            self.pos_encoder = Encoder(self.nb_inputs, channel_list, ksize_list, pad_list)
            self.pos_decoder = Decoder(dc_channel_list[-1], dc_channel_list, dc_ksize_list[::-1])
            self.inter = Conv_Module(channel_list[-1], dc_channel_list[0], inter_list)
            self.last = Conv_Module(dc_channel_list[-1], self.nb_inputs, last_list, True)

    def __call__(self, traj_in):
        # traj_in ~ [batch_size, obs_len, 2]

        h = self.pos_encoder(traj_in)
        h = self.inter(h)
        h = self.pos_decoder(h)
        pred_locations = self.last(h)      
        pred_locations = F.swapaxes(pred_locations, 1, 2)         #  ~ [batch_size, pred_len, 2]

        return pred_locations 

class CNN_scale(chainer.Chain):
    """
    Baseline: location only
    """
    def __init__(self,channel_list, dc_channel_list, ksize_list,
                 dc_ksize_list, inter_list, last_list, pad_list):
        super(CNN_scale, self).__init__()
        if len(ksize_list) > 0 and len(dc_ksize_list) == 0:
            dc_ksize_list = ksize_list
        with self.init_scope():
            self.nb_inputs = 3         # [x,y,scale]
            self.pos_encoder = Encoder(self.nb_inputs, channel_list, ksize_list, pad_list)
            self.pos_decoder = Decoder(dc_channel_list[-1], dc_channel_list, dc_ksize_list[::-1])
            self.inter = Conv_Module(channel_list[-1], dc_channel_list[0], inter_list)
            self.last = Conv_Module(dc_channel_list[-1], self.nb_inputs, last_list, True)

    def __call__(self, traj_in, scale_in, pose_in, ego_in, traj_gt, scale_gt):

        h = self.pos_encoder(traj_scale_in)
        h = self.inter(h)
        h = self.pos_decoder(h)
        pred_locations = self.last(h)
        
        pred_locations = F.swapaxes(pred_locations, 1, 2)       # [num_ped, pred_len, 3]
        loss = F.mean_squared_error(pred_locations, traj_scale_gt)

        return pred_locations[:,:,:2], loss

class CNN_pose_scale(chainer.Chain):
    """
    Baseline: feeds locations and poses
    """
    def __init__(self,channel_list, dc_channel_list, ksize_list,
                 dc_ksize_list, inter_list, last_list, pad_list):
        super(CNN_pose_scale, self).__init__()
        if len(ksize_list) > 0 and len(dc_ksize_list) == 0:
            dc_ksize_list = ksize_list
        with self.init_scope():
            self.nb_inputs = 3
            self.pos_encoder = Encoder(self.nb_inputs, channel_list, ksize_list, pad_list)
            self.pose_encoder = Encoder(36, channel_list, ksize_list, pad_list)
            self.pos_decoder = Decoder(dc_channel_list[-1], dc_channel_list, dc_ksize_list[::-1])
            self.inter = Conv_Module(channel_list[-1]*2, dc_channel_list[0], inter_list)
            self.last = Conv_Module(dc_channel_list[-1], self.nb_inputs, last_list, True)

    def __call__(self, traj_in, scale_in, pose_in, ego_in, traj_gt, scale_gt):

        traj_scale_in = F.concat((traj_in, scale_in), axis=2)
        traj_scale_gt = F.concat((traj_gt, scale_gt), axis=2)
        
        h_pos = self.pos_encoder(traj_scale_in)
        h_pose = self.pose_encoder(pose_in)
        h = F.concat((h_pos, h_pose), axis=1) 
        h = self.inter(h)
        h_pos = self.pos_decoder(h)
        pred_locations = self.last(h_pos)  
        pred_locations = F.swapaxes(pred_locations, 1, 2)       # [num_ped, pred_len, 3]
        loss = F.mean_squared_error(pred_locations, traj_scale_gt)

        return pred_locations[:,:,:2], loss

class CNN_ego_scale(chainer.Chain):
    """
    Baseline: feeds locations and poses
    """
    def __init__(self,channel_list, dc_channel_list, ksize_list,
                 dc_ksize_list, inter_list, last_list, pad_list):
        super(CNN_ego_scale, self).__init__()
        if len(ksize_list) > 0 and len(dc_ksize_list) == 0:
            dc_ksize_list = ksize_list
        with self.init_scope():
            self.nb_inputs = 3
            self.pos_encoder = Encoder(self.nb_inputs, channel_list, ksize_list, pad_list)
            self.ego_encoder = Encoder(24, channel_list, ksize_list, pad_list)
            self.pos_decoder = Decoder(dc_channel_list[-1], dc_channel_list, dc_ksize_list[::-1])
            self.inter = Conv_Module(channel_list[-1]*2, dc_channel_list[0], inter_list)
            self.last = Conv_Module(dc_channel_list[-1], self.nb_inputs, last_list, True)

    def __call__(self, traj_in, scale_in, pose_in, ego_in, traj_gt, scale_gt):
       
        traj_scale_in = F.concat((traj_in, scale_in), axis=2)
        traj_scale_gt = F.concat((traj_gt, scale_gt), axis=2)

        h_pos = self.pos_encoder(traj_scale_in)
        h_ego =  self.ego_encoder(ego_in)
        h = F.concat((h_pos, h_ego), axis=1) 
        h = self.inter(h)
        h_pos = self.pos_decoder(h)
        pred_locations = self.last(h_pos)  
        pred_locations = F.swapaxes(pred_locations, 1, 2)       # [num_ped, pred_len, 3]
        loss = F.mean_squared_error(pred_locations, traj_scale_gt)

        return pred_locations[:,:,:2], loss

class CNN_ego(chainer.Chain):
    """
    Baseline: feeds locations and poses
    """
    def __init__(self,channel_list, dc_channel_list, ksize_list,
                 dc_ksize_list, inter_list, last_list, pad_list):
        super(CNN_ego, self).__init__()
        if len(ksize_list) > 0 and len(dc_ksize_list) == 0:
            dc_ksize_list = ksize_list
        with self.init_scope():
            self.nb_inputs = 2
            self.pos_encoder = Encoder(self.nb_inputs, channel_list, ksize_list, pad_list)
            self.ego_encoder = Encoder(24, channel_list, ksize_list, pad_list)
            self.pos_decoder = Decoder(dc_channel_list[-1], dc_channel_list, dc_ksize_list[::-1])
            self.inter = Conv_Module(channel_list[-1]*2, dc_channel_list[0], inter_list)
            self.last = Conv_Module(dc_channel_list[-1], self.nb_inputs, last_list, True)

    def __call__(self, traj_in, scale_in, pose_in, ego_in, traj_gt, scale_gt):
       
        h_pos = self.pos_encoder(traj_in)
        h_ego =  self.ego_encoder(ego_in)
        h = F.concat((h_pos, h_ego), axis=1) 
        h = self.inter(h)
        h_pos = self.pos_decoder(h)
        pred_locations = self.last(h_pos)  
        pred_locations = F.swapaxes(pred_locations, 1, 2)       # [num_ped, pred_len, 3]
        loss = F.mean_squared_error(pred_locations, traj_gt)

        return pred_locations[:,:,:2], loss

class CNN_ego_pose_scale(chainer.Chain):
    """
    Baseline: feeds locations and poses
    """
    def __init__(self,channel_list, dc_channel_list, ksize_list,
                 dc_ksize_list, inter_list, last_list, pad_list):
        super(CNN_ego_pose_scale, self).__init__()
        if len(ksize_list) > 0 and len(dc_ksize_list) == 0:
            dc_ksize_list = ksize_list
        with self.init_scope():
            self.nb_inputs = 3
            self.pos_encoder = Encoder(self.nb_inputs, channel_list, ksize_list, pad_list)
            self.pose_encoder = Encoder(36, channel_list, ksize_list, pad_list)
            self.ego_encoder = Encoder(24, channel_list, ksize_list, pad_list)
            self.pos_decoder = Decoder(dc_channel_list[-1], dc_channel_list, dc_ksize_list[::-1])
            self.inter = Conv_Module(channel_list[-1]*3, dc_channel_list[0], inter_list)
            self.last = Conv_Module(dc_channel_list[-1], self.nb_inputs, last_list, True)

    def __call__(self, traj_in, scale_in, pose_in, ego_in, traj_gt, scale_gt):

        traj_scale_in = F.concat((traj_in, scale_in), axis=2)
        traj_scale_gt = F.concat((traj_gt, scale_gt), axis=2)

        h_pos = self.pos_encoder(traj_scale_in)
        h_pose = self.pose_encoder(pose_in)
        h_ego =  self.ego_encoder(ego_in)
        h = F.concat((h_pos, h_pose, h_ego), axis=1) 
        h = self.inter(h)
        h_pos = self.pos_decoder(h)
        pred_locations = self.last(h_pos)  
        pred_locations = F.swapaxes(pred_locations, 1, 2)       # [num_ped, pred_len, 3]
        loss = F.mean_squared_error(pred_locations, traj_scale_gt)

        return pred_locations[:,:,:2], loss