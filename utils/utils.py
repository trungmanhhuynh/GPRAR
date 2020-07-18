import torch
import numpy as np

def calc_mean_variance(data):

    # data shape must be [num_traj, traj_len, N], where N is feature size
    # data type is torch tensor

    #data = torch.cat(data, axis = 0)  # (num_traj*traj_len, N)
    num_samples, traj_len, feature_size = data.size()
    data = data.view(num_samples* traj_len, feature_size )
    mean = data.mean(axis = 0)
    var = data.std(axis = 0)

    return mean, var


def std_normalize(data_in, mean, var):

    # data_in ~ [num_ped, traj_len, 2]
    # mean, var ~ [1,2]
    data_out = (data_in - mean)/var
    return data_out


def std_denormalize(data_in, mean, var):

    # data_in ~ [num_ped, traj_len, 2]
    # mean, var ~ [1,2]
    data_out = data_in*var + mean
    return data_out

def calculate_ade_fde(traj_gt, traj_pred, mean, var):


    # pred_locations ~ tensor [batch_size, pred_len, 2]
    # pred_locations ~ [batch_size, pred_len, 2]

    # denormalize 
    traj_pred_abs = std_denormalize(traj_pred, mean, var)
    pred_gt_abs = std_denormalize(traj_gt, mean, var)



    temp = (traj_pred_abs - pred_gt_abs)**2
    ade = torch.sqrt(temp[:,:,0] + temp[:,:,1])
    ade = torch.mean(ade)

    fde = torch.sqrt(temp[:,-1,0] + temp[:,-1,1])
    fde = torch.mean(fde)


    return ade, fde
