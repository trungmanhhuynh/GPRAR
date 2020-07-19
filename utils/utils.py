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

    # transfer to cpu 
    traj_gt = traj_gt.data.cpu()
    traj_pred = traj_pred.data.cpu()

    # denormalize 
    traj_pred_abs = std_denormalize(traj_pred, mean, var)
    traj_gt_abs = std_denormalize(traj_gt, mean, var)



    temp = (traj_pred_abs - traj_gt_abs)**2
    ade = torch.sqrt(temp[:,:,0] + temp[:,:,1])
    ade = torch.mean(ade)

    fde = torch.sqrt(temp[:,-1,0] + temp[:,-1,1])
    fde = torch.mean(fde)


    return ade, fde


def save_traj_json(traj_dict, traj_pred, video_names, image_names, person_ids, mean, var):
    """
        Save predicted trajectories to json file
        Input: 
            traj_gt (tensor ~ (batch_size, pred_len, 2)):ground-truth trajectories
            traj_pred (tensor ~ (batch_size, pred_len, 2)): predicted trajectories
            video_names (tuple ~ (batch_size)): video name of each trajectory
            image_names (tuple ~ (batch_size)): image name of each trajectory
            person_ids (tuple ~ (batch_size)): pedestrian id of each trajectory    
        Output: 
            traj_dict : dictionary 
    """

    # transfer to cpu 
    traj_pred = traj_pred.data.cpu()

    # denormalize 
    traj_pred_abs = std_denormalize(traj_pred, mean, var)  # use center position's mean


    traj_dict['video_names'].append(list(video_names))
    traj_dict['image_names'].append(list(image_names))
    traj_dict['person_ids'].append(list(person_ids))
    traj_dict['traj_pred'].append(traj_pred_abs.tolist())



    return traj_dict