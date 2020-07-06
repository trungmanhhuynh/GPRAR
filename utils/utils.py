
import torch


def calc_mean_variance(data):

    #data shape must be [num_traj, traj_len, N], where N is feature size

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
