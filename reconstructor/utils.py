import torch
import os
import matplotlib.pyplot as plt
from common.plot_utils import plot_pose

def calc_mean_variance(data):

    # data shape must be [num_traj, traj_len, N], where N is feature size
    # data type is torch tensor

    # data = torch.cat(data, axis = 0)  # (num_traj*traj_len, N)
    num_samples, traj_len, feature_size = data.size()
    data = data.view(num_samples * traj_len, feature_size)
    mean = data.mean(axis=0)
    var = data.std(axis=0)

    return mean, var


def std_normalize(data_in, mean, var):

    # data_in ~ [num_ped, traj_len, 2]
    # mean, var ~ [1,2]
    data_out = (data_in - mean) / var
    return data_out


def std_denormalize(data_in, mean, var):

    # data_in ~ [num_ped, traj_len, 2]
    # mean, var ~ [1,2]
    data_in = data_in.data.cpu()

    data_out = data_in * var + mean
    return data_out

def calculate_ade(gt_poses, pred_poses):

    # pred_locations ~ tensor [batch_size, pred_len, 2]
    # pred_locations ~ [batch_size, pred_len, 2]

    temp = (gt_poses - pred_poses)**2
    ade = torch.sqrt(temp[:, :, 0] + temp[:, :, 1])
    ade = torch.mean(ade)

    return ade


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


def plot_results(gt_poses,
                 pred_poses,
                 bboxes,
                 epoch,
                 video_name,
                 image_name,
                 image_dir,
                 result_dir,
                 obs_len=10,
                 image_width=1920,
                 image_height=1080):
    """

    """

    num_samples = pred_poses.shape[0]
    for i in range(1):

        # plot image
        # img = plt.imread(os.path.join(image_dir, video_name[i], image_name[i]))

        x1 = int(bboxes[i, obs_len - 1, 0])
        y1 = int(bboxes[i, obs_len - 1, 1])
        x2 = int(bboxes[i, obs_len - 1, 2])
        y2 = int(bboxes[i, obs_len - 1, 3])

        fig, ax = plt.subplots(nrows=1, ncols=2)
        # ax[0].imshow(img)
        # ax[1].imshow(img)

        # plot pose
        plot_pose(gt_poses[i, obs_len - 1, :].tolist(), "", ax[0])
        plot_pose(pred_poses[i, obs_len - 1, :].tolist(), "", ax[1])

        # ax[0].set_xlim(x1, x2)
        # ax[0].set_ylim(y1, y2)
        # ax[1].set_xlim(x1, x2)
        # ax[1].set_ylim(y1, y2)

        fig.savefig(os.path.join(result_dir, "e_{}_s_{}_v_{}_".format(epoch, i, video_name[i]) + image_name[i]))
        plt.close()

    print("saved samples to:", result_dir)
