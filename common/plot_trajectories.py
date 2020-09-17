import torch
import json
import joblib
import os
import numpy as np
import argparse

import matplotlib.pyplot as plt
from scipy import ndimage

def calculate_ade_fde(traj_gt, traj_pred):

    # traj_gt ~ numpy [batch_size, args.pred_len, 2] : absolute gt location
    # pred_locations ~ numpy[batch_size, args.pred_len, 2] : absolute pred location

    temp = (traj_pred - traj_gt)**2
    ade = np.sqrt(temp[:, 0] + temp[:, 1])
    ade = np.mean(ade)
    fde = np.sqrt(temp[-1, 0] + temp[-1, 1])
    fde = np.mean(fde)

    return ade, fde

def read_val_data(args):

    # with open(args.val_data, "r") as f:
    val_data = joblib.load(args.val_data)

    poses, gt_locations, bboxes = [], [], []
    video_names, image_names, person_ids = [], [], []
    for sample in val_data:
        poses.append(sample['poses'])
        gt_locations.append(sample['gt_locations'])
        video_names.append(sample['video_names'][args.obs_len - 1])
        image_names.append(sample['image_names'][args.obs_len - 1])  # the image_name is at the last observed location.
        person_ids.append(sample['person_ids'][args.obs_len - 1])

    # convert to numpy
    poses = np.asarray(poses)
    observed_pose = poses[:, :args.obs_len, :]
    gt_locations = np.asarray(gt_locations)

    return poses, observed_pose, gt_locations, video_names, image_names


def calc_ade_per_keypoint(args, observed_pose, pred_traj, gt_locations, video_names, image_names):
    '''
    '''

    args.obs_len = 10
    num_keypoints = 25
    num_samples = observed_pose.shape[0]
    num_keypoints_per_sample = [0] * num_samples
    ade_per_sample = [0] * num_samples

    # calculate number of keypoints for each sample
    for i in range(num_samples):
        ade, fde = calculate_ade_fde(gt_locations[i, -args.pred_len:, :], pred_traj[i])
        ade_per_sample[i] = ade
        for t in range(args.obs_len):
            for k in range(num_keypoints):
                if(observed_pose[i, t, 2 * k] != 0 and observed_pose[i, t, 2 * k + 1] != 0):
                    num_keypoints_per_sample[i] += 1

    ade_per_sample = np.asarray(ade_per_sample)
    num_keypoints_per_sample = np.asarray(num_keypoints_per_sample)
    sort_index = np.argsort(num_keypoints_per_sample)

    return num_keypoints_per_sample, ade_per_sample, sort_index


def visualize_trajs(args, observed_pose, pred_traj, gt_locations, video_names, image_names,
                    num_keypoints_per_sample, ade_per_sample,
                    plot_pose=False):
    """
        Visualize processed data: pose, gt_locations for a video
        The directories args.images_dir, LOCATION_DATA_DIR, POSE_ID_DATA_DIR must be exist
        and their structures must be the same as mentioned in README.md.
        The resulted images will be written to args.output_traj_dir
    """

    # read result data and ground-truth data

    num_samples = observed_pose.shape[0]
    for i in range(num_samples):

        # only plot sample that ade >= 40
        if(ade_per_sample[i] < 20):
            continue

        print("Processing sample:{}, video_name:{}, image_name:{}".format(i, video_names[i], image_names[i]))

        if not os.path.exists(os.path.join(args.output_traj_dir, video_names[i])):
            os.makedirs(os.path.join(args.output_traj_dir, video_names[i]))

        # read image
        img = plt.imread(os.path.join(args.images_dir, video_names[i], image_names[i]))
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(img, extent=[0, args.image_width, args.image_height, 0])

        # plot poses at the last observed frame
        if(plot_pose):
            ax = plot_pose(poses[i], "", ax)

        # plot past trajectory
        ax.plot(gt_locations[i, :args.obs_len, 0], gt_locations[i, :args.obs_len, 1], color='b', linewidth=2.0, label='observed')
        ax.plot(gt_locations[i, -args.pred_len:, 0], gt_locations[i, -args.pred_len:, 1], 'r', linewidth=2.0, label='ground-truth')
        ax.plot(pred_traj[i, :, 0], pred_traj[i, :, 1], 'y', linewidth=2.0, label='predicted')
        ax.legend()
        ax.text(50, 50, "ade = {:.2f}".format(ade_per_sample[i]), fontsize=12, color='r')
        ax.text(50, 150, "#visible_keypoints = {}/250".format(num_keypoints_per_sample[i]), fontsize=12, color='r')

        # # plot bounding box (from ground truth)
        # with open(os.path.join(LOCATION_DATA_DIR, video_name, os.path.splitext(image_name)[0] + "_gt_locations.json"), "r") as f:
        #     location_data = json.load(f)
        # fig = plot_bbox(bboxes[i], fig)

        fig.savefig(os.path.join(args.output_traj_dir, video_names[i], "s{}_".format(i) + image_names[i]))
        plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_len', type=int, default=10)
    parser.add_argument('--pred_len', type=int, default=10)
    parser.add_argument('--image_width', type=int, default=1920)
    parser.add_argument('--image_height', type=int, default=1080)
    parser.add_argument('--traj_file', type=str, default="",
                        help='load predicted trajectory from json file')
    parser.add_argument('--val_data', type=str, default="train_val_data/JAAD/predictor/val_medium.joblib",
                        help='load test data consisting of ground truth trajectory')
    parser.add_argument('--images_dir', type=str, default="/home/manhh/github/datasets/JAAD/images",
                        help='output trajectory directory')
    parser.add_argument('--output_traj_dir', type=str, default="traj_results",
                        help='output trajectory directory')
    args = parser.parse_args()

    # read trajectory result data
    with open(args.traj_file, "r") as f:
        result_data = json.load(f)
    pred_traj = result_data['traj_pred']
    pred_traj = np.asarray(pred_traj)
    num_samples = pred_traj.shape[0]

    # read validation data for ground truth
    poses, observed_pose, gt_locations, video_names, image_names = read_val_data(args)
    assert num_samples == len(video_names)
    print("Number of samples :", num_samples)

    # calculate ade for each sample
    num_keypoints_per_sample, ade_per_sample, sort_index = calc_ade_per_keypoint(args, observed_pose, pred_traj, gt_locations, video_names, image_names)

    # plot trajectories
    visualize_trajs(args, observed_pose, pred_traj, gt_locations, video_names, image_names,
                    num_keypoints_per_sample, ade_per_sample,
                    plot_pose=False)
