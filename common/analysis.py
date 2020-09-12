import torch
import json
import joblib
import numpy as np
import argparse

import matplotlib.pyplot as plt
from scipy import ndimage


def calculate_ade_fde(traj_gt, traj_pred):

    # traj_gt ~ numpy [batch_size, pred_len, 2] : absolute gt location
    # pred_locations ~ numpy[batch_size, pred_len, 2] : absolute pred location

    temp = (traj_pred - traj_gt)**2
    ade = np.sqrt(temp[:, 0] + temp[:, 1])
    ade = np.mean(ade)

    fde = np.sqrt(temp[-1, 0] + temp[-1, 1])
    fde = np.mean(fde)

    return ade, fde


def calc_percent_occlusion(pose, keypoints=25, feature=3):

    # pose shape (obs_len, keypoints*3)

    obs_len, points = pose.shape     # points = keypoints*3
    pose = pose.reshape((obs_len, keypoints, feature))
    n_missing_pts = 0

    missing_kpt_counts = [0] * keypoints
    for t in range(obs_len):
        for k in range(keypoints):
            if(pose[t][k][0] == 0 and pose[t][k][1] == 0):
                n_missing_pts += 1
                missing_kpt_counts[k] += 1

    return missing_kpt_counts, n_missing_pts / (keypoints * obs_len)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_len', type=int, default=10)
    parser.add_argument('--pred_len', type=int, default=10)

    parser.add_argument('--traj_file', type=str, default="",
                        help='load predicted trajectory from json file')
    parser.add_argument('--val_data', type=str, default="train_val_data/JAAD/predictor/val_small.joblib",
                        help='load test data consisting of ground truth trajectory')

    args = parser.parse_args()

    # make sure that args.val_data is the one used to generate traj_file
    # this can be done by running "test.py"
    with open(args.traj_file, "r") as f:
        result_data = json.load(f)

    pred_traj = result_data['traj_pred']

    # with open(args.val_data, "r") as f:
    val_data = joblib.load(args.val_data)

    poses, locations, bboxes = [], [], []
    video_names, image_names, person_ids = [], [], []
    for sample in val_data:
        poses.append(sample['poses'])
        locations.append(sample['gt_locations'])
        video_names.append(sample['video_names'][args.obs_len - 1])
        image_names.append(sample['image_names'][args.obs_len - 1])  # the image_name is at the last observed location.
        person_ids.append(sample['person_ids'][args.obs_len - 1])

    assert len(pred_traj) == len(video_names)
    print("Number of samples :", len(pred_traj))

    # convert to numpy
    poses = np.asarray(poses)
    observed_pose = poses[:, :args.obs_len, :]
    locations = np.asarray(locations)
    gt_locations = locations[:, -args.pred_len:, :]
    pred_traj = np.asarray(pred_traj)
    num_samples = len(pred_traj)

    # init variables
    ade_r0, fde_r0, num_r0 = 0, 0, 0            # r0: 0-> 0.1 %
    ade_r1, fde_r1, num_r1 = 0, 0, 0            # r10: 0.1-> 0.2 %
    ade_r2, fde_r2, num_r2 = 0, 0, 0            # r10: 0.2-> 0.3 %
    ade_r3, fde_r3, num_r3 = 0, 0, 0            # r10: 0.3-> 0.4 %
    ade_r4, fde_r4, num_r4 = 0, 0, 0            # r10: 0.4->  %
    test_ade, test_fde = 0, 0

    ade_ms_kpts = [0] * 25
    fde_ms_kpts = [0] * 25
    ms_kpts = [0] * 25

    r3_missing_kpts = [0] * 25
    r4_missing_kpts = [0] * 25

    occlusions = {'occlusion_percent': [], 'ade': [], 'fde': []}

    for i in range(num_samples):

        ade, fde = calculate_ade_fde(gt_locations[i], pred_traj[i])
        missing_kpt_counts, occlusion_percent = calc_percent_occlusion(observed_pose[i], keypoints=25, feature=3)

        occlusions['occlusion_percent'].append(occlusion_percent)
        occlusions['ade'].append(ade)
        occlusions['fde'].append(fde)

        # ade/fde by occlusion percentage
        if(occlusion_percent >= 0 and occlusion_percent < 0.1):
            ade_r0 += ade
            fde_r0 += fde
            num_r0 += 1
        if(occlusion_percent >= 0.1 and occlusion_percent < 0.2):
            ade_r1 += ade
            fde_r1 += fde
            num_r1 += 1
        elif(occlusion_percent >= 0.2 and occlusion_percent < 0.3):
            ade_r2 += ade
            fde_r2 += fde
            num_r2 += 1
        elif(occlusion_percent >= 0.3 and occlusion_percent < 0.4):
            ade_r3 += ade
            fde_r3 += fde
            num_r3 += 1
            for k, num_ms in enumerate(missing_kpt_counts):
                if(missing_kpt_counts[k] > 0):
                    r3_missing_kpts[k] += 1

        elif(occlusion_percent >= 0.4 and occlusion_percent < 0.5):
            ade_r4 += ade
            fde_r4 += fde
            num_r4 += 1
            for k, num_ms in enumerate(missing_kpt_counts):
                if(missing_kpt_counts[k] > 0):
                    r4_missing_kpts[k] += 1

        test_ade += ade
        test_fde += fde

        # ade/fde by missing human keypoints:
        for k, num_ms in enumerate(missing_kpt_counts):
            if(missing_kpt_counts[k] > 0):
                ade_ms_kpts[k] += ade
                fde_ms_kpts[k] += fde
                ms_kpts[k] += 1

    # plot ade by occlusion percentage
    occlusions['occlusion_percent'] = np.asarray(occlusions['occlusion_percent'])
    occlusions['ade'] = np.asarray(occlusions['ade'])
    occlusions['fde'] = np.asarray(occlusions['fde'])
    sort_index = np.argsort(occlusions['occlusion_percent'])

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(occlusions['occlusion_percent'][sort_index], occlusions['ade'][sort_index], color='b', linewidth=2.0, label='ade')
    plt.xlabel('occlusion (%)')
    plt.ylabel('ade (pixels)')
    fig.savefig("occlusion.png")
    plt.close()

    print("num_r0= {}, ade_r0={}, fde_r0={}".format(num_r0, ade_r0 / num_r0, fde_r0 / num_r0))
    print("num_r1= {}, ade_r1={}, fde_r1={}".format(num_r1, ade_r1 / num_r1, fde_r1 / num_r1))
    print("num_r2= {}, ade_r2={}, fde_r2={}".format(num_r2, ade_r2 / num_r2, fde_r2 / num_r2))
    print("num_r3= {}, ade_r3={}, fde_r3={}".format(num_r3, ade_r3 / num_r3, fde_r3 / num_r3))
    print("num_r4= {}, ade_r4={}, fde_r4={}".format(num_r4, ade_r4 / num_r4, fde_r4 / num_r4))
    print("ade = {}, fde ={}".format(test_ade / num_samples, test_fde / num_samples))

    for k, num in enumerate(ms_kpts):
        if(ms_kpts[k] >= 0):
            ade_ms_kpts[k] /= ms_kpts[k]

    # plot ade by missing keypoints
    print("ade_ms_kpts", ade_ms_kpts)
    x = np.arange(len(ade_ms_kpts))
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.bar(x, ade_ms_kpts, width=0.5, label='ade')
    ax.set_ylabel('ADE')
    ax.set_xlabel('ith keypoint')

    ax.set_title('Impacts of missing a keypoint')
    ax.set_xticks(x)
    # ax.set_xticklabels(labels)
    ax.legend()
    fig.savefig("keypoint_impact_ade.png")
    plt.close()

    # plot samples that missing a keypoints in r3 and r4

    print("r3_missing_kpts", r3_missing_kpts)
    print("r4_missing_kpts", r4_missing_kpts)

    x = np.arange(len(r3_missing_kpts))
    width = 0.35
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.bar(x - width / 2, [i / num_r3 * 100 for i in r3_missing_kpts], width, label='30-40% occ.')
    ax.bar(x + width / 2, [i / num_r4 * 100 for i in r4_missing_kpts], width, label='>40% occ.')

    ax.set_ylabel('# samples')
    ax.set_xlabel('ith keypoint')

    #ax.set_title('Impacts of missing a keypoint')
    ax.set_xticks(x)
    # ax.set_xticklabels(labels)
    ax.legend()
    fig.savefig("r3vsr4.png")
    plt.close()
