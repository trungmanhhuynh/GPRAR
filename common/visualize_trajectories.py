
'''
visualize predicted trajectories

Author: Manh Huynh
Last Update: 07/18/2020
'''

import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pac


IMAGES_DIR = "/home/manhh/github/datasets/JAAD/images"
TRAJ_FILE = "save/predictor/trajs.json"
TEST_DATA = "train_val_data/JAAD/predictor/val_medium.joblib"
OUTPUT_IMAGES_DIR = "./visualize_trajs"

def extract_parts(pose):

    # print(pose)
    Nose = pose[0:2]
    Neck = pose[3:5]
    RShoulder = pose[6:8]
    RElbow = pose[9:11]
    RWrist = pose[12:14]
    LShoulder = pose[15:17]
    LElbow = pose[18:20]
    LWrist = pose[21:23]
    MidHip = pose[24:26]
    RHip = pose[27:29]
    RKnee = pose[30:32]
    RAnkle = pose[33:35]
    LHip = pose[36:38]
    LKnee = pose[39:41]
    LAnkle = pose[42:44]
    REye = pose[45:47]
    LEye = pose[48:50]
    REar = pose[51:53]
    LEar = pose[54:56]
    LBigToe = pose[57:59]
    LSmallToe = pose[60:62]
    LHeel = pose[63:65]
    RBigToe = pose[66:68]
    RSmallToe = pose[69:71]
    RHeel = pose[72:74]
    # Background =   [pose['pose_keypoints_2d'][75], pose['pose_keypoints_2d'][76]]

    return Nose, Neck, RShoulder, RElbow, RWrist, LShoulder, LElbow, LWrist, MidHip, RHip, \
        RKnee, RAnkle, LHip, LKnee, LAnkle, REye, LEye, REar, LEar, LBigToe, \
        LSmallToe, LHeel, RBigToe, RSmallToe, RHeel  # , Background


def is_zero(point):
    if(point[0] == 0 and point[1] == 0):
        return True
    else:
        return False

def plot_pose(pose_keypoints_2d, person_id, ax):

    Nose, Neck, RShoulder, RElbow, RWrist, LShoulder, LElbow, LWrist, MidHip, RHip, \
        RKnee, RAnkle, LHip, LKnee, LAnkle, REye, LEye, REar, LEar, LBigToe, \
        LSmallToe, LHeel, RSmallToe, RBigToe, RHeel = extract_parts(pose_keypoints_2d)

    if(is_zero(Nose) == False and is_zero(Neck) == False):
        ax.plot([Nose[0], Neck[0]], [Nose[1], Neck[1]], '-', linewidth=3, color='C0')
    if(is_zero(Neck) == False and is_zero(RShoulder) == False):
        ax.plot([Neck[0], RShoulder[0]], [Neck[1], RShoulder[1]], '-', linewidth=3, color='C1')
    if(is_zero(RShoulder) == False and is_zero(RElbow) == False):
        ax.plot([RShoulder[0], RElbow[0]], [RShoulder[1], RElbow[1]], '-', linewidth=3, color='C1')
    if(is_zero(RElbow) == False and is_zero(RWrist) == False):
        ax.plot([RElbow[0], RWrist[0]], [RElbow[1], RWrist[1]], '-', linewidth=3, color='C1')
    if(is_zero(Neck) == False and is_zero(LShoulder) == False):
        ax.plot([Neck[0], LShoulder[0]], [Neck[1], LShoulder[1]], '-', linewidth=3, color='C2')
    if(is_zero(LShoulder) == False and is_zero(LElbow) == False):
        ax.plot([LShoulder[0], LElbow[0]], [LShoulder[1], LElbow[1]], '-', linewidth=3, color='C2')
    if(is_zero(LElbow) == False and is_zero(LWrist) == False):
        ax.plot([LElbow[0], LWrist[0]], [LElbow[1], LWrist[1]], '-', linewidth=3, color='C2')
    if(is_zero(Nose) == False and is_zero(REye) == False):
        ax.plot([Nose[0], REye[0]], [Nose[1], REye[1]], '-', linewidth=3, color='C3')  # 0-15
    if(is_zero(REye) == False and is_zero(REar) == False):
        ax.plot([REye[0], REar[0]], [REye[1], REar[1]], '-', linewidth=3, color='C3')  # 15-17
    if(is_zero(Nose) == False and is_zero(LEye) == False):
        ax.plot([Nose[0], LEye[0]], [Nose[1], LEye[1]], '-', linewidth=3, color='C4')  # 0-16
    if(is_zero(LEye) == False and is_zero(LEar) == False):
        ax.plot([LEye[0], LEar[0]], [LEye[1], LEar[1]], '-', linewidth=3, color='C4')  # 16-18
    if(is_zero(Neck) == False and is_zero(MidHip) == False):
        ax.plot([Neck[0], MidHip[0]], [Neck[1], MidHip[1]], '-', linewidth=3, color='C5')  # 1-8

    if(is_zero(MidHip) == False and is_zero(RHip) == False):
        ax.plot([MidHip[0], RHip[0]], [MidHip[1], RHip[1]], '-', linewidth=3, color='C6')  # 8-9
    if(is_zero(RHip) == False and is_zero(RKnee) == False):
        ax.plot([RHip[0], RKnee[0]], [RHip[1], RKnee[1]], '-', linewidth=3, color='C6')  # 9-10
    if(is_zero(RKnee) == False and is_zero(RAnkle) == False):
        ax.plot([RKnee[0], RAnkle[0]], [RKnee[1], RAnkle[1]], '-', linewidth=3, color='C6')  # 10-11
    if(is_zero(RAnkle) == False and is_zero(RHeel) == False):
        ax.plot([RAnkle[0], RHeel[0]], [RAnkle[1], RHeel[1]], '-', linewidth=3, color='C6')  # 11-24
    if(is_zero(RAnkle) == False and is_zero(RBigToe) == False):
        ax.plot([RAnkle[0], RBigToe[0]], [RAnkle[1], RBigToe[1]], '-', linewidth=3, color='C6')  # 11-22
    if(is_zero(RBigToe) == False and is_zero(RSmallToe) == False):
        ax.plot([RBigToe[0], RSmallToe[0]], [RBigToe[1], RSmallToe[1]], '-', linewidth=3, color='C6')  # 22-23

    if(is_zero(MidHip) == False and is_zero(LHip) == False):
        ax.plot([MidHip[0], LHip[0]], [MidHip[1], LHip[1]], '-', linewidth=3, color='C7')  # 8-12
    if(is_zero(LHip) == False and is_zero(LKnee) == False):
        ax.plot([LHip[0], LKnee[0]], [LHip[1], LKnee[1]], '-', linewidth=3, color='C7')  # 12-13
    if(is_zero(LKnee) == False and is_zero(LAnkle) == False):
        ax.plot([LKnee[0], LAnkle[0]], [LKnee[1], LAnkle[1]], '-', linewidth=3, color='C7')  # 13-14
    if(is_zero(LAnkle) == False and is_zero(LHeel) == False):
        ax.plot([LAnkle[0], LHeel[0]], [LAnkle[1], LHeel[1]], '-', linewidth=3, color='C7')  # 14-21
    if(is_zero(LAnkle) == False and is_zero(LBigToe) == False):
        ax.plot([LAnkle[0], LBigToe[0]], [LAnkle[1], LBigToe[1]], '-', linewidth=3, color='C7')  # 14-19
    if(is_zero(LBigToe) == False and is_zero(LSmallToe) == False):
        ax.plot([LBigToe[0], LSmallToe[0]], [LBigToe[1], LSmallToe[1]], '-', linewidth=3, color='C7')  # 19-20

    if(is_zero(LBigToe) == False and is_zero(LSmallToe) == False):
        ax.plot([LBigToe[0], LSmallToe[0]], [LBigToe[1], LSmallToe[1]], '-', linewidth=3, color='C7')  # 19-20

    # plot id
    ax.text(MidHip[0], MidHip[1], person_id, bbox=dict(facecolor='red', alpha=0.2))

    return ax

def plot_bbox(bbox, plt):

    # bbox = [tl_x, tl_y, br_x, br_y]

    bbox_w = bbox[2] - bbox[0]      # x bottom right - x top left
    bbox_h = bbox[3] - bbox[1]      # y bottom right - y top left

    rectangle = pac.Rectangle((bbox[0], bbox[1]), bbox_w, bbox_h, fill=False, ec="b")
    plt.gca().add_patch(rectangle)

    return plt


def plot_trajectory(gt_locations, ax, color='r', image_height=1080):

    ax.plot(gt_locations[:, 0], gt_locations[:, 1], 'r', label='observed')

    return ax


def visualize_trajs(image_width=1920, image_height=1080, obs_len=10, pred_len=10):
    """
        Visualize processed data: pose, gt_locations for a video 
        The directories IMAGES_DIR, LOCATION_DATA_DIR, POSE_ID_DATA_DIR must be exist
        and their structures must be the same as mentioned in README.md. 
        The resulted images will be written to OUTPUT_IMAGES_DIR
    """

    # read result data and ground-truth data
    with open(TRAJ_FILE, "r") as f:
        result_data = json.load(f)
        pred_traj = result_data['traj_pred']

    val_data = joblib.load(TEST_DATA)
    poses, gt_locations, bboxes = [], [], []
    video_names, image_names, person_ids = [], [], []
    for sample in val_data:
        poses.append(sample['poses'][obs_len - 1])
        gt_locations.append(sample['gt_locations'])
        # bboxes.append(sample['bboxes'][obs_len - 1])
        video_names.append(sample['video_names'][obs_len - 1])
        image_names.append(sample['image_names'][obs_len - 1])  # the image_name is at the last observed location.
        person_ids.append(sample['person_ids'][obs_len - 1])

    assert len(pred_traj) == len(video_names)

    for s_id in range(len(pred_traj)):

        print("Processing sample:{}, video_name:{}, image_name:{}".format(s_id, video_names[s_id], image_names[s_id]))

        if not os.path.exists(os.path.join(OUTPUT_IMAGES_DIR, video_names[s_id])):
            os.makedirs(os.path.join(OUTPUT_IMAGES_DIR, video_names[s_id]))

        # read image
        img = plt.imread(os.path.join(IMAGES_DIR, video_names[s_id], image_names[s_id]))
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(img, extent=[0, image_width, image_height, 0])

        # plot poses at the last observed frame
        # ax = plot_pose(poses[s_id], "", ax)

        gt_locations = np.array(gt_locations)
        pred_traj = np.array(pred_traj)

        # plot past trajectory
        ax.plot(gt_locations[s_id, :obs_len, 0], gt_locations[s_id, :obs_len, 1], color='b', linewidth=2.0, label='observed')
        ax.plot(gt_locations[s_id, -pred_len:, 0], gt_locations[s_id, -pred_len:, 1], 'r', linewidth=4.0, label='ground-truth')
        ax.plot(pred_traj[s_id, :, 0], pred_traj[s_id, :, 1], 'y', linewidth=2.0, label='predicted')

        # # plot bounding box (from ground truth)
        # with open(os.path.join(LOCATION_DATA_DIR, video_name, os.path.splitext(image_name)[0] + "_gt_locations.json"), "r") as f:
        #     location_data = json.load(f)

        # fig = plot_bbox(bboxes[s_id], fig)

        fig.savefig(os.path.join(OUTPUT_IMAGES_DIR, video_names[s_id], "s{}_".format(s_id) + image_names[s_id]))
        plt.close()


if __name__ == '__main__':

    visualize_trajs()
