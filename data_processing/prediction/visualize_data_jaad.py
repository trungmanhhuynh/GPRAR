import cv2
import os
import matplotlib.pyplot as plt
import argparse
import pickle
import numpy as np


def plot_pose(args, img, pose):
    '''
        pose (C,V,1)
    '''
    H, W, c = img.shape
    scale_factor = 2 * H / 1080

    # reshape pose to list C*V
    C, V, _ = pose.shape
    assert C == 3
    assert V == args.num_keypoint

    if args.num_keypoint == 18:
        edge = [(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12, 11),
                (10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
                (0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
    else:
        print("have not supported 25 keypoints yet")
        exit(-1)

    for i, j in edge:
        xi = pose[0, i, 0]
        yi = pose[1, i, 0]
        xj = pose[0, j, 0]
        yj = pose[1, j, 0]
        if xi + yi == 0 or xj + yj == 0:
            continue
        else:
            xi = int((xi + 0.5) * W)
            yi = int((yi + 0.5) * H)
            xj = int((xj + 0.5) * W)
            yj = int((yj + 0.5) * H)
        cv2.line(img, (xi, yi), (xj, yj), (0, 128, 0),
                 int(np.ceil(2 * scale_factor)))

    return img


def plot_sample(args, location, pose, video_name, image_name, ith_sample):
    '''
        Inputs:
            + location (T, 2)
            + pose (C, T, V, 1)
            + video_name (string)
            + image_name (string) : first frame of the sequence
    '''
    output_folder = os.path.join(args.output_path, ith_sample, video_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_list = os.listdir(os.path.join(args.image_path, video_name))
    st_frame_nb = image_list.index(image_name)

    # plot on each image
    traj_len = location.shape[0]
    for t in range(traj_len):

        # plot image
        img = cv2.imread(os.path.join(args.image_path, video_name, image_list[st_frame_nb + t]), 1)
        img = img.astype(np.uint8)

        # image resize
        H, W, c = img.shape
        # print(img.shape)
        img = cv2.resize(img, (args.image_height * W // H // 2, args.image_height // 2))
        H, W, c = img.shape

        # plot location
        # de-normalize data
        x = int((location[t, 0] + 0.5) * W)
        y = int((location[t, 1] + 0.5) * H)
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

        # img[x, y, :] = [0, 0, 255]            # red dot

        # plot pose
        img = plot_pose(args, img, pose[:, t, :, :])

        # save figure
        filename = os.path.join(output_folder, image_list[st_frame_nb + t])
        cv2.imwrite(filename, img)


def visualize(args):
    '''
    '''

    # load data
    with open(args.data_file, 'rb') as f:
        location, pose, _ = pickle.load(f)
    # load label
    with open(args.info_file, 'rb') as f:
        video_name, image_name, _, _ = pickle.load(f)

    # locations (N, T, 2)
    # poses (N, C, T, V, 1)
    if(args.debug):
        location = location[:10]
        pose = pose[:10]
        video_name = video_name[0:10]
        image_name = image_name[0:10]

    num_sample = location.shape[0]
    for i in range(0, num_sample, 100):
        print("Processing sample:", i)
        plot_sample(args, location[i], pose[i], video_name[i], image_name[i], "sample" + str(i))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_file', default='../datasets/processed_data/prediction/jaad/train_data.pkl')
    parser.add_argument(
        '--info_file', default='../datasets/processed_data/prediction/jaad/train_metadata.pkl')
    parser.add_argument(
        '--image_path', default='../datasets/JAAD/images')
    parser.add_argument(
        '--output_path', default='temp')
    parser.add_argument(
        '--plot_location', action="store_true", default=True)
    parser.add_argument(
        '--plot_pose', action="store_true", default=True)
    parser.add_argument(
        '--image_width', type=int, default=1080)
    parser.add_argument(
        '--image_height', type=int, default=1080)
    parser.add_argument(
        '--num_keypoint', type=int, default=18)
    parser.add_argument(
        '--debug', action="store_true", default=False, help='debug mode')

    args = parser.parse_args()

    visualize(args)
