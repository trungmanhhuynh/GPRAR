import os
import json
import numpy as np
import pickle
from itertools import islice
from sklearn.impute import KNNImputer
from numpy.lib.format import open_memmap
import argparse
import csv


def chunks(data, traj_len=20, slide=1):
    # it = iter(data)
    for i in range(0, len(data), slide):
        yield list(islice(data, i, i + traj_len))


def impute_poses(args, poses):
    """
            interplote poses
            input: poses ~ [traj_len, 75]  : 75 is 25*3, 25 keypoints, 3: x,y,c
            ouput: imputed_poses ~ [traj_len, 75]
    """

    valid = True

    imputer = KNNImputer(missing_values=0, n_neighbors=5, weights="uniform")
    imputed_poses = imputer.fit_transform(poses)

    if (imputed_poses.shape[1] != 54):
        imputed_poses = poses  # return the original data
        valid = False  # return -1 if shape is not right

    return imputed_poses, valid


def generate_samples(video_data, video_name, args, mode):
    """
        Input:
            + video_data: a dictionary
                'image_name'(str):{
                        'people': [
                                {'person_id': (str)
                                'pose'  : list}]}
        Output:
            extract samples for each video. Each sample has the following dict structure:
            {
                    'video_names': [traj_len]
                    'image_names': [traj_len]
                    'person_ids': [traj_len]
                    'poses': list ~[traj_len, 75]
                    'locations': list ~[traj_len, 2]
                    'bboxes':  list ~ [traj_len, 4]
            }
    """
    pose_seqs, video_names, image_name_out = [], [], []
    action_label_out, action_index_out = [], []
    location_seqs, gridflow_seqs = [], []
    bbox_seqs = []

    # read video_info for frame width and height
    video_info = {}
    with open(args.video_info_fn, 'r') as data:
        for line in csv.reader(data):
            video_info[line[0]] = [line[1], line[2]]

    # find list of pedestrian id
    id_list = []
    for image_name in video_data:
        for person in video_data[image_name]["people"]:
            if person['person_id'] not in id_list:
                id_list.append(person['person_id'])

    # extract trajectory for each pedestrian
    for pid in id_list:

        # extract whole trajectory of a pedestrian in video
        long_poses, long_image_names, long_action_indexes, long_action_labels = [], [], [], []
        long_locations, long_gridflow = [], []
        for image_name in video_data:
            for person in video_data[image_name]["people"]:
                if person['person_id'] == pid:
                    long_locations.append(person['location'])
                    long_poses.append(person['pose'])
                    long_image_names.append(image_name)
                    long_action_labels.append(person['action_label'])
                    long_action_indexes.append(person['action_index'])
                    long_gridflow.append(person['gridflow'])

        # split trajectories into chunk of pre-defined trajectory length
        for locations, pose, gridflow, image_names, action_labels, action_indexes in zip(
                chunks(long_locations, traj_len=args.traj_len, slide=args.slide),
                chunks(long_poses, traj_len=args.traj_len, slide=args.slide),
                chunks(long_gridflow, traj_len=args.traj_len, slide=args.slide),
                chunks(long_image_names, traj_len=args.traj_len, slide=args.slide),
                chunks(long_action_labels, traj_len=args.traj_len, slide=args.slide),
                chunks(long_action_indexes, traj_len=args.traj_len, slide=args.slide)):

            # skip if trajectory is short
            if len(pose) < args.traj_len:
                continue

            # skip of the label is not the same for all poses in video
            consistent_label = True
            for action_index in long_action_indexes:
                if action_index != long_action_indexes[0]:
                    consistent_label = False
            if not consistent_label:
                continue

            # check if trajectory is continous by extracting frame number [-11:-4] from names
            # this applies for image names with format: 0000x.png  or {:05d}.png
            gap = abs(int(image_names[0][-9:-4]) - int(image_names[-1][-9:-4]))
            if gap > args.traj_len:
                continue

            # get pose data
            pose = np.array(pose)  # (traj_len, 54)
            pose[:, 0::3] = pose[:, 0::3] / float(video_info[video_name][0])  # normalize x by frame width
            pose[:, 1::3] = pose[:, 1::3] / float(video_info[video_name][1])  # normalize y by frame height
            pose[:, 0::3][pose[:, 2::3] == 0] = 0
            pose[:, 1::3][pose[:, 2::3] == 0] = 0

            if mode == 'test':
                if 0 not in pose:
                    continue

            # get pose data
            if mode == 'train':
                if args.obs_type == 'impute' or args.obs_type == 'gt':
                    pose, valid = impute_poses(args, pose)
            if mode == 'val':
                if args.obs_type == 'impute':
                    pose, valid = impute_poses(args, pose)
                if args.obs_type == 'gt':
                    # pose, valid = impute_poses(args, pose)
                    if 0 in pose:
                        continue

            # get gt location data
            locations = np.array(locations)  # (traj_len, 2)
            locations[:, 0] = locations[:, 0] / float(video_info[video_name][0])
            locations[:, 1] = locations[:, 1] / float(video_info[video_name][1])

            # calculate bounding box
            bbox = np.zeros((args.traj_len, 4))
            for t in range(args.traj_len):
                xmin, xmax, ymin, ymax = 20000, -1, 20000, -1
                for i in range(0, 18):
                    if pose[t, 3 * i + 2] != 0:
                        if pose[t, 3 * i] <= xmin:
                            xmin = pose[t, 3 * i]
                        if pose[t, 3 * i] >= xmax:
                            xmax = pose[t, 3 * i]
                        if pose[t, 3 * i + 1] <= ymin:
                            ymin = pose[t, 3 * i + 1]
                        if pose[t, 3 * i + 1] >= ymax:
                            ymax = pose[t, 3 * i + 1]

                if (xmax - xmin == 0) or (ymax - ymin == 0):
                    continue
                bbox[t, :] = [xmin, ymin, xmax, ymax]

            # add to sample list
            location_seqs.append(locations.tolist())
            pose_seqs.append(pose.tolist())
            gridflow_seqs.append(gridflow)
            bbox_seqs.append(bbox.tolist())
            video_names.append(video_name)
            image_name_out.append(image_names[0])  # keep first image name for meta-data
            action_label_out.append(action_labels[0])
            action_index_out.append(action_indexes[0])

    return location_seqs, pose_seqs, gridflow_seqs, bbox_seqs, video_names, image_name_out, action_label_out, action_index_out


def generate_data(args, mode):
    """
        mode: train of val
        output:
            - fp: numpy array of shape (N, C, T, V, 1) --> written to *.npy file
            - label: list of N samples --> writting to *.json file
    """
    video_dir = os.listdir(os.path.join(args.pose_path))
    if mode == "train":
        video_dir = video_dir[:int(len(video_dir) * 0.8)]  # 80% of data for training
    else:  # val
        video_dir = video_dir[-int(len(video_dir) * 0.2):]  # 20% of data for val/test
    if args.debug:
        video_dir = video_dir[:10]  # use 10 videos for debug

    all_pose_seqs, all_video_names, all_image_names = [], [], []
    all_action_labels, all_action_indexes = [], []
    all_gridflow_seqs, all_location_seqs = [], []
    all_bbox_seqs = []
    for video_name in video_dir:
        print("processing video: ", video_name)

        # gather data for each video
        video_data = {}
        for frame_number in range(len(os.listdir(os.path.join(args.pose_path, video_name))) - 1):
            # -1 because there is no gridflow data for the last frame

            with open(os.path.join(args.location_path, video_name, "{:05d}_locations.json".format(frame_number)),
                      "r") as f:
                location_data = json.load(f)
            with open(os.path.join(args.pose_path, video_name, "{:05d}_keypoints.json".format(frame_number)), "r") as f:
                pose_data = json.load(f)
            with open(os.path.join(args.gridflow_path, video_name, "{:05d}_gridflow.json".format(frame_number)),
                      "r") as f:
                gridflow_data = json.load(f)

            image_name = "{:05d}.png".format(frame_number)
            video_data[image_name] = {'people': []}
            for ped in pose_data['people']:
                if ped['person_id'][0] == -1:
                    continue  # skip ped withput id

                temp_location = []
                for p_in_loc in location_data['people']:
                    if ped['person_id'] == p_in_loc['person_id']:
                        temp_location = p_in_loc['center']

                if not temp_location:
                    continue

                video_data[image_name]['people'].append({'person_id': ped['person_id'],
                                                         'location': temp_location,
                                                         'pose': ped['pose_keypoints_2d'],
                                                         'gridflow': gridflow_data[str(int(frame_number))],
                                                         'action_label': ped['action_label'],
                                                         'action_index': ped['action_index']
                                                         })

        # extract trajectory within a video
        location_seqs, pose_seqs, gridflow_seqs, bbox_seqs, video_names, \
        image_names, action_labels, action_indexes = generate_samples(video_data, video_name, args, mode)

        all_location_seqs.append(location_seqs)
        all_pose_seqs.append(pose_seqs)
        all_gridflow_seqs.append(gridflow_seqs)
        all_bbox_seqs.append(bbox_seqs)
        all_video_names.append(video_names)
        all_image_names.append(image_names)
        all_action_labels.append(action_labels)
        all_action_indexes.append(action_indexes)

    # convert data to numpy array of shape (N, C, T, V, 1)
    all_location_seqs = np.array(sum(all_location_seqs, []))  # (N, T, 2)
    all_pose_seqs = np.array(sum(all_pose_seqs, []))  # (N, T, V*C)
    all_gridflow_seqs = np.array(sum(all_gridflow_seqs, []))  # (N, T, 24)
    all_bbox_seqs = np.array(sum(all_bbox_seqs, []))  # (N, T, 4)
    all_bbox_seqs = all_bbox_seqs.transpose(0, 2, 1)

    all_video_names = sum(all_video_names, [])
    all_image_names = sum(all_image_names, [])
    all_action_labels = sum(all_action_labels, [])
    all_action_indexes = sum(all_action_indexes, [])

    N, T, _ = all_pose_seqs.shape
    if (args.pose_18):
        all_pose_seqs = all_pose_seqs.reshape(N, T, 18, 3)  # (N, T, V, C)
    if (args.pose_25):
        all_pose_seqs = all_pose_seqs.reshape(N, T, 25, 3)
    all_pose_seqs = all_pose_seqs.transpose(0, 3, 1, 2)  # (N, C, T, V)

    # normalize gridflow
    all_gridflow_seqs = all_gridflow_seqs.reshape(N * T, 24)
    all_gridflow_seqs = (all_gridflow_seqs - all_gridflow_seqs.mean(axis=0)) / all_gridflow_seqs.var(axis=0)
    all_gridflow_seqs = all_gridflow_seqs.reshape(N, T, 24)

    # write to file
    if not os.path.exists(os.path.join(args.out_folder)):
        os.makedirs(os.path.join(args.out_folder))
    data_file = os.path.join(args.out_folder, "{}_data_{}.pkl".format(mode, args.obs_type))
    metadata_file = os.path.join(args.out_folder, "{}_metadata_{}.pkl".format(mode, args.obs_type))

    with open(data_file, 'wb') as f:
        pickle.dump((all_location_seqs, all_pose_seqs, all_gridflow_seqs), f)

    with open(metadata_file, 'wb') as f:
        pickle.dump((all_bbox_seqs, all_video_names, all_image_names, all_action_labels, all_action_indexes), f)

    print("number of video used: {}".format(len(video_dir)))
    print("pose shape: {}".format(all_pose_seqs.shape))
    print("total number of samples walking: {}".format(len([i for i in all_action_labels if i == 'walking'])))
    print("total number of samples standing: {}".format(len([i for i in all_action_labels if i == 'standing'])))

    print("Write data to file :", data_file)
    print("Write metadata to file :", metadata_file)


def test_KNNImputer():
    nan = np.nan

    X = np.array([[1, 2, nan, nan], [3, 4, 3, nan], [nan, 6, 5, nan], [8, 8, 7, nan]])  # X can be array or list
    print("Before imputing X = ")
    print(X)
    print(X.shape)

    imputer = KNNImputer(n_neighbors=2, weights="uniform")
    Y = imputer.fit_transform(X)

    print("After imputing X = ")
    print(Y)
    print(Y.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate train/val data for prediction on jaad')
    parser.add_argument(
        '--video_info_fn', default='/home/manhh/github/datasets/processed_data/features/jaad/video_info.csv')
    parser.add_argument(
        '--location_path', default='/home/manhh/github/datasets/processed_data/features/jaad/location')
    parser.add_argument(
        '--pose_path', default='/home/manhh/github/datasets/processed_data/features/jaad/pose_18_id')
    parser.add_argument(
        '--gridflow_path', default='/home/manhh/github/datasets/processed_data/features/jaad/gridflow')
    parser.add_argument(
        '--out_folder', default='/home/manhh/github/datasets/processed_data/prediction/jaad')
    parser.add_argument(
        '--pose_18', action="store_true", default=True, help='by default, using 18 keypoints')
    parser.add_argument(
        '--pose_25', action="store_true", default=False, help='by default, using 25 keypoints')
    parser.add_argument(
        '--traj_len', type=int, default=20, help='trajectory len')
    parser.add_argument(
        '--obs_len', type=int, default=10, help='trajectory len')
    parser.add_argument(
        '--slide', type=int, default=1, help='gap between trajectory')
    parser.add_argument(
        '--obs_type', type=str, default='noisy', help='noisy, impute, or gt')
    parser.add_argument(
        '--debug', action="store_true", default=False, help='debug mode')
    args = parser.parse_args()

    assert args.obs_type == 'noisy' or args.obs_type == 'impute' or args.obs_type == 'gt'

    # test_KNNImputer()
    generate_data(args, "train")
    generate_data(args, "val")
    # generate_data(args, "test")