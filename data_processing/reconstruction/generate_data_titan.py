import os
import json
import numpy as np
import pickle
import csv
from itertools import islice
from sklearn.impute import KNNImputer
from numpy.lib.format import open_memmap
import argparse


def chunks(data, traj_len=20, slide=1):
    # it = iter(data)
    for i in range(0, len(data), slide):
        yield list(islice(data, i, i + traj_len))


def impute_poses(poses):
    """
            interplote poses
            input: poses ~ [traj_len, 75]  : 75 is 25*3, 25 keypoints, 3: x,y,c
            ouput: imputed_poses ~ [traj_len, 75]
    """

    valid_sample = True
    poses_array = np.array(poses)

    imputer = KNNImputer(missing_values=0, n_neighbors=5, weights="uniform")
    imputed_poses = imputer.fit_transform(poses_array)

    imputed_poses_array = np.array(imputed_poses)

    if (imputed_poses_array.shape[1] != NUM_KEYPOINTS):
        valid_sample = False  # return -1 if shape is not right

    return imputed_poses, valid_sample


def generate_samples(video_data, video_name, mode, args):
    """Function to generate complete pose trajectory for each pedestrian in a video.
        Args:
            video_data (dict): data of each video, has the following format:
                'image_name'(str):{ 'people': [ {'person_id': (str),
                                                 'pose': (list),
                                                 'action_label': (str),
                                                  action_index': (int)}]
            video_name(str): name of the video
            traj_len(int): len of trajectory
            video_info_fn(str): file contains video_info (width and height), we need this because each video
                           in JAAD has different frame width and height
        Returns:
            video_samples (list(list)):  list of pose trajectories.
            video_names (list(str)): list of video names.  Each element is the video name
                of a corresponding pose trajectory.
            image_name_out (list(str)): list of start image names. Each element is
                the first frame of a corresponding pose trajectory.
            action_label_out (list(str)): list of action labels.
            action_index_out (list(int)): list of action indexes.

    """
    video_samples, video_names, image_name_out = [], [], []
    action_label_out, action_index_out = [], []
    video_bbox = []

    # find list of pedestrian id
    id_list = []
    for image_name in video_data:
        for person in video_data[image_name]["people"]:
            if (person['person_id'] not in id_list):
                id_list.append(person['person_id'])

    # extract trajectory for each pedestrian
    for pid in id_list:

        # extract whole trajectory of a pedestrian in video
        long_poses, long_image_names, long_action_indexes, long_action_labels = [], [], [], []
        for image_name in video_data:
            for person in video_data[image_name]["people"]:
                if (person['person_id'] == pid):
                    long_poses.append(person['pose'])
                    long_image_names.append(image_name)
                    long_action_labels.append(person['action_label'])
                    long_action_indexes.append(person['action_index'])


        # split trajectories into chunk of pre-defined trajectory length
        for pose, image_names, action_labels, action_indexes in zip(
                chunks(long_poses, traj_len=args.traj_len, slide=args.slide),
                chunks(long_image_names, traj_len=args.traj_len, slide=args.slide),
                chunks(long_action_labels, traj_len=args.traj_len, slide=args.slide),
                chunks(long_action_indexes, traj_len=args.traj_len, slide=args.slide)):

            # skip if trajectory is short
            if (len(pose) < args.traj_len):
                continue

            # skip of the label is not consistent
            consistent_label = True
            for action_index in long_action_indexes:
                if action_index != long_action_indexes[0]:
                    consistent_label = False
            if not consistent_label:
                continue

            # check if trajectory is continous by extracting frame number [-11:-4] from names
            # this applies for image names with format: 0000x.png  or {:05d}.png
            # gap = abs(int(image_names[0][-9:-4]) - int(image_names[-1][-9:-4]))
            # if (gap > args.traj_len):
            #     continue

            # skip if pose is not complete
            if mode == "train" or mode == "val":
                if 0 in sum(pose, []):
                    continue
            else:  # test data
                if 0 not in sum(pose, []):
                    continue


            # normalize poses
            pose = np.array(pose)  # (traj_len, 2)
            pose[:, 0::3] = pose[:, 0::3] / float(args.width)  # normalize x by frame width
            pose[:, 1::3] = pose[:, 1::3] / float(args.height)  # normalize y by frame height


            # calculate bounding box
            bbox = np.zeros((args.traj_len, 4))
            invalid_box = False
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
                    invalid_box = True
                bbox[t, :] = [xmin, ymin, xmax, ymax]

            if(invalid_box):
                continue

            pose[:, 0::3][pose[:, 2::3] == 0] = 0
            pose[:, 1::3][pose[:, 2::3] == 0] = 0

            # add to sample list
            video_samples.append(pose.tolist())
            video_bbox.append(bbox.tolist())
            video_names.append(video_name)
            image_name_out.append(image_names[0])  # keep first image name for meta-data
            action_label_out.append(action_labels[0])
            action_index_out.append(action_indexes[0])


    return video_samples, video_bbox, video_names, image_name_out, action_label_out, action_index_out


def generate_data(args, mode):
    """Function generate train/val data for the dataset.
        Args:
            mode(str): train or val
        Returns:
            fp (np.array): data ready for train/val.  It is written to *.npy file
               shape (N, C, T, V, 1) --> written to *.npy file
            metadata(tuple): (all_video_names, all_image_names, all_action_labels, all_action_indexes)
            It is written to *.json file
    """
    video_dir = os.listdir(os.path.join(args.pose_path))
    if (mode == "train"):
        video_dir = video_dir[:int(len(video_dir) * 0.8)]  # 80% of data for training
    else:  # val
        video_dir = video_dir[-int(len(video_dir) * 0.2):]  # 20% of data for val
    if (args.debug):
        video_dir = video_dir[:10]  # use 10 videos for debug

    all_samples, all_video_names, all_image_names = [], [], []
    all_action_labels, all_action_indexes = [], []
    all_bbox = []
    for video_name in video_dir:
        print("processing video: ", video_name)

        # gather data for each video
        video_data = {}
        frame_list = os.listdir(os.path.join(args.pose_path, video_name))

        for frame_name in frame_list:
            frame_number = int(frame_name[:6])

            with open(os.path.join(args.pose_path, video_name, "{:06d}_keypoints.json".format(frame_number)), "r") as f:
                pose_data = json.load(f)

            image_name = "{:06d}.png".format(frame_number)
            video_data[image_name] = {'people': []}
            for ped in pose_data['people']:
                if (ped['person_id'][0] == -1):
                    continue  # skip ped withput id
                video_data[image_name]['people'].append({'person_id': ped['person_id'],
                                                         'pose': ped['pose_keypoints_2d'],
                                                         'action_label': ped['action_label'],
                                                         'action_index': ped['action_index']
                                                         })

        # extract trajectory within a video
        video_samples,video_bbox, video_names, image_names, \
        action_labels, action_indexes = generate_samples(video_data, video_name, mode, args)

        all_samples.append(video_samples)
        all_bbox.append(video_bbox)
        all_video_names.append(video_names)
        all_image_names.append(image_names)
        all_action_labels.append(action_labels)
        all_action_indexes.append(action_indexes)


    # convert data to numpy array of shape (N, C, T, V, 1)
    all_samples = np.array(sum(all_samples, []))  # (N, T, V*C)
    all_bbox =  np.array(sum(all_bbox, []))  # (N, T, 4)
    all_bbox = all_bbox.transpose(0, 2, 1)          # (N, 4, T)
    all_video_names = sum(all_video_names, [])
    all_image_names = sum(all_image_names, [])
    all_action_labels = sum(all_action_labels, [])
    all_action_indexes = sum(all_action_indexes, [])
    print(len(all_samples))

    N, T, _ = all_samples.shape
    all_samples = all_samples.reshape(N, T, args.num_keypoints, 3)  # (N, T, V, C)
    all_samples = np.expand_dims(all_samples.transpose(0, 3, 1, 2), axis=4)  # (N, C, T, V, 1)

    # write to file
    if not os.path.exists(os.path.join(args.out_folder)):
        os.makedirs(os.path.join(args.out_folder))
    data_file = os.path.join(args.out_folder, "{}_data.npy".format(mode))
    fp = open_memmap(
        data_file,
        dtype='float32',
        mode='w+',
        shape=all_samples.shape)
    fp[...] = all_samples

    metadata_file = os.path.join(args.out_folder, "{}_metadata.pkl".format(mode))
    with open(metadata_file, 'wb') as f:
        pickle.dump((all_bbox, all_video_names, all_image_names, all_action_labels, all_action_indexes), f)

    # print
    print("number of video used: {}".format(len(video_dir)))
    print("data shape: {}".format(fp.shape))
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
        description='Generate train/val data for reconstruction on titan')
    parser.add_argument(
        '--pose_path', default='/home/manhh/github/datasets/processed_data/features/titan/pose_18_id')
    parser.add_argument(
        '--out_folder', default='/home/manhh/github/datasets/processed_data/reconstruction/titan/')
    parser.add_argument(
        '--num_keypoints', type=int, default=18, help='number of human keypoints')
    parser.add_argument(
        '--traj_len', type=int, default=10, help='trajectory len')
    parser.add_argument(
        '--slide', type=int, default=1, help='gap between trajectory')
    parser.add_argument(
        '--width', type=int, default=2704)
    parser.add_argument(
        '--height', default=1520)
    parser.add_argument(
        '--debug', action="store_true", default=False, help='debug mode')
    args = parser.parse_args()

    # test_KNNImputer()
    generate_data(args, "train")
    generate_data(args, "val")
