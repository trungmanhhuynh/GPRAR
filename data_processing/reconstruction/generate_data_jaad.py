import os
import json
import numpy as np
import pickle
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

    if(imputed_poses_array.shape[1] != NUM_KEYPOINTS):
        valid_sample = False            # return -1 if shape is not right

    return imputed_poses, valid_sample


def generate_samples(video_data, video_name, args):
    '''
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
    '''
    video_samples, video_names, image_name_out = [], [], []
    action_label_out, action_index_out = [], []
    # find list of pedestrian id
    id_list = []
    for image_name in video_data:
        for person in video_data[image_name]["people"]:
            if(person['person_id'] not in id_list):
                id_list.append(person['person_id'])

    # extract trajectory for each pedestrian
    for pid in id_list:

        # extract whole trajectory of a pedestrian in video
        long_poses, long_image_names, long_action_indexes, long_action_labels = [], [], [], []
        for image_name in video_data:
            for person in video_data[image_name]["people"]:
                if(person['person_id'] == pid):
                    long_poses.append(person['pose'])
                    long_image_names.append(image_name)
                    long_action_labels.append(person['action_label'])
                    long_action_indexes.append(person['action_index'])

        # split trajectories into chunk of pre-defined trajectory length
        for poses, image_names, action_labels, action_indexes in zip(
                chunks(long_poses, traj_len=args.traj_len, slide=args.slide),
                chunks(long_image_names, traj_len=args.traj_len, slide=args.slide),
                chunks(long_action_labels, traj_len=args.traj_len, slide=args.slide),
                chunks(long_action_indexes, traj_len=args.traj_len, slide=args.slide)):

            # skip if trajectory is short
            if(len(poses) < args.traj_len):
                continue

            # skip of the label is not consistent
            consistent_label = True
            for action_index in long_action_indexes:
                if action_index != long_action_indexes[0]:
                    consistent_label = False
            if(not consistent_label):
                continue

            # check if trajectory is continous by extracting frame number [-11:-4] from names
            # this applies for image names with format: 0000x.png  or {:05d}.png
            gap = abs(int(image_names[0][-9:-4]) - int(image_names[-1][-9:-4]))
            if(gap > args.traj_len):
                continue

            # add to sample list
            video_samples.append(poses)
            video_names.append(video_name)
            image_name_out.append(image_names[0])       # keep first image name for meta-data
            action_label_out.append(action_labels[0])
            action_index_out.append(action_indexes[0])

    return video_samples, video_names, image_name_out, action_label_out, action_index_out


def generate_data(args, mode):
    """
        mode: train of val
        output:
            - fp: numpy array of shape (N, C, T, V, 1) --> written to *.npy file
            - label: list of N samples --> writting to *.json file
    """
    video_dir = os.listdir(os.path.join(args.pose_path))
    if(mode == "train"):
        video_dir = video_dir[:int(len(video_dir) * 0.8)]           # 80% of data for training
    else:  # val
        video_dir = video_dir[-int(len(video_dir) * 0.2):]          # 20% of data for val
    if(args.debug):
        video_dir = video_dir[:10]                                  # use 10 videos for debug

    all_samples, all_video_names, all_image_names = [], [], []
    all_action_labels, all_action_indexes = [], []
    for video_name in video_dir:
        print("processing video: ", video_name)

        # gather data for each video
        video_data = {}
        for frame_number in range(len(os.listdir(os.path.join(args.pose_path, video_name))) - 1):
            # -1 because there is no gridflow data for the last frame
            with open(os.path.join(args.pose_path, video_name, "{:05d}_keypoints.json".format(frame_number)), "r") as f:
                pose_data = json.load(f)

            image_name = "{:05d}.png".format(frame_number)
            video_data[image_name] = {'people': []}
            for ped in pose_data['people']:
                if(ped['person_id'][0] == -1):
                    continue  # skip ped withput id
                video_data[image_name]['people'].append({'person_id': ped['person_id'],
                                                         'pose': ped['pose_keypoints_2d'],
                                                         'action_label': ped['action_label'],
                                                         'action_index': ped['action_index']
                                                         })

        # extract trajectory within a video
        video_samples, video_names, image_names, action_labels, action_indexes = generate_samples(video_data, video_name, args)

        all_samples.append(video_samples)
        all_video_names.append(video_names)
        all_image_names.append(image_names)
        all_action_labels.append(action_labels)
        all_action_indexes.append(action_indexes)

    # convert data to numpy array of shape (N, C, T, V, 1)
    all_samples = np.array(sum(all_samples, []))   # (N, T, V*C)
    all_video_names = sum(all_video_names, [])
    all_image_names = sum(all_image_names, [])
    all_action_labels = sum(all_action_labels, [])
    all_action_indexes = sum(all_action_indexes, [])

    N, T, _ = all_samples.shape
    if(args.pose_18):
        all_samples = all_samples.reshape(N, T, 18, 3)       # (N, T, V, C)
    if(args.pose_25):
        all_samples = all_samples.reshape(N, T, 25, 3)
    all_samples = np.expand_dims(all_samples.transpose(0, 3, 1, 2), axis=4)   # (N, C, T, V, 1)

    # normalize data
    all_samples[:, 0, :, :, :] = all_samples[:, 0, :, :, :] / args.width  # normalize x
    all_samples[:, 1, :, :, :] = all_samples[:, 1, :, :, :] / args.height  # normalize x
    all_samples[:, 0:2, :, :, :] = all_samples[:, 0:2, :, :, :] - 0.5  # centralize
    all_samples[:, 0, :, :, :][all_samples[:, 2, :, :, :] == 0] = 0
    all_samples[:, 1, :, :, :][all_samples[:, 2, :, :, :] == 0] = 0

    #--- write to file
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
        pickle.dump((all_video_names, all_image_names, all_action_labels, all_action_indexes), f)

    print("number of video used: {}".format(len(video_dir)))
    print("data shape: {}".format(fp.shape))
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
        description='Generate train/val data for reconstruction on jaad')
    parser.add_argument(
        '--pose_path', default='data/features/jaad/pose_18_id')
    parser.add_argument(
        '--out_folder', default='data/reconstruction/jaad/')
    parser.add_argument(
        '--pose_18', action="store_true", default=True, help='by default, using 18 keypoints')
    parser.add_argument(
        '--pose_25', action="store_true", default=False, help='by default, using 25 keypoints')
    parser.add_argument(
        '--traj_len', type=int, default=10, help='trajectory len')
    parser.add_argument(
        '--slide', type=int, default=1, help='gap between trajectory')
    parser.add_argument(
        '--width', type=float, default=1920, help='default frame_width given by jadd dataset')
    parser.add_argument(
        '--height', type=float, default=1920, help='default frame_height given by jadd dataset')
    parser.add_argument(
        '--debug', action="store_true", default=False, help='debug mode')
    args = parser.parse_args()

    # test_KNNImputer()
    generate_data(args, "train")
    generate_data(args, "val")
