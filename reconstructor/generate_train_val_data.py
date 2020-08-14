"""
Script to generate train and validation data for reconstrutor
author: Manh Huynh

"""
import os
import json
import joblib
import random
import time
import argparse
import numpy as np
from itertools import islice
from sklearn.impute import KNNImputer


def chunks(data, traj_len=20, slide=1):
    # it = iter(data)
    for i in range(0, len(data), slide):
        yield list(islice(data, i, i + traj_len))


def impute_poses(poses,
                 num_keypoints,
                 num_feats):
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

    if(imputed_poses_array.shape[1] != num_keypoints * num_feats):
        valid_sample = False            # return -1 if shape is not right

    return imputed_poses, valid_sample


def generate_samples(video_data,
                     video_name,
                     num_keypoints,
                     num_feats,
                     obs_len,
                     pred_len):
    '''
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
    video_samples = []
    traj_len = obs_len + pred_len
    # find list of pedestrian id
    id_list = []
    for image_name in video_data:
        for person in video_data[image_name]["people"]:
            if(person['person_id'] not in id_list):
                id_list.append(person['person_id'])

    # extract trajectory of each pedestrian
    num_nonvalid = 0

    for pid in id_list:

        # extra whole trajector of a pedestrian
        long_video_names = []
        long_image_names = []
        long_person_ids = []
        long_poses = []
        long_locations = []
        long_bboxes = []

        for image_name in video_data:
            for person in video_data[image_name]["people"]:
                if(person['person_id'] == pid):
                    long_video_names.append(video_name)
                    long_image_names.append(image_name)
                    long_person_ids.append(pid)
                    long_poses.append(person['pose'])
                    long_locations.append(person['center'])
                    long_bboxes.append(person['bbox'])

        # interpolate poses
        long_imputed_pose, valid_sample = impute_poses(poses=long_poses,
                                                       num_keypoints=num_keypoints,
                                                       num_feats=num_feats)
        if(not valid_sample):
            num_nonvalid += 1
            continue

        # cut trajectories into chunk of pre-defined trajectory length
        for video_names, image_names, person_ids, poses, imputed_poses, locations, bboxes in \
            zip(chunks(long_video_names, traj_len=traj_len, slide=1),
                chunks(long_image_names, traj_len=traj_len, slide=1),
                chunks(long_person_ids, traj_len=traj_len, slide=1),
                chunks(long_poses, traj_len=traj_len, slide=1),
                chunks(long_imputed_pose, traj_len=traj_len, slide=1),
                chunks(long_locations, traj_len=traj_len, slide=1),
                chunks(long_bboxes, traj_len=traj_len, slide=1)):

            # skip if extracted trajectory is shorted thatn pre-defined one
            if(len(locations) < traj_len):
                continue

            # check if trajectory is continous by extracting frame number [-11:-4] from names
            # this applies for image names with format: 0000x.png  or {:05d}.png
            gap = abs(int(image_names[0][-9:-4]) - int(image_names[-1][-9:-4]))
            if(gap > traj_len):
                continue

            exist_zeros = False
            for t in range(traj_len):
                for k in range(num_keypoints * num_feats):
                    if(imputed_poses[t][k] == 0):
                        exist_zeros = True

            if(exist_zeros):
                continue

            # add to sample list
            video_samples.append({
                'video_names': video_names,
                'image_names': image_names,
                'person_ids': person_ids,
                'poses': poses,
                'imputed_poses': imputed_poses,
                'locations': locations,
                'bboxes': bboxes
            })

            # print(video_samples)
            # input("here")

    return video_samples, len(video_samples), num_nonvalid


def generate_train_val_data(processed_location_dir,
                            proccesed_pose_dir,
                            output_dir,
                            num_keypoints,
                            num_feats,
                            obs_len,
                            pred_len,
                            data_size
                            ):
    '''

    '''

    # generate samples (list of features) for each pedestrian
    video_dir = os.path.join(proccesed_pose_dir)

    if(data_size == "small"):
        video_files = random.sample(os.listdir(video_dir), k=int(0.25 * len(os.listdir(video_dir))))
    elif(data_size == "medium"):
        video_files = random.sample(os.listdir(video_dir), k=int(0.50 * len(os.listdir(video_dir))))
    elif(data_size == "large"):
        video_files = os.listdir(video_dir)
    else:
        print("wrong data size")
        exit(-1)

    # init variables
    total_samples = []
    total_num_samples = 0
    total_num_nonvalid = 0

    stop_timer = time.time()
    for video_name in video_files:

        '''
            a video data has the following dictionary structure
            'image_name'(str):{
                'people': [
                    {
                    'person_id': (str)
                    'bbox':  [x1, y1, x2, y2]
                    'center': [xc, yc]
                    'pose'  : [1x75]  // [x,y,c] x 3
                    }
                ]
            }
        '''

        print("processing video: ", video_name)
        video_data = {}

        for frame_number in range(len(os.listdir(os.path.join(args.proccesed_pose_dir, video_name)))):

            # read processed data
            with open(os.path.join(args.proccesed_pose_dir, video_name, "{:05d}_keypoints.json".format(frame_number)), "r") as f:
                pose_data = json.load(f)
            with open(os.path.join(args.processed_location_dir, video_name, "{:05d}_locations.json".format(frame_number)), "r") as f:
                location_data = json.load(f)

            image_name = "{:05d}.png".format(frame_number)
            video_data[image_name] = {}
            video_data[image_name]['people'] = []

            for ped in pose_data['people']:

                if(ped['person_id'][0] == -1):
                    continue

                temp = {}
                temp['person_id'] = ped['person_id']
                temp['pose'] = ped['pose_keypoints_2d']

                for p_in_loc in location_data['people']:
                    if(ped['person_id'] == p_in_loc['person_id']):

                        temp['bbox'] = p_in_loc['bbox']
                        temp['center'] = p_in_loc['center']

                video_data[image_name]['people'].append(temp)

        # extract samples in a video
        video_samples, num_samples, num_nonvalid = generate_samples(video_data,
                                                                    video_name,
                                                                    num_keypoints,
                                                                    num_feats,
                                                                    obs_len,
                                                                    pred_len
                                                                    )

        total_samples.append(video_samples)
        total_num_samples += num_samples
        total_num_nonvalid += num_nonvalid

    total_samples = sum(total_samples, [])
    assert total_num_samples == len(total_samples)

    # split train/val data with ratio 80%/20%
    val_sample_indexes = random.sample(range(len(total_samples)), k=int(len(total_samples) * 0.2))
    train_sample_indexes = list(set(range(len(total_samples))) - set(val_sample_indexes))

    val_samples = [total_samples[i] for i in val_sample_indexes]
    train_samples = [total_samples[i] for i in train_sample_indexes]

    # dump to file
    if not os.path.exists(os.path.join(output_dir, data_size)):
        os.makedirs(os.path.join(output_dir, data_size))

    train_data_file = os.path.join(output_dir, data_size + "_train_data.joblib")
    val_data_file = os.path.join(output_dir, data_size + "_val_data.joblib")

    joblib.dump(train_samples, train_data_file)
    joblib.dump(val_samples, val_data_file)
    start_timer = time.time()

    print("number of video used: {}".format(len(video_files)))
    print("total number of samples used: {}".format(total_num_samples))
    print("total nonvalid samples: {}".format(total_num_nonvalid))
    print("train/val samples = {}/{}".format(len(train_samples), len(val_samples)))
    print("Dumped train data to file :", train_data_file)
    print("Dumped val data to file :", val_data_file)
    print("Processing time : {:.2f} (s)".format((start_timer - stop_timer)))


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

    # parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_location_dir', type=str, default="processed_data/JAAD/location",
                        help='directory of proccesed locations')
    parser.add_argument('--proccesed_pose_dir', type=str, default="processed_data/JAAD/pose_id",
                        help='directory of proccesed pose')
    parser.add_argument('--output_dir', type=str, default="reconstructor/train_val_data/JAAD",
                        help='directory of output train/val data')
    parser.add_argument('--num_keypoints', type=int, default=25,
                        help='number of keypoints (default using POSE_25)')
    parser.add_argument('--num_feats', type=int, default=3,
                        help='each keypoint has [x,y,c]')
    parser.add_argument('--obs_len', type=int, default=10,
                        help='observed length')
    parser.add_argument('--pred_len', type=int, default=10,
                        help='predicted length')
    parser.add_argument('--data_size', type=str, default="small",
                        help='choose the size of data: small (25%), medium (50%), large (100%)')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed for random')
    args = parser.parse_args()

    random.seed(args.seed)

    # test_KNNImputer()

    generate_train_val_data(
        processed_location_dir=args.processed_location_dir,
        proccesed_pose_dir=args.proccesed_pose_dir,
        output_dir=args.output_dir,
        num_keypoints=args.num_keypoints,
        num_feats=args.num_feats,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        data_size=args.data_size
    )
