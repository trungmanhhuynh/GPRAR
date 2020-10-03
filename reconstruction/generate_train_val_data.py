"""
Script to generate train and validation data

author: Manh Huynh
last update: 06/21/2020

"""
import os
import json
import joblib
import random
import time
import numpy as np
from itertools import islice
from sklearn.impute import KNNImputer


# parameters
DATA_DIR = "/home/manhh/github/datasets/features/kinetics"
TRAIN_VAL_DIR = "/home/manhh/github/Traj-STGCNN/reconstruction/data/kinectics"

random.seed(1)


def chunks(data, T=10, slide=1):
    # it = iter(data)
    for i in range(0, len(data), slide):
        yield list(islice(data, i, i + T))


def generate_samples(video_data, video_name, T=10):
    '''
        Input:
            video_data

    '''
    video_samples = []

    # find list of pedestrian id
    id_list = []
    for frame_index in video_data:
        for person in video_data[frame_index]["people"]:
            if(person['person_id'] not in id_list):
                id_list.append(person['person_id'])

    # extract trajectory of each pedestrian
    num_nonvalid = 0
    for pid in id_list:

        # extra whole trajector of a pedestrian
        long_poses = []
        long_frame_indexes = []
        long_label, long_label_index = [], []
        for frame_index in video_data:
            for person in video_data[frame_index]["people"]:
                if(person['person_id'] == pid):
                    long_poses.append(person['keypoints'])
                    long_frame_indexes.append(frame_index)
                    long_label.append(person['label'])
                    long_label_index.append(person['label_index'])

        # divide trajectories into chunk of pre-defined trajectory length
        for poses, frame_indexes, labels, label_indexes in zip(chunks(long_poses, T=T),
                                                               chunks(long_frame_indexes, T=T),
                                                               chunks(long_label, T=T),
                                                               chunks(long_label_index, T=T)
                                                               ):

            # skip if extracted len is shorted thatn pre-defined one
            if(len(poses) < T):
                continue

            # check if trajectory is continous by extracting frame number [-11:-4] from names
            # this applies for image names with format: 0000x.png  or {:05d}.png
            gap = abs(int(frame_indexes[0]) - int(frame_indexes[-1]))
            if(gap > T):
                continue

            # add to sample list
            video_samples.append({
                'video_names': video_name,
                'start_frame': frame_indexes[0],
                'person_ids': pid,
                'poses': poses,
                'label': labels[0],
                'label_index': label_indexes[0]
            })

            # print(video_samples)
            # input("here")

    return video_samples, len(video_samples), num_nonvalid


def generate_data(mode, debug=True):
    """


    """
    #---generate samples (list of features) for each pedestrian
    data_path = os.path.join(DATA_DIR, mode)
    video_list = os.listdir(data_path)
    if debug:
        video_list = video_list[:2]

    total_samples, total_num_samples, total_num_nonvalid = [], 0, 0
    for video_name in video_list:

        print("processing video: ", video_name)
        video_data = {}
        frame_list = os.listdir(os.path.join(DATA_DIR, mode, video_name))
        for frame_index in range(1, len(frame_list) + 1):

            with open(os.path.join(DATA_DIR, mode, video_name, "{:05d}_keypoints.json".format(frame_index)), "r") as f:
                frame_data = json.load(f)

            video_data[frame_index] = {'people': []}
            for person in frame_data['people']:
                video_data[frame_index]['people'].append({
                    'person_id': person['person_id'],
                    'keypoints': person['keypoints'],
                    'label': person['label'],
                    'label_index': person['label_index']
                })

        # extract samples in a video
        video_samples, num_samples, num_nonvalid = generate_samples(video_data, video_name)

        video_samples.append(video_samples)
        total_num_samples += num_samples
        total_num_nonvalid += num_nonvalid

    if not os.path.exists(os.path.join(TRAIN_VAL_DIR)):
        os.makedirs(os.path.join(TRAIN_VAL_DIR))

    output_file = os.path.join(TRAIN_VAL_DIR, "{}_data.joblib".format(mode))
    joblib.dump(video_samples, output_file)

    print("Number of samples:", num_samples)
    print("Saved data to file :", output_file)


if __name__ == "__main__":

    generate_data("train", debug=True)
    generate_data("val", debug=True)
