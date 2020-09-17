"""
Script to generate train and validation data for reconstrutor
author: Manh Huynh

"""
import os
import json
import joblib
import random
import argparse

import time
from itertools import islice


def chunks(data, traj_len=20, slide=1):
    # it = iter(data)
    for i in range(0, len(data), slide):
        yield list(islice(data, i, i + traj_len))


def generate_samples(video_data, video_name, args):
    '''
        extract samples for each video. Each sample has the following dict structure:

        {
            'poses': list ~[traj_len, 75]
            'locations': list ~[traj_len, 2]
            'video_names': [traj_len]
            'image_names': [traj_len]
            'person_ids': [traj_len]
        }

    '''
    video_samples = []
    traj_len = args.obs_len + args.pred_len

    # find list of pedestrian id
    id_list = []
    for image_name in video_data:
        for person in video_data[image_name]["people"]:
            if(person['person_id'] not in id_list):
                id_list.append(person['person_id'])

    # extract trajectory of each pedestrian
    num_nonvalid = 0

    for pid in id_list:

        # extra whole trajectory of a pedestrian
        long_poses = []
        long_locations = []
        long_video_names = []
        long_image_names = []
        long_person_ids = []
        long_flow = []

        for image_name in video_data:
            for person in video_data[image_name]["people"]:
                if(person['person_id'] == pid):
                    long_video_names.append(video_name)
                    long_image_names.append(image_name)
                    long_person_ids.append(pid)
                    long_poses.append(person['pose'])
                    long_locations.append(person['center'])
                    long_flow.append(person['flow'])

        # cut trajectories into chunk of pre-defined trajectory length
        for video_names, image_names, person_ids, poses, locations, flow in \
            zip(chunks(long_video_names, traj_len=traj_len, slide=1),
                chunks(long_image_names, traj_len=traj_len, slide=1),
                chunks(long_person_ids, traj_len=traj_len, slide=1),
                chunks(long_poses, traj_len=traj_len, slide=1),
                chunks(long_locations, traj_len=traj_len, slide=1),
                chunks(long_flow, traj_len=traj_len, slide=1)):

            # skip if extracted trajectory is shorted thatn pre-defined one
            if(len(locations) < traj_len):
                continue

            # check if trajectory is continous by extracting frame number [-11:-4] from names
            # this applies for image names with format: 0000x.png  or {:05d}.png
            gap = abs(int(image_names[0][-9:-4]) - int(image_names[-1][-9:-4]))
            if(gap > traj_len):
                continue

            # add to sample list
            video_samples.append({
                'poses': poses,
                'gt_locations': locations,
                'video_names': video_names,
                'image_names': image_names,
                'person_ids': person_ids,
                'flow': flow
            })

    return video_samples, len(video_samples), num_nonvalid


def generate_train_val_data(args):

    # generate samples (list of features) for each pedestrian
    video_dir = os.path.join(args.pose_dir)

    if(args.d_size == "small"):
        num_videos = random.sample(os.listdir(video_dir), k=30)
    elif(args.d_size == "medium"):
        num_videos = random.sample(os.listdir(video_dir), k=50)
    elif(args.d_size == "large"):
        num_videos = os.listdir(video_dir)                              # use all videos
    else:
        print("wrong data size")
        exit(-1)

    total_samples = []
    total_num_samples = 0
    total_num_nonvalid = 0
    for video_name in num_videos:

        '''
            generate data for a video with this dictionary structure
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

        for frame_number in range(len(os.listdir(os.path.join(args.pose_dir, video_name))) - 1):

            with open(os.path.join(args.pose_dir, video_name, "{:05d}_keypoints.json".format(frame_number)), "r") as f:
                pose_data = json.load(f)
            with open(os.path.join(args.location_dir, video_name, "{:05d}_locations.json".format(frame_number)), "r") as f:
                location_data = json.load(f)
            with open(os.path.join(args.flow_dir, video_name, "{:05d}_gridflow.json".format(frame_number)), "r") as f:
                flow_data = json.load(f)

            image_name = "{:05d}.png".format(frame_number)
            video_data[image_name] = {}
            video_data[image_name]['people'] = []
            for ped in pose_data['people']:

                if(ped['person_id'][0] == -1):
                    continue

                person = {}
                person['person_id'] = ped['person_id']
                person['pose'] = ped['pose_keypoints_2d']
                person['flow'] = flow_data[str(int(frame_number))]

                for p_in_loc in location_data['people']:
                    if(ped['person_id'] == p_in_loc['person_id']):
                        person['bbox'] = p_in_loc['bbox']
                        person['center'] = p_in_loc['center']

                video_data[image_name]['people'].append(person)

        # extract samples in a video
        video_samples, num_samples, num_nonvalid = generate_samples(video_data, video_name, args)

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

    print("number of video used: {}".format(len(num_videos)))
    print("total number of samples used: {}".format(total_num_samples))
    print("total nonvalid samples: {}".format(total_num_nonvalid))
    print("train/val samples = {}/{}".format(len(train_samples), len(val_samples)))

    return train_samples, val_samples


if __name__ == "__main__":

    # test_KNNImputer()
    parser = argparse.ArgumentParser()
    parser.add_argument('--obs_len', type=int, default=10)
    parser.add_argument('--pred_len', type=int, default=10)
    parser.add_argument('--location_dir', type=str, default="/home/manhh/github/Traj-STGCNN/processed_data/JAAD/location",
                        help='directory of processed location feature')
    parser.add_argument('--pose_dir', type=str, default="/home/manhh/github/Traj-STGCNN/processed_data/JAAD/pose_id",
                        help='directory of processed pose feature')
    parser.add_argument('--flow_dir', type=str, default="/home/manhh/github/Traj-STGCNN/processed_data/JAAD/gridflow",
                        help='directory of processed flow feature')
    parser.add_argument('--output_dir', type=str, default="/home/manhh/github/Traj-STGCNN/train_val_data/JAAD",
                        help='directory of output data')
    parser.add_argument('--d_size', type=str, default="small",
                        help='d_size: small, medium, large')
    parser.add_argument('--num_keypoints', type=int, default=25,
                        help='number of keypoints per pose')
    args = parser.parse_args()

    random.seed(1)
    start_timer = time.time()

    # generate data
    train_samples, val_samples = generate_train_val_data(args)

    # dump to file
    if not os.path.exists(os.path.join(args.output_dir, "predictor")):
        os.makedirs(os.path.join(args.output_dir, "predictor"))
    train_data_file = os.path.join(args.output_dir, "predictor", "train_{}.joblib".format(args.d_size))
    val_data_file = os.path.join(args.output_dir, "predictor", "val_{}.joblib".format(args.d_size))
    joblib.dump(train_samples, train_data_file)
    joblib.dump(val_samples, val_data_file)

    print("Dumped train data to file :", train_data_file)
    print("Dumped val data to file :", val_data_file)
    print("Processing time : {:.2f} (s)".format((time.time() - start_timer)))
