import os
import csv
import json
import argparse
import numpy as np
from sklearn.impute import KNNImputer
from itertools import islice


def chunks(data, traj_len=20, slide=1):
    # it = iter(data)
    for i in range(0, len(data), slide):
        yield list(islice(data, i, i + traj_len))


def impute_poses(args, poses):
    """
            interpolate poses
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


def generate_samples(video_data, video_name, video_info_fn, traj_len):
    """
        Args:
            video_data: a dictionary
                'image_name'(str):{'people': [
                                {'person_id': (str)
                                'pose'  : list}]}
        Returns:
            loc_seq (np.array): trajectory data of the video, shape: (N, traj_len, 2)
            fid_seq (np.array): frame id of each trajectory, shape: (N, traj_len, 1)
            fid_seq (np.array): pedestrian id of each trajectory, shape: (N, traj_len, 1)
    """
    location_seq, fid_seq, pid_seq = [], [], []

    # read video_info for frame width and height
    video_info = {}
    with open(video_info_fn, 'r') as data:
        for line in csv.reader(data):
            video_info[line[0]] = [line[1], line[2]]

    # find list of pedestrian id
    id_list = []
    for image_name in video_data:
        for person in video_data[image_name]["people"]:
            if person['person_id'] not in id_list:
                id_list.append(person['person_id'])

    # extract trajectory for each pedestrian
    for pid, pName in enumerate(id_list):
        # extract whole trajectory of a pedestrian in video
        long_poses, long_inames = [], []
        long_fid, long_pid = [], []
        long_gtloc = []
        for fid, image_name in enumerate(video_data):
            for person in video_data[image_name]["people"]:
                if person['person_id'] == pName:
                    long_poses.append(person['pose'])
                    long_gtloc.append(person['location'])
                    long_fid.append(fid)
                    long_pid.append(pid)
                    long_inames.append(image_name)

        # split trajectories into chunk of pre-defined trajectory length
        for pose_list, gtloc_list, iname_list, fid_list, pid_list in zip(
                chunks(long_poses, traj_len=traj_len, slide=traj_len),
                chunks(long_gtloc, traj_len=traj_len, slide=traj_len),
                chunks(long_inames, traj_len=traj_len, slide=traj_len),
                chunks(long_fid, traj_len=traj_len, slide=traj_len),
                chunks(long_pid, traj_len=traj_len, slide=traj_len)):

            # skip if trajectory is short
            if len(pose_list) < traj_len:
                continue

            # check if trajectory is continous by extracting frame number [-11:-4] from names
            # this applies for image names with format: 0000x.png  or {:05d}.png
            gap = abs(int(iname_list[0][-9:-4]) - int(iname_list[-1][-9:-4]))
            if gap > traj_len:
                continue

            # impute data
            pose_arr = np.array(pose_list)  # (traj_len, 2)
            if args.obs_type == 'impute':
                pose_arr, valid = impute_poses(args, pose_arr)

                # normalize poses
                pose_arr[:, 0::3] = pose_arr[:, 0::3] / float(video_info[video_name][0])  # normalize x by frame width
                pose_arr[:, 1::3] = pose_arr[:, 1::3] / float(video_info[video_name][1])  # normalize y by frame height
                pose_arr[:, 0::3] = pose_arr[:, 0::3] - 0.5  # centralize
                pose_arr[:, 1::3] = pose_arr[:, 1::3] - 0.5  # centralize
                pose_arr[:, 0::3][pose_arr[:, 2::3] == 0] = 0
                pose_arr[:, 1::3][pose_arr[:, 2::3] == 0] = 0

                # get location data from pose
                location = np.zeros((traj_len, 2))
                left_hip = pose_arr[:, 24:26]
                right_hip = pose_arr[:, 33:35]
                if 0 not in left_hip and 0 not in right_hip:
                    location = 0.5 * (left_hip + right_hip)
                elif 0 in left_hip:
                    location = right_hip
                else:
                    location = left_hip

            if args.obs_type == 'gt':
                location = np.array(gtloc_list)  # (T, 2)
                location[:, 0] = location[:, 0] / float(video_info[video_name][0])  # normalize x by frame width
                location[:, 1] = location[:, 1] / float(video_info[video_name][1])  # normalize y by frame height
                location[:, 0] = location[:, 0] - 0.5  # centralize
                location[:, 1] = location[:, 1] - 0.5  # centralize

            # add to sample list
            location_seq.append(location)
            fid_seq.append(fid_list)
            pid_seq.append(pid_list)

    return np.array(location_seq), np.array(fid_seq), np.array(pid_seq)


def generate_data(pose_path, mode, debug=False):
    """ Generate data for social-stgcnn

    Args
        pose_path: path to pose features
        mode: 'train' or 'val
    Returns
        data (txt files)
    """
    video_list = os.listdir(pose_path)
    if mode is 'train':
        video_list = video_list[:int(len(video_list) * 0.8)]  # 80% of data for training
    else:  # val
        video_list = video_list[-int(len(video_list) * 0.2):]  # 20% of data for val
    if args.debug:
        video_list = video_list[:10]  # use 10 videos for debug
    for video_name in video_list:
        print("processing video: ", video_name)
        # gather data for each video
        video_data = {}
        for fid in range(len(os.listdir(os.path.join(args.pose_path, video_name))) - 1):

            # read pose data
            with open(os.path.join(args.pose_path, video_name, "{:05d}_keypoints.json".format(fid)), "r") as f:
                pose_data = json.load(f)

            # read location data (in case use gt observation)
            with open(os.path.join(args.location_path, video_name, "{:05d}_locations.json".format(fid)), "r") as f:
                location_data = json.load(f)

            image_name = "{:05d}.png".format(fid)
            video_data[image_name] = {'people': []}
            for ped in pose_data['people']:
                if ped['person_id'][0] == -1:
                    continue  # skip ped without id

                temp_location = []
                for p_in_loc in location_data['people']:
                    if ped['person_id'] == p_in_loc['person_id']:
                        temp_location = p_in_loc['center']

                if not temp_location:
                    continue

                video_data[image_name]['people'].append({'person_id': ped['person_id'],
                                                         'location': temp_location,
                                                         'pose': ped['pose_keypoints_2d']})

        location_seq, fid_seq, pid_seq = generate_samples(video_data, video_name, args.video_info_fn, args.traj_len)
        if location_seq.size is 0:
            continue

        # convert loc_seqs to data with format (frame_id, ped_id, x, y)
        data = convert_data(location_seq, fid_seq, pid_seq, args.traj_len)
        # write to text file
        out_dir = os.path.join(args.out_folder, 'jaad_{}'.format(args.obs_type), mode)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        file = open(os.path.join(out_dir, '{}.txt'.format(video_name)), 'w', newline='')
        with file:
            header = ['fid', 'pid', 'x', 'y']
            writer = csv.DictWriter(file, fieldnames=header, delimiter='\t')
            for row in data:
                writer.writerow({'fid': row[0], 'pid': row[1], 'x': row[2], 'y': row[3]})
    print("generated ", out_dir)


def convert_data(loc_seq, fid_seq, pid_seq, traj_len):
    """
        Args:
            loc_seq (np.array) : trajectory data, shape (N, T, 2)
            fid_seq (np.array): list of pedestrian id of each trajectory, shape (N, T)
            pid_seq (np.array): list of start frame of each trajectory, shape (N, T)
        Returns:
            out_data (dict): data for each frame, ready for write to text file. out_data has the
            following format:
                {'frame_id'(int): [], 'ped_id' (int): [], 'x': [] , 'y': []}
    """

    num_frames = np.amax(fid_seq)
    num_samples = fid_seq.shape[0]
    out_data = []
    for fid in range(num_frames):
        for s in range(num_samples):
            for t in range(traj_len):
                if fid_seq[s, t] == fid:
                    out_data.append([fid, pid_seq[s, t], loc_seq[s, t, 0], loc_seq[s, t, 1]])
    return out_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Generate train/val data JAAD for social-stgcnn')
    parser.add_argument(
        '--location_path', default='../datasets/processed_data/features/jaad/location')
    parser.add_argument(
        '--pose_path', default='../datasets/processed_data/features/jaad/pose_18_id')
    parser.add_argument(
        '--out_folder', default='../Social-STGCNN/datasets')
    parser.add_argument(
        '--video_info_fn', default='../datasets/processed_data/features/jaad/video_info.csv')
    parser.add_argument(
        '--traj_len', type=int, default=20, help='trajectory len')
    parser.add_argument(
        '--obs_type', type=str, default='raw', help='raw, impute, or gt')
    parser.add_argument(
        '--debug', action="store_true", default=False, help='debug mode')
    args = parser.parse_args()

    assert args.obs_type == 'raw' or args.obs_type == 'impute' or args.obs_type == 'gt'

    print("Generate train data jaad for social-stgcnn")
    generate_data(args.pose_path, "train", args.debug)
    print("Generate val data jaad for social-stgcnn")
    generate_data(args.pose_path, "val", args.debug)
