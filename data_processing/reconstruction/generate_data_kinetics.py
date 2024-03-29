import argparse
import pickle
import numpy as np
from numpy.lib.format import open_memmap
import json
import sys
import os


def print_toolbar(rate, annotation='', toolbar_width=30):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')

def end_toolbar():
    sys.stdout.write("\n")


def generate_data(args, mode):
    """ function to generate train/val data for reconstruction using kinetics dataset
        mode: train of val
        output:
            - fp: numpy array of shape (N, C, T, V, 1) --> written to *.npy file. Pose data
                  is normalized using standard normalization with mean 0 and variance 0.5.
            - tuple: (sample_name, all_label, all_bbox) --> writting to {}_label.pkl file
                sample_name (list of string): list of video names
                all_label (list of string): list of action labels.
                all_bbox (numpy array): bounding box data. Shape: (N, T, 4)
    """

    # read data and label. The data is already normalized to range (0,1) by divided
    # pose coordinates to image width and height.
    label_path = os.path.join(args.data_path, "kinetics_{}_label.json".format(mode))
    with open(label_path) as f:
        label_info = json.load(f)
    data_path = os.path.join(args.data_path, "kinetics_{}".format(mode))
    sample_name = os.listdir(data_path)

    if (args.debug):
        sample_name = sample_name[:1000]

    # ignore the samples which does not has skeleton sequence
    print("Total number of video: ", len(sample_name))
    sample_id = [name.split('.')[0] for name in sample_name]
    label = np.array(
        [label_info[id]['label_index'] for id in sample_id])
    has_skeleton = np.array(
        [label_info[id]['has_skeleton'] for id in sample_id])
    sample_name = [
        s for h, s in zip(has_skeleton, sample_name) if h
    ]
    label = label[has_skeleton]
    print("Total number of video having skeleton: ", len(sample_name))

    # generate data_sample from each video
    all_data, all_label, all_bbox = [], [], []
    for i, video_name in enumerate(sample_name):

        # read video data
        video_path = os.path.join(data_path, video_name)
        with open(video_path, 'r') as f:
            video_info = json.load(f)

        # get label
        video_label = video_info['label_index']
        assert (label[i] == video_label)

        # print toolbar
        num_frames = len(video_info['data'])
        print_toolbar(i * 1.0 / len(sample_name),
                      '({:>5}/{:<5}) Processing data: '.format(
                          i + 1, len(sample_name)))

        # We process a chunk of T frames at once. We assume pose has 18 joints and 3 features
        # (x,y,z).
        for start_frame in range(0, num_frames - args.T, args.slide):

            data_numpy = np.zeros((3, args.T, 18, args.max_num_person))
            bbox_numpy = np.zeros((4, args.T, args.max_num_person))
            for frame_index, frame_info in enumerate(video_info['data'][start_frame: start_frame + args.T]):
                # frame_index = frame_info['frame_index']
                for m, skeleton_info in enumerate(frame_info["skeleton"]):
                    if m >= args.max_num_person:
                        break

                    pose = np.asarray(skeleton_info['pose'])
                    score = np.asarray(skeleton_info['score'])

                    # calculate bounding box
                    xmin, xmax, ymin, ymax = 20000, -1, 20000, -1
                    for i in range(0, 18):
                        if score[i] != 0:
                            if pose[2 * i] <= xmin: xmin = pose[2 * i]
                            if pose[2 * i] >= xmax: xmax = pose[2 * i]
                            if pose[2 * i + 1] <= ymin: ymin = pose[2 * i + 1]
                            if pose[2 * i + 1] >= ymax: ymax = pose[2 * i + 1]

                    if (xmax - xmin == 0) or (ymax - ymin == 0):
                        continue

                    # gather pose data
                    data_numpy[0, frame_index, :, m] = pose[0::2]
                    data_numpy[1, frame_index, :, m] = pose[1::2]
                    data_numpy[2, frame_index, :, m] = score
                    bbox_numpy[:, frame_index, m] = np.array([xmin, ymin, xmax, ymax])

            # centralization
            data_numpy[0][data_numpy[2] == 0] = 0
            data_numpy[1][data_numpy[2] == 0] = 0

            # collect sample with complete poses only
            for m in range(args.max_num_person):
                if 0 in data_numpy[:, :, :, m]:
                    continue
                else:
                    all_data.append(np.expand_dims(data_numpy[:, :, :, m], axis=3))
                    all_label.append(video_label)
                    all_bbox.append(np.expand_dims(bbox_numpy[:, :, m], axis=2))

    all_data = np.concatenate(all_data, axis=3)   # (C, T, V, N)
    C, T, V, N = all_data.shape
    all_data = np.expand_dims(all_data.transpose(3, 0, 1, 2), axis=4)  # change to shape (N, C, T, V, 1)
    all_bbox = np.concatenate(all_bbox, axis=2)     # (4, T, N)
    all_bbox = all_bbox.transpose(2, 0, 1)          # (N, 4, T)

    print(all_bbox.shape)
    # write to file
    data_out_path = os.path.join(args.out_folder, "{}_data.npy".format(mode))
    fp = open_memmap(
        data_out_path,
        dtype='float32',
        mode='w+',
        shape=all_data.shape)
    fp[...] = all_data
    label_out_path = os.path.join(args.out_folder, "{}_label.pkl".format(mode))
    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_name, all_label, all_bbox), f)

    print("size of data:", fp.shape)
    print("save data to file: ", data_out_path)
    print("save label to file: ", label_out_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Generate Kinectic data for pose reconstruction')
    parser.add_argument(
        '--data_path', default='/home/manhh/github/datasets/kinetics')
    parser.add_argument(
        '--out_folder', default='/home/manhh/github/datasets/processed_data/reconstruction/kinetics')
    parser.add_argument(
        '--T', type=int, default=10, help='number of frames per sample')
    parser.add_argument(
        '--slide', type=int, default=5, help='frame gap ')
    parser.add_argument(
        '--bbox', type=list, default=[-128, -128, 128, 128], help='scaled bounding box')
    parser.add_argument(
        '--max_num_person', type=int, default=5, help='maximum number of person every args.T frames')
    parser.add_argument(
        '--debug', action="store_true", default=False, help='debug mode')

    args = parser.parse_args()

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    print("generate train data: ")
    generate_data(args, "train")
    print("generate val data: ")
    generate_data(args, "val")
