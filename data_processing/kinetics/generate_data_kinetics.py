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
    """

        mode: train of val
    """

    # read data and label
    data_path = os.path.join(args.data_path, "kinetics_{}".format(mode))
    label_path = os.path.join(args.data_path, "kinetics_{}_label.json".format(mode))

    sample_name = os.listdir(data_path)
    with open(label_path) as f:
        label_info = json.load(f)

    if(args.debug):
        sample_name = sample_name[:200]

    sample_id = [name.split('.')[0] for name in sample_name]
    label = np.array(
        [label_info[id]['label_index'] for id in sample_id])
    has_skeleton = np.array(
        [label_info[id]['has_skeleton'] for id in sample_id])

    print("Total number of video: ", len(sample_name))

    # ignore the samples which does not has skeleton sequence
    sample_name = [
        s for h, s in zip(has_skeleton, sample_name) if h
    ]
    label = label[has_skeleton]
    print("Total number of video having skeleton: ", len(sample_name))

    # generate data_sample from each video
    all_data, all_label = [], []
    for i, video_name in enumerate(sample_name):
        video_path = os.path.join(data_path, video_name)
        with open(video_path, 'r') as f:
            video_info = json.load(f)

        # get label
        video_label = video_info['label_index']
        assert (label[i] == video_label)

        num_frames = len(video_info['data'])
        # print("{}/{} processing {} num_frames={}".format(i, len(sample_name), video_name, num_frames))

        print_toolbar(i * 1.0 / len(sample_name),
                      '({:>5}/{:<5}) Processing data: '.format(
                          i + 1, len(sample_name)))

        for start_frame in range(0, num_frames - args.T, args.slide):

            data_numpy = np.zeros((3, args.T, 18, args.max_num_person))
            for frame_index, frame_info in enumerate(video_info['data'][start_frame: start_frame + args.T]):
                # frame_index = frame_info['frame_index']
                for m, skeleton_info in enumerate(frame_info["skeleton"]):
                    if m >= args.max_num_person:
                        break

                    pose = skeleton_info['pose']
                    score = skeleton_info['score']
                    data_numpy[0, frame_index, :, m] = pose[0::2]
                    data_numpy[1, frame_index, :, m] = pose[1::2]
                    data_numpy[2, frame_index, :, m] = score

            # centralization
            data_numpy[0:2] = data_numpy[0:2] - 0.5
            data_numpy[0][data_numpy[2] == 0] = 0
            data_numpy[1][data_numpy[2] == 0] = 0

            # collect sample with complete poses only
            for m in range(args.max_num_person):
                if 0 in data_numpy[:, :, :, m]:
                    continue
                else:
                    all_data.append(np.expand_dims(data_numpy[:, :, :, m], axis=3))
                    all_label.append(video_label)

    all_data = np.concatenate(all_data, axis=3)
    C, T, V, N = all_data.shape
    all_data = np.expand_dims(np.reshape(all_data, (N, C, T, V)), axis=4)  # change to shape (N, C, T, V, 1)

    # write to file
    data_out_path = os.path.join(args.out_folder, "{}_data.npy".format(mode))
    fp = open_memmap(
        data_out_path,
        dtype='float32',
        mode='w+',
        shape=all_data.shape)
    fp[...] = all_data
    # print(fp[3])
    # print("data shape: ")
    # print(fp.shape)
    # print("save data to file: ", data_out_path)
    # input("here")

    label_out_path = os.path.join(args.out_folder, "{}_label.pkl".format(mode))
    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_name, all_label), f)
    print("save label to file: ", label_out_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Generate Kinectic data for pose reconstruction')
    parser.add_argument(
        '--data_path', default='/home/manhh/github/datasets/kinetics')
    parser.add_argument(
        '--out_folder', default='data/reconstruction/kinetics')
    parser.add_argument(
        '--T', type=int, default=10, help='number of frames per sample')
    parser.add_argument(
        '--slide', type=int, default=5, help='frame gap ')
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
