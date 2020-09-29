
import os
import sys
import pickle
import argparse

import numpy as np
from numpy.lib.format import open_memmap

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from data_utils.kinetics_skeleton.feeder_kinetics import Feeder_kinetics

toolbar_width = 30


def print_toolbar(rate, annotation=''):
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


def gendata(
        data_path,
        label_path,
        data_out_path,
        label_out_path,
        num_person_in=5,  # observe the first 5 persons
        num_person_out=2,  # then choose 2 persons with the highest score
        max_frame=300,
        dsize='large'):

    feeder = Feeder_kinetics(
        data_path=data_path,
        label_path=label_path,
        num_person_in=num_person_in,
        num_person_out=num_person_out,
        window_size=max_frame)

    sample_name = feeder.sample_name
    sample_label = []

    fp = open_memmap(
        data_out_path,
        dtype='float32',
        mode='w+',
        shape=(len(sample_name), 3, max_frame, 18, num_person_out))

    if(dsize == "medium"):
        num_sample_used = int(len(sample_name) * 0.3)        # only 30% of data is used
    else:
        num_sample_used = len(sample_name)
    print("Total number of sample is: ", num_sample_used)

    for i, s in enumerate(sample_name):

        if(i >= num_sample_used):
            break

        data, label = feeder[i]
        print_toolbar(i * 1.0 / num_sample_used,
                      '({:>5}/{:<5}) Processing data: '.format(
                          i + 1, num_sample_used))
        fp[i, :, 0:data.shape[1], :, :] = data
        sample_label.append(label)

    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Kinetics-skeleton Data Converter.')
    parser.add_argument(
        '--data_path', default='../datasets/features/kinetics-skeleton')
    parser.add_argument(
        '--out_folder', default='data/kinetics-skeleton')
    parser.add_argument(
        '--dsize', type=str, default='large')
    arg = parser.parse_args()

    part = ['train', 'val']
    for p in part:
        data_path = '{}/kinetics_{}'.format(arg.data_path, p)
        label_path = '{}/kinetics_{}_label.json'.format(arg.data_path, p)
        data_out_path = '{}/{}_data_{}.npy'.format(arg.out_folder, p, arg.dsize)
        label_out_path = '{}/{}_label_{}.pkl'.format(arg.out_folder, p, arg.dsize)

        if not os.path.exists(arg.out_folder):
            os.makedirs(arg.out_folder)
        gendata(data_path, label_path, data_out_path, label_out_path, dsize=arg.dsize)
