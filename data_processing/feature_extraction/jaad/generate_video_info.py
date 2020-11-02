import sys
import csv
import os
import argparse


def generate_video_info(args, JAAD):
    """
    """
    # read ground truth pedestrian information given by JAAD dataset.
    imdb = JAAD(data_path=args.data_path)
    jaad_data = imdb.generate_database()            # look at  JAAD.generate_database() for dictionary structure

    video_info = []
    for num_v, video in enumerate(jaad_data):

        if args.debug and num_v > 5:
            break

        print("Processing video :", video)

        video_data = jaad_data[video]
        print(video_data['width'], 'x', video_data['height'])
        video_info.append([video, video_data['width'], video_data['height']])

    with open(filename, 'r') as data:
        for line in csv.reader(data):
            print(line)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='generate metadata into text file')
    parser.add_argument(
        '--data_path', default='/home/manhh/github/datasets/JAAD')
    parser.add_argument(
        '--out_folder', default='/home/manhh/github/Traj-STGCNN/data/features/jaad')
    parser.add_argument(
        '--debug', action="store_true", default=False, help='debug mode')
    args = parser.parse_args()

    sys.path.append(args.data_path)
    from jaad_data import JAAD

    generate_video_info(args, JAAD)
