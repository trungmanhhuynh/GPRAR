import sys
import os
import csv
import json
import numpy as np
import argparse


def generate_frame_data(args):
    """ Function to generate location data for TITAN dataset.

        Args:

        Returns:
             location_data (dict): data for each frame. It is written to json file
                'width': (int) frame width
                'height': (int) frame height
                'people': (list of dict):
                    'person_id': (str) pedestrian's id
                    'bbox': (list) bounding box coordinate [xtop, ytop, xleft, yleft]
                    'center': (list) center coordinate [xcenter, ycenter]
                    'action_label': (str) one of 9 classes of atomic action [standing, running, bending,
                     kneeling, walking, sitting, squatting, jumping, laying down, none of the above]
                    'action_index': (int) range (0, 8) corresponds to 9 classes
                    }
                ]
    """
    action_index_dict = {'standing': 0, 'running': 1, 'bending': 2,
                         'kneeling': 3, 'walking': 4, 'sitting': 5, 'squatting': 6, 'jumping': 7, 'laying down': 8,
                         'none of the above': 9}

    # read csv data of each clip
    csv_file_list = os.listdir(args.data_path)
    if args.debug:
        csv_file_list = csv_file_list[:10]

    for csv_file in csv_file_list:
        processed_video_dir = os.path.join(args.out_folder, csv_file[:-4])
        print("Processing:  ", csv_file[:-4])
        if not os.path.exists(processed_video_dir):
            os.makedirs(processed_video_dir)


        data = []
        with open(os.path.join(args.data_path, csv_file), mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dict_from_csv = dict(row)
                data.append(dict_from_csv)

        img_name_list = [i['frames'] for i in data]
        img_name_list = np.unique(img_name_list)

        # create location data dictionary for each frame
        for img_name in img_name_list:
            location_data = {'width': args.width, 'height': args.height, 'people': []}
            for fdata in data:
                if fdata['frames'] != img_name:
                    continue
                if fdata['label'] != 'person':
                    continue
                fdata['left'] = float(fdata['left'])
                fdata['top'] = float(fdata['top'])
                fdata['height'] = float(fdata['height'])
                fdata['width'] = float(fdata['width'])

                person_data = {'person_id': fdata['obj_track_id'],
                               'bbox': [fdata['left'], fdata['top'], fdata['left'] + fdata['width'],
                                        fdata['top'] + fdata['height']],
                               'center': [fdata['left'] + 0.5 * fdata['width'],
                                          fdata['top'] + 0.5 * fdata['height']],
                               'action_label': fdata['attributes.Atomic Actions'],
                               'action_index': action_index_dict[fdata['attributes.Atomic Actions']]}
                location_data['people'].append(person_data)

            outfile = os.path.join(processed_video_dir, "{:05d}_locations.json".format(int(img_name[:-4])))
            with open(outfile, 'w') as f:
                 json.dump(location_data, f)
    print("Done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='generate location data in each frame JAAD')
    parser.add_argument(
        '--data_path', default='/home/manhh/github/datasets/titan_data/dataset/titan_0_4')
    parser.add_argument(
        '--out_folder', default='/home/manhh/github/datasets/processed_data/features/titan/location')
    parser.add_argument(
        '--width', type=int, default=2704)
    parser.add_argument(
        '--height', default=1520)
    parser.add_argument(
        '--debug', action="store_true", default=False, help='debug mode')
    args = parser.parse_args()

    generate_frame_data(args)
