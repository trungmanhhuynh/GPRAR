'''
generate_location_data_jaad.py

Author: Manh Huynh
Last Update: 06/19/2020
'''

import sys
import os
import json


JAAD_DATASET_DIR = "/home/manhh/github/datasets/JAAD"
PROCESSED_DATA_DIR = "/home/manhh/github/datasets/features/JAAD"


sys.path.append(JAAD_DATASET_DIR)
from jaad_data import JAAD


def generate_location_data():
    """
        location_data dictionary structure of each image frame:
        'width': int
        'height': int
        'people':[{
            'person_id': str
            'bbox': [x1, y1, x2, y2]
            'center': [xc, yc]
            'occlusion': int}
        ]

    """

    # read ground truth pedestrian information given by JAAD dataset.
    imdb = JAAD(data_path=JAAD_DATASET_DIR)
    jaad_data = imdb.generate_database()            # look at  JAAD.generate_database() for dictionary structure

    for video in jaad_data:

        print("Processing video :", video)
        processed_video_dir = os.path.join(PROCESSED_DATA_DIR, "location", video)
        if not os.path.exists(processed_video_dir):
            os.makedirs(processed_video_dir)

        num_frames = jaad_data[video]['num_frames']

        for search_frame in range(num_frames):

            location_data = {}
            location_data['width'] = jaad_data[video]['width']
            location_data['height'] = jaad_data[video]['height']
            location_data['people'] = []

            # collect data in each frame
            for ped_id in jaad_data[video]['ped_annotations']:

                if(jaad_data[video]['ped_annotations'][ped_id]['frames'].count(search_frame)):
                    search_frame_index = jaad_data[video]['ped_annotations'][ped_id]['frames'].index(search_frame)

                    temp = {}
                    temp['person_id'] = ped_id
                    temp['bbox'] = jaad_data[video]['ped_annotations'][ped_id]['bbox'][search_frame_index]
                    temp['occlusion'] = jaad_data[video]['ped_annotations'][ped_id]['occlusion'][search_frame_index]
                    temp['center'] = [0.5 * (temp['bbox'][0] + temp['bbox'][2]), 0.5 * (temp['bbox'][1] + temp['bbox'][3])]

                    location_data['people'].append(temp)

            outfile = os.path.join(processed_video_dir, "{:05d}_locations.json".format(search_frame))
            with open(outfile, 'w') as f:
                json.dump(location_data, f)


if __name__ == '__main__':

    generate_location_data()
