import sys
import os
import json
import argparse


def generate_frame_data(args, JAAD):
    """
        frame_data dictionary structure of each image frame:
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
    imdb = JAAD(data_path=args.data_path)
    jaad_data = imdb.generate_database()            # look at  JAAD.generate_database() for dictionary structure

    total_frames, total_peds = 0, 0
    # jaad_data = jaad_data[0:5]

    for num_v, video in enumerate(jaad_data):

        if args.debug and num_v > 5:
            break

        print("Processing video :", video)
        processed_video_dir = os.path.join(args.out_folder, video)
        if not os.path.exists(processed_video_dir):
            os.makedirs(processed_video_dir)

        video_data = jaad_data[video]
        ped_annotations = video_data['ped_annotations']

        # count number of pedestrians.
        for ped_id in ped_annotations:
            if 'b' in ped_id:
                total_peds += 1 

        # gen data in each frame
        for search_frame in range(video_data['num_frames']):

            frame_data = {}
            frame_data['width'] = video_data['width']
            frame_data['height'] = video_data['height']
            frame_data['people'] = []
            print(frame_data['width'] )
            print(frame_data['width'] )
            input("here")

            for ped_id in ped_annotations:

                if(ped_annotations[ped_id]['frames'].count(search_frame)):
                    search_frame_index = ped_annotations[ped_id]['frames'].index(search_frame)

                    if 'b' not in ped_id:
                        continue

                    temp = {}
                    temp['person_id'] = ped_id
                    temp['bbox'] = ped_annotations[ped_id]['bbox'][search_frame_index]
                    temp['occlusion'] = ped_annotations[ped_id]['occlusion'][search_frame_index]
                    temp['center'] = [0.5 * (temp['bbox'][0] + temp['bbox'][2]), 0.5 * (temp['bbox'][1] + temp['bbox'][3])]
                    temp['action_index'] = ped_annotations[ped_id]['behavior']['action'][search_frame_index]
                    temp['action_label'] = 'standing' if temp['action_index'] == 0 else 'walking'

                    frame_data['people'].append(temp)
                    total_frames += 1

            outfile = os.path.join(processed_video_dir, "{:05d}_locations.json".format(search_frame))
            with open(outfile, 'w') as f:
                json.dump(frame_data, f)
    print("total number of frames: ", total_frames)
    print("total number of pedestrians: ", total_peds)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='generate location data in each frame JAAD')
    parser.add_argument(
        '--data_path', default='/home/manhh/github/datasets/JAAD')
    parser.add_argument(
        '--out_folder', default='/home/manhh/github/Traj-STGCNN/data/features/jaad/location')
    parser.add_argument(
        '--debug', action="store_true", default=False, help='debug mode')
    args = parser.parse_args()

    sys.path.append(args.data_path)
    from jaad_data import JAAD

    generate_frame_data(args, JAAD)
