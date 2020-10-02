import os
import json


KINETICS_DATASET_DIR = "/home/manhh/github/datasets/kinetics"
PROCESSED_DATA_DIR = "/home/manhh/github/datasets/features/kinetics"

def generate_pose_data_jaad(mode, debug=False):
    """ 
        This scripts convert pose data of kinectics datasets, 
        given by st-gcnn method to the pose data format that will be used
        in this project
        
        Dictionary for each frame data:
        'people':[{
            'person_id': int
            'keypoints': list
            'label': string
            'label_index': int
        ]
    """

    data_path = os.path.join(KINETICS_DATASET_DIR, "kinetics_{}".format(mode))
    video_name = os.listdir(data_path)
    output_path = os.path.join(PROCESSED_DATA_DIR, "{}".format(mode))

    if debug:
        video_name= video_name[0:2]

    video_name = [name.split('.')[0] for name in video_name]
    print("Number of video sequences: ", len(video_name))

    # processing each video
    for i, video in enumerate(video_name): 
        video_path = os.path.join(data_path, "{}.json".format(video))
        with open(video_path, 'r') as f:
            video_info = json.load(f)

        if not os.path.exists(os.path.join(output_path, video)):
            os.makedirs(os.path.join(output_path, video))

        # process each frame data
        for frame_info in video_info['data']:
            frame_index = frame_info['frame_index']
            frame_data = {'people':[]}

            for m, skeleton_info in enumerate(frame_info["skeleton"]):

                person_id = m 
                keypoints = [0]*54      # using 18 keypoints, thus 18*3 
                keypoints[0::3] = skeleton_info['pose'][0::2]   # store x 
                keypoints[1::3] = skeleton_info['pose'][1::2]   # store y 
                keypoints[2::3] = skeleton_info['score']
                label = video_info['label']
                label_index = video_info['label_index']

                person_dict = {'person_id': m, 
                                'keypoints': keypoints, 
                                'label':label, 
                                'label_index': label_index}

                frame_data['people'].append(person_dict)

            outfile = os.path.join(output_path, video, "{:05d}_keypoints.json".format(frame_index))
            with open(outfile, 'w') as f:
                json.dump(frame_data, f)

        print("{}/{} Done processing video {}:".format(i, len(video_name), video))

if __name__ == '__main__':


    print("Generate train data:")
    generate_pose_data_jaad("train", debug=False)
    print("Generate val data:")
    generate_pose_data_jaad("val", debug=False)
