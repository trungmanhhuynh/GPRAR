import os
import json
import numpy as np
import argparse
from scipy.spatial import distance
from munkres import Munkres, print_matrix


def generate_cost_matrix(pose_data, location_data):
    # Extract center position of all pedestrian within a frame
    # from pose data and location data

    pose_center = []
    pose_location = np.zeros(2)
    for ped in pose_data["people"]:
        pose = np.array(ped['pose_keypoints_2d'])

        left_hip = pose[24:26]
        right_hip = pose[33:35]
        num_keypoints = 18
        visible_keypt = 0

        assert num_keypoints == len(ped['pose_keypoints_2d']) / 3

        if (0 not in left_hip and 0 not in right_hip):
            pose_location = 0.5 * (right_hip + left_hip)
        elif (0 not in left_hip):
            pose_location = left_hip
        elif (0 not in right_hip):
            pose_location = right_hip
        else:
            # average all visible keypoints for center location
            for k in range(0, num_keypoints):
                if (0 not in pose[k * 3:(k + 1) * 3]):
                    visible_keypt += 1
                    pose_location += pose[k * 3:k * 3 + 2]
            pose_location = pose_location / visible_keypt

        pose_center.append(pose_location)

    location_center = []
    for ped in location_data['people']:
        location_center.append(np.array(ped['center']))

    # build cost matrix as a numpy array ~ (num_pose, num_gt_ped)
    num_pose = len(pose_center)
    num_gt_ped = len(location_center)
    cost_matrix = np.zeros((num_pose, num_gt_ped))  # could be rectangular matrix
    # pose is on row
    # gt people is on col
    for i in range(cost_matrix.shape[0]):
        for j in range(cost_matrix.shape[1]):
            cost_matrix[i][j] = distance.euclidean(pose_center[i], location_center[j])

    return cost_matrix.tolist()


def generate_pose_id(args):
    """
        function to assign 'person_id' for each pose
    """

    video_dir = os.listdir(os.path.join(args.pose_path))
    if (args.debug):
        video_dir = video_dir[:10]

    for video_name in video_dir:

        print("processing video :", video_name)

        if not os.path.exists(os.path.join(args.out_folder, video_name)):
            os.makedirs(os.path.join(args.out_folder, video_name))

        # Assign person_id for each pose in a frame
        frame_list = os.listdir(os.path.join(args.pose_path, video_name))

        for frame_name in frame_list:
            frame_number = int(frame_name[:6])
            # read pose+location data

            with open(os.path.join(args.pose_path, video_name, "{:06d}_keypoints.json".format(frame_number)), "r") as f:
                pose_data = json.load(f)

            exist_location = True
            if os.path.exists(os.path.join(args.location_path, video_name, "{:06d}_locations.json".format(frame_number))):
                with open(os.path.join(args.location_path, video_name, "{:06d}_locations.json".format(frame_number)), "r") as f:
                    location_data = json.load(f)
            else:
                exist_location = False
                print(os.path.join(args.location_path, video_name, "{:06d}_locations.json".format(frame_number)), "does not exist")

            if pose_data["people"] and exist_location:

                cost_matrix = generate_cost_matrix(pose_data, location_data)
                # print("num_pose={}, num_gt_peds ={}".format(len(cost_matrix), len(cost_matrix[0])))
                # print_matrix(cost_matrix, msg='Lowest cost through this matrix:')

                # run matching
                m = Munkres()
                indexes = m.compute(cost_matrix)  # (pose_idx, actual_person_id)

                for row, col in indexes:

                    # avoid false assignment when the distance is big
                    if (cost_matrix[row][col] > 300):
                        continue

                    # ped_id of pose at row matched with location at col
                    pose_data["people"][row]['person_id'] = location_data['people'][col]['person_id']
                    pose_data["people"][row]['action_label'] = location_data['people'][col][
                        'action_label']  # also add action label
                    pose_data["people"][row]['action_index'] = location_data['people'][col][
                        'action_index']  # also add action label

            # save to file
            outfile = os.path.join(args.out_folder, video_name, "{:06d}_keypoints.json".format(frame_number))
            with open(outfile, 'w') as f:
                json.dump(pose_data, f)

    print("processing done")


def test_munkres():
    """
        Testing munkres algorithm
    """

    # test square matrix
    print("Test square matrix")
    matrix = [[5, 9, 1],
              [10, 3, 2],
              [8, 7, 4]]
    m = Munkres()
    indexes = m.compute(matrix)
    print_matrix(matrix, msg='Lowest cost through this matrix:')
    total = 0
    for row, column in indexes:
        value = matrix[row][column]
        total += value
        print(f'({row}, {column}) -> {value}')
    print(f'total cost: {total}')

    print("Test non-square matrix 1")
    matrix = [[5, 1],
              [10, 2],
              [8, 4]]
    m = Munkres()
    indexes = m.compute(matrix)
    print_matrix(matrix, msg='Lowest cost through this matrix:')
    total = 0
    for row, column in indexes:
        value = matrix[row][column]
        total += value
        print(f'({row}, {column}) -> {value}')
    print(f'total cost: {total}')

    print("Test non-square matrix 2")
    matrix = [[5, 10, 1],
              [1, 4, 2]]
    m = Munkres()
    indexes = m.compute(matrix)
    print_matrix(matrix, msg='Lowest cost through this matrix:')
    total = 0
    for row, column in indexes:
        value = matrix[row][column]
        total += value
        print(f'({row}, {column}) -> {value}')
    print(f'total cost: {total}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Assign id to pose data')
    parser.add_argument(
        '--pose_path', default='/home/manhh/github/datasets/processed_data/features/titan/pose_18')
    parser.add_argument(
        '--location_path', default='/home/manhh/github/datasets/processed_data/features/titan/location')
    parser.add_argument(
        '--out_folder', default='/home/manhh/github/datasets/processed_data/features/titan/pose_18_id')
    parser.add_argument(
        '--debug', action="store_true", default=False, help='debug mode')
    args = parser.parse_args()

    # test munkres
    # test_munkres()

    generate_pose_id(args)
