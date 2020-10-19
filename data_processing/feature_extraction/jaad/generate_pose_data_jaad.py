import os
import argparse


def generate_pose_data_jaad(args):
    '''
        generate pose feature for jaad datasets using openpose. The script run the following command line
        for each video in the dataset.
        #./build/examples/openpose/openpose.bin --image_dir /home/manhh/github/datasets/JAAD/images \
        --write_json out_folder--display 0 --render_pose 0

        Read README.md for output directory structure of the processed data

        Look https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
        for dictionary structure of each frame data
    '''

    video_dir = os.path.join(args.data_path, "images")
    for video_name in os.listdir(video_dir):

        print("Generate pose data for : ", video_name)

        if args.pose_18:
            out_folder = os.path.join(args.out_folder, "pose_18", video_name)
            cmd = "cd {0} && {1} --model_pose COCO --image_dir {2} --write_json {3} --display 0 --render_pose 0".format(
                args.openpose_path,
                os.path.join(args.openpose_path, "build/examples/openpose/openpose.bin"),
                os.path.join(video_dir, video_name),
                out_folder)
        else:  # pose 25
            out_folder = os.path.join(args.out_folder, "pose_25", video_name)
            cmd = "cd {0} && {1} --image_dir {2} --write_json {3} --display 0 --render_pose 0".format(
                args.openpose_path,
                os.path.join(args.openpose_path, "build/examples/openpose/openpose.bin"),
                os.path.join(video_dir, video_name),
                out_folder)

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        # execute the command
        print(cmd)
        os.system(cmd)
        if args.debug:
            exit(1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='generate pose data')
    parser.add_argument(
        '--openpose_path', default='/home/manhh/github/test/openpose')
    parser.add_argument(
        '--data_path', default='/home/manhh/github/datasets/JAAD')
    parser.add_argument(
        '--out_folder', default='/home/manhh/github/Traj-STGCNN/data/features/jaad')
    parser.add_argument(
        '--pose_18', action="store_true", default=True, help='by default, use 18 keypoints')
    parser.add_argument(
        '--pose_25', action="store_true", default=False, help='use 25 keypoints, it requires to modify your network arch')
    parser.add_argument(
        '--debug', action="store_true", default=False, help='debug mode')
    args = parser.parse_args()

    generate_pose_data_jaad(args)
