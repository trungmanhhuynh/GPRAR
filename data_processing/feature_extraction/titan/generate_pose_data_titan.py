import os
import time
import argparse
from tqdm import tqdm


def generate_pose_data_titan(args):
    """
        generate pose feature for jaad datasets using openpose. The script run the following command line
        for each video in the dataset.
        Use system (not any conda env) to run the following command:
        #./build/examples/openpose/openpose.bin --image_dir /home/manhh/github/datasets/JAAD/images \
        --write_json out_folder --display 0 --render_pose 0

        Read README.md for output directory structure of the processed data

        Look https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
        for dictionary structure of each frame data
    """

    video_dir = os.path.join(args.data_path)
    list_video = os.listdir(video_dir)

    # manually select videos if error occurred
    # list_video = ['clip_669','clip_382','clip_281','clip_525','clip_39','clip_464','clip_283', 'clip_505', 'clip_370',\
    #               'clip_476', 'clip_25', 'clip_563', 'clip_320', 'clip_135', 'clip_314', 'clip_303', 'clip_307', 'clip_725', \
    #               'clip_776', 'clip_9', 'clip_331', 'clip_507', 'clip_506', 'clip_304', 'clip_437', 'clip_326', 'clip_207']
    # list_video = ['clip_437']

    if args.debug:
        list_video = list_video[:2]

    pbar = tqdm(total=len(list_video))
    for video_name in list_video:
        pbar.update(1)

        print("Generate pose data for : ", video_name)

        if args.pose_18:
            out_folder = os.path.join(args.out_folder, "pose_18", video_name)
            if args.write_images:
                if not os.path.exists(os.path.join(args.plot_path, video_name)):
                    os.makedirs(os.path.join(args.plot_path, video_name))

                cmd = "cd {0} && {1} --model_pose COCO --image_dir {2} --write_json {3} --write_images {4} --write_images_format png --display 0".format(
                    args.openpose_path,
                    os.path.join(args.openpose_path, "build/examples/openpose/openpose.bin"),
                    os.path.join(video_dir, video_name, 'images'),
                    out_folder,
                    os.path.join(args.plot_path, video_name))
            else:
                cmd = "cd {0} && {1} --model_pose COCO --image_dir {2} --write_json {3} --display 0 --render_pose 0".format(
                    args.openpose_path,
                    os.path.join(args.openpose_path, "build/examples/openpose/openpose.bin"),
                    os.path.join(video_dir, video_name, 'images'),
                    out_folder)
        else:  # pose 25
            out_folder = os.path.join(args.out_folder, "pose_25", video_name)
            cmd = "cd {0} && {1} --image_dir {2} --write_json {3} --display 0 --render_pose 0".format(
                args.openpose_path,
                os.path.join(args.openpose_path, "build/examples/openpose/openpose.bin"),
                os.path.join(video_dir, video_name, 'images'),
                out_folder)

        if not os.path.exists(out_folder):
            os.makedirs(out_folder)


        # execute the command
        print(cmd)
        os.system(cmd)

        time.sleep(5)
    pbar.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='generate pose data')
    parser.add_argument(
        '--openpose_path', default='/home/manhh/github/test/openpose')
    parser.add_argument(
        '--data_path', default='/home/manhh/github/datasets/titan/dataset/images_anonymized')
    parser.add_argument(
        '--out_folder', default='/home/manhh/github/datasets/processed_data/features/titan')
    parser.add_argument(
        '--plot_path', default='/home/manhh/github/Traj-STGCNN/pose_path')
    parser.add_argument(
        '--write_images', action="store_true", default=False, help='plot pose on images')
    parser.add_argument(
        '--pose_18', action="store_true", default=True, help='by default, use 18 keypoints')
    parser.add_argument(
        '--pose_25', action="store_true", default=False, help='use 25 keypoints, it requires to modify your network arch')
    parser.add_argument(
        '--debug', action="store_true", default=False, help='debug mode')
    args = parser.parse_args()

    generate_pose_data_titan(args)
