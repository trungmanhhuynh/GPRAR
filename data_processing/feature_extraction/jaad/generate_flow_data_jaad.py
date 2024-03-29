import torch
import numpy as np
import argparse
import os
import cv2
import sys

# parameters
FLOWNET2_DIR = "/home/manhh/github/flownet2-pytorch"
# load flow net directory
sys.path.append(FLOWNET2_DIR)  # import flownet directory
from models import FlowNet2  # the path is depended on where you create this module
from utils.frame_utils import read_gen  # the path is depended on where you create this module
# save flow, I reference the code in scripts/run-flownet.py in flownet2-caffe project


def writeFlow(name, flow):
    f = open(name, 'wb')
    f.write('PIEH'.encode('utf-8'))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)
    f.flush()
    f.close()

def generate_data_flow(args):

    FlowNetparser = argparse.ArgumentParser(
        description='generate location data in each frame JAAD')
    FlowNetparser.add_argument(
        '--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    FlowNetparser.add_argument(
        '--rgb_max', type=float, default=255.)
    FlowNetargs = FlowNetparser.parse_args()

    # initial a Net
    net = FlowNet2(FlowNetargs).cuda()
    dict = torch.load(os.path.join(FLOWNET2_DIR, "checkpoints/FlowNet2_checkpoint.pth.tar"))
    net.load_state_dict(dict["state_dict"])

    video_dir = os.listdir(os.path.join(args.data_dir, "images"))

    for video_name in video_dir:

        print("Processing video: {}".format(video_name))

        # Create output folder if not exists
        processed_gridflow_dir = os.path.join(args.output_dir, video_name)
        if not os.path.exists(processed_gridflow_dir):
            os.makedirs(processed_gridflow_dir)

        # process each image in a video
        image_files = os.listdir(os.path.join(args.data_dir, "images", video_name))
        image_files.sort()
        for i in range(0, len(image_files) - 1):

            img_cur = read_gen(os.path.join(args.data_dir, "images", video_name, image_files[i]))
            img_next = read_gen(os.path.join(args.data_dir, "images", video_name, image_files[i + 1]))
            img_cur = cv2.resize(img_cur, dsize=(args.image_width, args.image_height), interpolation=cv2.INTER_CUBIC)
            img_next = cv2.resize(img_next, dsize=(args.image_width, args.image_height), interpolation=cv2.INTER_CUBIC)

            images = [img_cur, img_next]
            images = np.array(images).transpose(3, 0, 1, 2)
            im = torch.from_numpy(images.astype(np.float32)).unsqueeze(0).cuda()

            # process the image pair to obtian the flow
            result = net(im).squeeze()
            data = result.data.cpu().numpy().transpose(1, 2, 0)
            outfile = os.path.join(processed_gridflow_dir, "{}.flo".format(image_files[i][:-4]))

            writeFlow(outfile, data)
            print("Processed frame {} -> {}".format(image_files[i], outfile))

            # write last frame
            if(i == len(image_files) - 2):
                outfile = os.path.join(processed_gridflow_dir, "{}.flo".format(image_files[i + 1][:-4]))
                writeFlow(outfile, data)
                print("Processed frame {} -> {}".format(image_files[i+1], outfile))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='generate location data in each frame JAAD')
    parser.add_argument(
        '--data_dir', default='/home/manhh/github/datasets/JAAD')
    parser.add_argument(
        '--output_dir', default='/home/manhh/github/Traj-STGCNN/data/features/jaad/flow')
    parser.add_argument(
        '--image_width', type=float, default=1280)
    parser.add_argument(
        '--image_height', type=float, default=960)
    args = parser.parse_args()

    generate_data_flow(args)
