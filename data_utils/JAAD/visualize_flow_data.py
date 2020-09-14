import cv2
import os

import cvbase as cvb
from cvbase.optflow.io import read_flow
from cvbase.image import rgb2bgr
import scipy.misc
import argparse


if __name__ == '__main__':

    # obtain the necessary args for construct the flownet framework
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_flow_dir", type=str, help="directory of input flows")
    parser.add_argument("--output_image_dir", type=str, help="directory of output images")
    args = parser.parse_args()

    # Create output folder if not exists
    if not os.path.exists(args.output_image_dir):
        os.makedirs(args.output_image_dir)

    flow_files = os.listdir(args.input_flow_dir)
    for i in range(0, len(flow_files)):

        print("Processing frame {} ".format(flow_files[i]), end="->")

        # to visualize a flow file
        flow_data = read_flow(os.path.join(args.input_flow_dir, flow_files[i]))
        # to visualize a loaded flow map
        flow_img = cvb.flow2rgb(flow_data)

        image_file = os.path.join(args.output_image_dir, "flow_image_{0:06d}.png".format(i))
        print(image_file)
        cv2.imwrite(image_file, rgb2bgr(255 * flow_img))
