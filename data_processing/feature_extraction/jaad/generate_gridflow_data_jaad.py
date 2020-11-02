import os
import numpy as np
import json
import argparse
from cvbase.optflow.io import read_flow


def generate_gridflow_data(args):

    video_dir = os.path.join(args.input_dir)
    for video_name in os.listdir(video_dir):

        print("Processing video: ", video_name)

        flow_files = os.listdir(os.path.join(args.input_dir, video_name))
        flow_files.sort()
        flow_data = {}  # dictionary with structor {framenumber: [x,y,x,y .... x,y]}

        for flow_file in flow_files:

            input_file = os.path.join(args.input_dir, video_name, flow_file)
            # print("Processing flow file {}".format(input_file))
            flow = read_flow(input_file)  # [height, width, 2]
            framenumber, _ = os.path.splitext(flow_file)

            frame_flow = []
            for grid_tl_x in range(0, args.image_width, int(args.image_width / args.grid_dim_x)):
                for grid_tl_y in range(0, args.image_height, int(args.image_height / args.grid_dim_y)):

                    grid_flow = flow[grid_tl_y:grid_tl_y + int(args.image_width / args.grid_dim_x),
                                     grid_tl_x:grid_tl_x + int(args.image_height / args.grid_dim_y)]
                    avg_grid_flow = np.sum(grid_flow, axis=(0, 1)) / ((args.image_width / args.grid_dim_x) * (args.image_height / args.grid_dim_y))

                    frame_flow.append(avg_grid_flow[0].astype('float'))
                    frame_flow.append(avg_grid_flow[1].astype('float'))

            flow_data[str(int(framenumber))] = frame_flow

            # write to file
            output_file = os.path.join(args.output_dir, video_name, "{}_gridflow.json".format(framenumber))
            # print("Write to file {}".format(output_file))

            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))
            with open(output_file, "w") as f:
                json.dump(flow_data, f)

        print("done")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='generate location data in each frame JAAD')
    parser.add_argument(
        '--input_dir', default='data/features/jaad/flow')
    parser.add_argument(
        '--output_dir', default='data/features/jaad/gridflow')
    parser.add_argument(
        '--grid_dim_x', type=float, default=4)
    parser.add_argument(
        '--grid_dim_y', type=float, default=3)
    parser.add_argument(
        '--image_width', type=float, default=1280)
    parser.add_argument(
        '--image_height', type=float, default=960)
    parser.add_argument(
        '--debug', action="store_true", default=False, help='debug mode')
    args = parser.parse_args()

    generate_gridflow_data(args)
