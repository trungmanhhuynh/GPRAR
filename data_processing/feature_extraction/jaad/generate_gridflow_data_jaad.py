import os
import numpy as np
import json
from cvbase.optflow.io import read_flow


# Define input argurments
INPUT_FLOW_DIR = "/home/manhh/github/Traj-STGCNN/processed_data/JAAD/flow"
INPUT_GRIRFLOW_DIR = "/home/manhh/github/Traj-STGCNN/processed_data/JAAD/gridflow"
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 960
GRID_DIM_X = 4
GRID_DIM_Y = 3

if __name__ == "__main__":

    video_dir = os.path.join(INPUT_FLOW_DIR)
    for video_name in os.listdir(video_dir):

        print("Processing video: ", video_name)

        flow_files = os.listdir(os.path.join(INPUT_FLOW_DIR, video_name))
        flow_files.sort()
        flow_data = {}  # dictionary with structor {framenumber: [x,y,x,y .... x,y]}

        for flow_file in flow_files:

            input_file = os.path.join(INPUT_FLOW_DIR, video_name, flow_file)
            # print("Processing flow file {}".format(input_file))
            flow = read_flow(input_file)  # [height, width, 2]
            framenumber, _ = os.path.splitext(flow_file)

            frame_flow = []
            for grid_tl_x in range(0, IMAGE_WIDTH, int(IMAGE_WIDTH / GRID_DIM_X)):
                for grid_tl_y in range(0, IMAGE_HEIGHT, int(IMAGE_HEIGHT / GRID_DIM_Y)):

                    grid_flow = flow[grid_tl_y:grid_tl_y + int(IMAGE_WIDTH / GRID_DIM_X),
                                     grid_tl_x:grid_tl_x + int(IMAGE_HEIGHT / GRID_DIM_Y)]
                    avg_grid_flow = np.sum(grid_flow, axis=(0, 1)) / ((IMAGE_WIDTH / GRID_DIM_X) * (IMAGE_HEIGHT / GRID_DIM_Y))

                    frame_flow.append(avg_grid_flow[0].astype('float'))
                    frame_flow.append(avg_grid_flow[1].astype('float'))

            flow_data[str(int(framenumber))] = frame_flow

            # write to file
            output_file = os.path.join(INPUT_GRIRFLOW_DIR, video_name, "{}_gridflow.json".format(framenumber))
            # print("Write to file {}".format(output_file))

            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))
            with open(output_file, "w") as f:
                json.dump(flow_data, f)

        print("done")
