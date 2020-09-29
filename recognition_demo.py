import os
import numpy as np
import cv2
import argparse
from torchlight import str2bool

from recognition_settings import RecognitionSettings
import tools.utils as utils

class RecognitionDemo(RecognitionSettings):

    def start(self):
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))

        # initiate
        label_name_path = '/home/manhh/github/datasets/features/kinetics-skeleton/label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]
            self.label_name = label_name

        # create res dir
        if not os.path.exists(self.arg.res_video_dir):
            os.makedirs(self.arg.res_video_dir)

        if self.arg.weights is None:
            raise ValueError('Please appoint --weights.')
        self.io.print_log('Model:   {}.'.format(self.arg.model))
        self.io.print_log('Weights: {}.'.format(self.arg.weights))

        self.model.eval()
        loader = self.data_loader['test']

        self.io.print_log('Evaluation Start:')
        ith_sample = 0
        for data, label in loader:

            # get data
            data = data.float().to(self.dev)  # (1, channel, frame, joint, person)

            voting_label_name, video_label_name, output, intensity = self.predict(data)

            data_numpy = data.clone()
            data_numpy = data_numpy.squeeze(0)
            images = self.render_video(data_numpy, voting_label_name,
                                       video_label_name, intensity, video=None)

            video_name = os.path.join(self.arg.res_video_dir, '{}.avi'.format(ith_sample))
            out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 15, (1080, 1080))
            for image in images:
                image = image.astype(np.uint8)
                out.write(image)
            out.release()
            print("done processing video: ", video_name)
            ith_sample += 1

    def predict(self, data):
        # forward
        output, feature = self.model.extract_feature(data)
        output = output[0]
        feature = feature[0]
        intensity = (feature * feature).sum(dim=0)**0.5
        intensity = intensity.cpu().detach().numpy()

        # get result
        # classification result of the full sequence
        voting_label = output.sum(dim=3).sum(
            dim=2).sum(dim=1).argmax(dim=0)
        voting_label_name = self.label_name[voting_label]
        # classification result for each person of the latest frame
        num_person = data.size(4)
        latest_frame_label = [output[:, :, :, m].sum(
            dim=2)[:, -1].argmax(dim=0) for m in range(num_person)]
        latest_frame_label_name = [self.label_name[l]
                                   for l in latest_frame_label]

        num_person = output.size(3)
        num_frame = output.size(1)
        video_label_name = list()
        for t in range(num_frame):
            frame_label_name = list()
            for m in range(num_person):
                person_label = output[:, t, :, m].sum(dim=1).argmax(dim=0)
                person_label_name = self.label_name[person_label]
                frame_label_name.append(person_label_name)
            video_label_name.append(frame_label_name)
        return voting_label_name, video_label_name, output, intensity

    def render_video(self, data_numpy, voting_label_name, video_label_name, intensity, video):
        images = utils.visualization.stgcn_visualize(
            data_numpy,
            self.model.graph.edge,
            intensity, video,
            voting_label_name,
            video_label_name,
            self.arg.height
        )
        return images

    @staticmethod
    def get_parser(add_help=False):

        parent_parser = RecognitionSettings.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        parser.add_argument('--visualize', type=str2bool, default=False, help='visualize the results')
        parser.add_argument('--height', default=1080, type=int, help='height of frame in the output video.')
        parser.add_argument('--res_video_dir', type=str, default="results/recognition/video", help='result video dir')

        return parser
