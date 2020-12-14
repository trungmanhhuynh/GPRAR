import os
import numpy as np
import cv2
import argparse
from torchlight import str2bool

from base_setting import BaseSetting
import tools.utils as utils


class ReconstructionDemo(BaseSetting):

    def start(self):
        self.io.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))

        # initiate
        label_name_path = '/home/manhh/github/datasets/kinetics/label_name.txt'
        with open(label_name_path) as f:
            label_name = f.readlines()
            label_name = [line.rstrip() for line in label_name]
            self.label_name = label_name

        if self.arg.weights is None:
            raise ValueError('Please appoint --weights.')
        self.io.print_log('Model:   {}.'.format(self.arg.model))
        self.io.print_log('Weights: {}.'.format(self.arg.weights))

        self.model.eval()
        loader = self.data_loader['test']

        self.io.print_log('Evaluation Start:')
        ith_sample = 0

        # create res dir
        if (self.arg.gen_video):
            if not os.path.exists(self.arg.res_video_dir):
                os.makedirs(self.arg.res_video_dir)

        for noisy_data, data, label in loader:

            # get data
            data = data.float().to(self.dev)  # (1, channel, frame, joint, person)
            noisy_data = noisy_data.float().to(self.dev)  # (1, channel, frame, joint, person)

            voting_label_name, video_label_name, output, intensity, recOut = self.predict(noisy_data)

            # reconstruction video
            images = self.render_video_v2(noisy_data.squeeze(0), recOut.squeeze(0), data.squeeze(0), voting_label_name,
                                          video_label_name, intensity, video=None)

            if (self.arg.gen_video):
                video_name = os.path.join(self.arg.res_video_dir, '{}.avi'.format(ith_sample))
                out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 2, (1080, 1080))

            if (self.arg.gen_image):
                video_dir = os.path.join(self.arg.res_image_dir, str(ith_sample))
                if not os.path.exists(video_dir):
                    os.makedirs(video_dir)

            for i, image in enumerate(images):
                image = image.astype(np.uint8)
                out.write(image)
                if (self.arg.gen_image):
                    cv2.imwrite(os.path.join(video_dir, '{}.jpg'.format(i)), image)

            out.release()
            ith_sample += 1

            if (self.arg.gen_video):
                print("done gen video: ", video_name)
            if (self.arg.gen_image):
                print("done gen images: ", video_dir)

    def predict(self, data):
        # forward

        output, recOut, feature = self.model.extract_feature(data)
        output = output[0]
        feature = feature[0]
        intensity = (feature * feature).sum(dim=0) ** 0.5
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
        return voting_label_name, video_label_name, output, intensity, recOut

    def render_video(self, data_in, voting_label_name, video_label_name, intensity, video):
        images = utils.visualization.stgcn_visualize(
            data_in,
            self.model.graph.edge,
            intensity, video,
            voting_label_name,
            video_label_name,
            self.arg.height
        )
        return images

    def render_video_v2(self, data_in, data_rec, data_gt, voting_label_name, video_label_name, intensity, video):
        images = utils.visualization.stgcn_visualize_v2(
            data_in,
            data_rec,
            data_gt,
            self.model.graph.edge,
            intensity, video,
            voting_label_name,
            video_label_name,
            self.arg.height
        )
        return images

    @staticmethod
    def get_parser(add_help=False):

        parent_parser = BaseSetting.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Pose Reconstruction Demo')

        parser.add_argument('--height', default=1080, type=int, help='height of frame in the output video.')
        parser.add_argument('--gen_video', action="store_true", default=True, help='generate video')
        parser.add_argument('--gen_image', action="store_true", default=True, help='generate image')
        parser.add_argument('--res_video_dir', type=str, default="results/reconstruction/video",
                            help='result video dir')
        parser.add_argument('--res_image_dir', type=str, default="results/reconstruction/image",
                            help='result images dir')

        return parser
