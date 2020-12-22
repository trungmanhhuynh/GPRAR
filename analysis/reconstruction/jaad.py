import os
import sys
import pickle
import argparse
import yaml
import cv2
import numpy as np
from torchlight import str2bool
from analysis.utils.visualization import gen_reconstructed_pose_on_image
from net.utils.graph import Graph
from tqdm import tqdm


class JAADReconstructionAnalysis():
    def __init__(self, argv=None):
        self.load_arg(argv)

    def start(self):
        print("start analysis ...")
        if self.args.gen_image_res:
            self.generate_image_result()

    def generate_image_result(self):
        """ Plot reconstruction result on images
        Args:

        Shape:
        """

        print("start generate images...")

        # load result file
        with open(self.args.res_file, 'rb') as f:
            result = pickle.load(f)

        G = Graph(layout='openpose', strategy='spatial')
        if self.args.debug:
            result = result[:100]

        n_sample = len(result)
        print("Number of samples:", n_sample)

        pbar = tqdm(total=len(range(0, n_sample, 10)))
        for i in range(0, n_sample, 10):

            pbar.update(1)
            # generate images
            in_pose = result[i]['in_pose'].squeeze(3)  # (C, T, V)
            out_pose = result[i]['out_pose'].squeeze(3)  # (C, T, V)
            gt_pose = result[i]['gt_pose'].squeeze(3)  # (C, T, V)

            occ_ratio = self.calculate_occlusion_ratio(in_pose)

            if occ_ratio < 0.3:
                continue

            bbox = result[i]['bbox']  # (T, 4)
            video_name = result[i]['video_name']
            st_image_name = result[i]['image_name']
            image_path = os.path.join(self.args.image_dir, video_name)

            images = gen_reconstructed_pose_on_image(in_pose, out_pose, gt_pose, G.edge, bbox,
                                                     image_path=image_path, st_mage_name=st_image_name)
            # plot
            sample_dir = os.path.join(self.args.res_image_dir, str(i))
            if not os.path.exists(sample_dir):
                os.makedirs(sample_dir)

            for i, image in enumerate(images):
                image = image.astype(np.uint8)
                cv2.imwrite(os.path.join(sample_dir, '{}_{}_{}.jpg'.format(i, video_name, st_image_name)), image)

        pbar.close()

    def load_arg(self, argv=None):

        parser = self.get_parser()

        # load arg form config file
        p = parser.parse_args(argv)
        if p.config is not None:
            # load config file
            with open(p.config, 'r') as f:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)

            # update parser from config file
            key = vars(p).keys()
            for k in default_arg.keys():
                if k not in key:
                    print('Unknown Arguments: {}'.format(k))
                    print(k)
                    assert k in key

            parser.set_defaults(**default_arg)

        self.args = parser.parse_args(argv)
        print(self.args)

    def calculate_occlusion_ratio(self, pose):
        """
            Args:

            Shapes:
                pose: (C, T, V)
            Returns:

        """
        C, T, V = pose.shape
        temp = pose[0] + pose[1] + pose[2]  # (T,V)
        non_zeros = np.count_nonzero(temp)

        return 1 - non_zeros / (T * V)

    def get_parser(add_help=False):

        parser = argparse.ArgumentParser(add_help=add_help, description='Pose Reconstruction Analysis')
        parser.add_argument('-c', '--config', default=None,
                            help='path to the configuration file')
        parser.add_argument('--image_dir', type=str, default='/home/manhh/github/datasets/JAAD/images',
                            help='original images dir')
        parser.add_argument('--res_image_dir', type=str, default='./results/reconstruction/jaad/image/val',
                            help='result images dir')
        parser.add_argument('--gen_image_res', type=str2bool, default=True,
                            help='generate image results')
        parser.add_argument('--res_file', type=str, default='./work_dir/reconstruction/jaad/test_result.pkl',
                            help='result file path')
        parser.add_argument('--debug', type=str2bool, default=False,
                            help='debug with 10 samples')

        return parser


if __name__ == '__main__':
    r = JAADReconstructionAnalysis(sys.argv[1:])
    r.start()
