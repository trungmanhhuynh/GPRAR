import joblib
import torch
import random
from torch.utils.data import Dataset
from reconstructor.utils import calc_mean_variance, std_normalize


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(self, data_file,
                 obs_len=10,
                 pred_len=10,
                 flip=False,
                 image_width=1280):
        """
        Args:
            data_file: file name of train/val data. Data in data_file has the following structure:
            [{
                'video_names': [traj_len]
                'image_names': [traj_len]
                'person_ids': [traj_len]
                'poses': list ~[traj_len, 75]
                'imputed_poses': list ~[traj_len, 75]
                'gt_locations': list ~[traj_len, 2]
                'bboxes':  list ~ [traj_len, 4]
             }]

        """
        super(TrajectoryDataset, self).__init__()

        random.seed(3)

        # set parapmeters
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.traj_len = obs_len + pred_len
        self.pose_features = 3              # x, y, c
        self.keypoints = 25                 # using openpose 25 keypoints
        self.image_width = image_width

        # read train/val from joblib file
        self.read_data(data_file)
        print(self.imputed_poses.shape)

        # print(self.imputed_poses[0, 0, :])
        # self.imputed_poses = self.generate_occluded_pose(self.imputed_poses)
        # print(self.imputed_poses[0, 0, :])
        # input("here")
        # normalize data
        self.normalize_data()

    def __len__(self):
        return self.imputed_poses.shape[0]

    def read_data(self, data_file):
        '''
            read train/val data
        '''

        data = joblib.load(data_file)

        imputed_poses, bboxes, video_names, image_names, person_ids = [], [], [], [], []
        for sample in data:
            imputed_poses.append(sample['imputed_poses'])
            bboxes.append(sample['bboxes'])
            video_names.append(sample['video_names'][0])
            image_names.append(sample['image_names'][self.obs_len - 1])
            person_ids.append(sample['person_ids'][0])

        # convert to tensor
        imputed_poses = torch.tensor(imputed_poses, dtype=torch.float)                         # ~ (num_samples, traj_len, keypoints*pose_features)
        bboxes = torch.tensor(bboxes, dtype=torch.float)                                        # ~ (num_samples, traj_len, 4)

        self.imputed_poses = imputed_poses
        self.bboxes = bboxes
        self.video_names = video_names
        self.image_names = image_names
        self.person_ids = person_ids

    def generate_occluded_pose(self, pose):

        # set a random number of keypoints to zero

        # occluded_rate = random.uniform(0, 0.5)
        occluded_rate = 0.1
        occluded_kpt = random.choices(range(0, self.obs_len * self.keypoints), k=int(self.obs_len * self.keypoints * occluded_rate))

        occ_t, occ_k = [], []
        for k in occluded_kpt:
            occ_t.append(int(k / (self.obs_len * self.keypoints)))
            occ_k.append(k % (self.obs_len * self.keypoints))

        for t in occ_t:
            for k in occ_k:
                pose[t, k * 3: k * 3 + 3] = 0
                # pose[:, t, k * 3: k * 3 + 3] = 0

        return pose

    def normalize_data(self):
        '''
            Calculate mean/var for each data feature and normalize
        '''

        self.pose_mean, self.pose_var = calc_mean_variance(self.imputed_poses)                               # pose mean, var ~ [75]
        self.imputed_poses = std_normalize(self.imputed_poses, self.pose_mean, self.pose_var)

    def __getitem__(self, index):
        """
            pose: size ~ [batch_size, pose_features, obs_len, keypoints, instances] or [N, C, T, V, M]
        """
        sample = {
            'poses': self.imputed_poses[index, :self.obs_len, :],
            'poses_gt': self.imputed_poses[index, :self.obs_len, :],
            'bboxes': self.bboxes[index, :self.obs_len, :],
            'video_names': self.video_names[index],
            'image_names': self.image_names[index],
            'person_ids': self.person_ids[index]
        }

        return sample
