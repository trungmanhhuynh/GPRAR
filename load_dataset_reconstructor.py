import joblib
import torch
from torch.utils.data import Dataset
from common.utils import calc_mean_variance, std_normalize
import random


class PoseDataset(Dataset):
    """Dataloader for the Pose datasets"""

    def __init__(self, data_file,
                 generate_noisy_pose=False,
                 obs_len=10,
                 pred_len=10,
                 pose_features=3,              # x, y, c
                 num_keypoints=25,                 # using openpose 25 keypoints
                 flip=False,
                 image_width=1280):
        """
        Args:
                data_file: file name of train/val data. Data in data_file has the following structure:
                [{
                        'poses': list ~[traj_len, 75]
                        'video_names': [traj_len]
                        'image_names': [traj_len]
                        'person_ids': [traj_len]
                 }]

        """
        super(PoseDataset, self).__init__()

        # set parapmeters
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.traj_len = obs_len + pred_len
        self.pose_features = pose_features
        self.num_keypoints = num_keypoints
        self.image_width = image_width

        self.generate_noisy_pose = generate_noisy_pose
        self.flip = flip

        # read train/val from joblib file
        self.read_data(data_file)

        # calculate mean/var
        self.pose_mean, self.pose_var = calc_mean_variance(self.poses)                               # pose mean, var ~ [75]
        # print("pose_mean = ", self.pose_mean)
        # print("pose_var = ", self.pose_var)

    def __len__(self):
        return self.num_samples

    def read_data(self, data_file):
        '''
            read train/val data
        '''

        print("Reading datasets ...")
        data = joblib.load(data_file)
        poses, video_names, image_names, person_ids = [], [], [], []
        for sample in data:
            poses.append(sample['poses'])
            video_names.append(sample['video_names'][0])
            image_names.append(sample['image_names'][self.obs_len - 1])
            person_ids.append(sample['person_ids'][0])

        # convert to tensor
        poses = torch.tensor(poses, dtype=torch.float)                                          # ~ (num_samples, traj_len, num_keypoints*pose_features)

        self.poses = poses
        self.video_names = video_names
        self.image_names = image_names
        self.person_ids = person_ids
        self.num_samples = poses.shape[0]

        print(self.poses.shape)

    def generate_noisy_poses(self, poses, occluded_ratio):

        noisy_poses = poses.clone()
        occluded_kpt = random.choices(range(0, self.obs_len * self.num_keypoints), k=int(self.obs_len * self.num_keypoints * occluded_ratio))
        for k in occluded_kpt:
            occ_t = int(k / self.num_keypoints)
            occ_k = int(k % self.num_keypoints)
            noisy_poses[occ_t, occ_k * 3: occ_k * 3 + 3] = 0

        # print(poses)
        # input("jere")
        return noisy_poses

    def __getitem__(self, index):
        """
            pose: size ~ [batch_size, obs_len, 75]
        """
        sample = {
            'poses': self.poses[index, :self.obs_len, :],                           # raw poses
            'poses_gt': self.poses[index, :self.obs_len, :].clone(),        # gt_pose has been linearly interpolated
            'video_names': self.video_names[index],
            'image_names': self.image_names[index],
            'person_ids': self.person_ids[index]
        }

        if (self.flip):
            sample['poses'][:, 0::3] = self.image_width - sample['poses'][:, 0::3]
            sample['poses_gt'][:, 0::3] = self.image_width - sample['poses_gt'][:, 0::3]

        # Reconstructor requires input must be noisy poses
        if (self.generate_noisy_pose):
            occluded_ratio = random.uniform(0, 0.5)
            sample['noisy_poses'] = self.generate_noisy_poses(sample['poses'], occluded_ratio=occluded_ratio)
            sample['present_idx'] = sample['noisy_poses'] != 0
        else:
            sample['noisy_poses'] = sample['poses'].clone()

        # normalize
        sample['poses'] = std_normalize(sample['poses'], self.pose_mean, self.pose_var)
        sample['poses_gt'] = std_normalize(sample['poses_gt'], self.pose_mean, self.pose_var)
        sample['noisy_poses'] = std_normalize(sample['noisy_poses'], self.pose_mean, self.pose_var)

        return sample
