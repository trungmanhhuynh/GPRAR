import joblib
import torch
from torch.utils.data import Dataset
from common.utils import calc_mean_variance, std_normalize


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

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

        # set parapmeters
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.traj_len = obs_len + pred_len
        self.pose_features = pose_features
        self.num_keypoints = num_keypoints
        self.image_width = image_width
        self.flip = flip
        self.xy_indexes = []
        for k in range(0, 25):
            self.xy_indexes.append(3 * k)
            self.xy_indexes.append(3 * k + 1)

        # read train/val from joblib file
        self.read_data(data_file)

        # calculate mean/var
        self.loc_mean, self.loc_var = calc_mean_variance(self.gt_locations)                          # location mean, var ~ [2]
        self.pose_mean, self.pose_var = calc_mean_variance(self.poses)                               # pose mean, var ~ [75]

    def __len__(self):
        return self.num_samples

    def read_data(self, data_file):
        '''
            read train/val data
        '''

        data = joblib.load(data_file)

        poses, gt_locations = [], []
        video_names, image_names, person_ids = [], [], []
        for sample in data:
            poses.append(sample['poses'])
            gt_locations.append(sample['gt_locations'])
            video_names.append(sample['video_names'][0])
            image_names.append(sample['image_names'][self.obs_len - 1])
            person_ids.append(sample['person_ids'][0])

        # convert to tensor
        poses = torch.tensor(poses, dtype=torch.float)                                          # ~ (num_samples, traj_len, keypoints*pose_features)
        gt_locations = torch.tensor(gt_locations, dtype=torch.float)                            # ~ (num_samples, traj_len, 2)

        self.poses = poses[:, :, self.xy_indexes]
        self.gt_locations = gt_locations
        self.video_names = video_names
        self.image_names = image_names
        self.person_ids = person_ids

        self.num_samples = poses.shape[0]

        print(self.poses.shape)

    def __getitem__(self, index):

        sample = {
            'poses': self.poses[index, :self.obs_len, :],                           # raw poses
            'obs_locations': self.gt_locations[index, :self.obs_len, :],
            'gt_locations': self.gt_locations[index, -self.pred_len:, :],
            'video_names': self.video_names[index],
            'image_names': self.image_names[index],
            'person_ids': self.person_ids[index]
        }

        if (self.flip):
            sample['poses'][:, 0::3] = self.image_width - sample['poses'][:, 0::3]
            sample['gt_locations'][:, 0] = self.image_width - sample['gt_locations'][:, 0]
            sample['obs_locations'][:, 0] = self.image_width - sample['obs_locations'][:, 0]

        sample['missing_keypoints'] = sample['poses'] == 0
        sample['poses'] = std_normalize(sample['poses'], self.pose_mean, self.pose_var)
        sample['gt_locations'] = std_normalize(sample['gt_locations'], self.loc_mean, self.loc_var)
        sample['obs_locations'] = std_normalize(sample['obs_locations'], self.loc_mean, self.loc_var)

        return sample
