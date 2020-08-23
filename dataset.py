import joblib
import torch
from torch.utils.data import Dataset
from common.utils import calc_mean_variance, std_normalize


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

        # set parapmeters
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.traj_len = obs_len + pred_len
        self.pose_features = 3              # x, y, c
        self.keypoints = 25                 # using openpose 25 keypoints

        # read train/val from joblib file
        self.read_data(data_file)

        if(flip):
            self.augment_flip_data()

        self.normalize_data()

    def __len__(self):
        return self.num_samples

    def read_data(self, data_file):
        '''
            read train/val data
        '''

        data = joblib.load(data_file)

        poses, imputed_poses, gt_locations, bboxes = [], [], [], []
        video_names, image_names, person_ids = [], [], []
        for sample in data:
            poses.append(sample['poses'])
            gt_locations.append(sample['locations'])
            bboxes.append(sample['bboxes'])
            video_names.append(sample['video_names'][0])
            image_names.append(sample['image_names'][self.obs_len - 1])
            person_ids.append(sample['person_ids'][0])

        # convert to tensor
        poses = torch.tensor(poses, dtype=torch.float)                                          # ~ (num_samples, traj_len, keypoints*pose_features)
        imputed_poses = torch.tensor(imputed_poses, dtype=torch.float)                          # ~ (num_samples, traj_len, keypoints*pose_features)
        gt_locations = torch.tensor(gt_locations, dtype=torch.float)                            # ~ (num_samples, traj_len, 2)
        bboxes = torch.tensor(bboxes, dtype=torch.float)                                        # ~ (num_samples, traj_len, 4)

        self.poses = poses
        self.gt_locations = gt_locations
        self.video_names = video_names
        self.image_names = image_names
        self.person_ids = person_ids

        self.num_samples = poses.shape[0]

        print(self.poses.shape)

    def augment_flip_data(self):
        '''
                augment data by horizontal flip
        '''

        f_poses = self.poses.clone()
        f_poses[:, :, 0::3] = image_width - f_poses[:, :, 0::3]
        f_gt_locations = self.gt_locations.clone()
        f_gt_locations[:, :, 0::3] = image_width - f_gt_locations[:, :, 0::3]
        f_bboxes = self.bboxes.clone()
        f_bboxes[:, :, 0::3] = image_width - f_bboxes[:, :, 0::3]

        self.poses = torch.cat([poses, f_poses], dim=0)
        self.gt_locations = torch.cat([gt_locations, f_gt_locations], dim=0)
        self.bboxes = torch.cat([bboxes, f_bboxes], dim=0)

        self.video_names = self.video_names + self.video_names
        self.image_names = self.image_names + self.image_names
        self.person_ids = self.person_ids + self.person_ids

    def normalize_data(self):
        '''
                Calculate mean/var for each data feature and normalize
        '''
        # calculate mean/var
        self.loc_mean, self.loc_var = calc_mean_variance(self.gt_locations)                          # location mean, var ~ [2]
        self.pose_mean, self.pose_var = calc_mean_variance(self.poses)                               # pose mean, var ~ [75]

        # normalize data
        self.gt_locations = std_normalize(self.gt_locations, self.loc_mean, self.loc_var)
        self.poses = std_normalize(self.poses, self.pose_mean, self.pose_var)

    def __getitem__(self, index):
        """
                pose: size ~ [batch_size, pose_features, obs_len, keypoints, instances] or [N, C, T, V, M]
        """
        sample = {
            'poses': self.poses[index, :self.obs_len, :],
            'poses_gt': self.poses[index, -self.pred_len:, :],
            'gt_locations': self.gt_locations[index, -self.pred_len:, :],
            'video_names': self.video_names[index],
            'image_names': self.image_names[index],
            'person_ids': self.person_ids[index]
        }

        return sample
