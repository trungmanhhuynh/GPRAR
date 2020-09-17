import joblib
import torch
from torch.utils.data import Dataset
from common.utils import calc_mean_variance, std_normalize


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(self,
                 data_file,
                 args,
                 mean=None,
                 var=None,

                 ):
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
        self.obs_len = args.obs_len
        self.pred_len = args.pred_len
        self.traj_len = args.obs_len + args.pred_len
        self.flip = args.flip
        self.pose_feats = 75
        self.num_keypoints = 25
        self.flow_feats = 24
        self.image_width = 1280
        self.image_height = 960

        self.xy_indexes = []
        for k in range(0, self.num_keypoints):
            self.xy_indexes.append(3 * k)
            self.xy_indexes.append(3 * k + 1)

        # read train/val from joblib file
        self.read_data(data_file)

        # calulate mean/var for pose features
        if(mean is None or var is None):

            # 1st method is to calculate mean/var using training dataset
            self.pose_mean, self.pose_var = calc_mean_variance(self.poses)                               # pose mean, var ~ [75]
            self.loc_mean, self.loc_var = calc_mean_variance(self.gt_locations)                          # location mean, var ~ [2]
            self.flow_mean, self.flow_var = calc_mean_variance(self.flow)                          # location mean, var ~ [2]

            # 2nd method is to calculate mean/var using image width/height
            self.pose_mean[0::2] = self.image_width / 2
            self.pose_mean[1::2] = self.image_height / 2
            self.pose_var[0::2] = self.image_width - self.image_width / 2
            self.pose_var[1::2] = self.image_height - self.image_height / 2

            self.mean, self.var = {}, {}
            self.mean['pose'], self.mean['loc'], self.mean['flow'] = self.pose_mean, self.loc_mean, self.flow_mean
            self.var['pose'], self.var['loc'], self.var['flow'] = self.pose_var, self.loc_var, self.flow_var
        else:
            self.pose_mean, self.loc_mean, self.flow_mean = mean['pose'], mean['loc'], mean['flow']
            self.pose_var, self.loc_var, self.flow_var = var['pose'], var['loc'], var['flow']

    def __len__(self):
        return self.num_samples

    def read_data(self, data_file):
        '''
            read train/val data
        '''

        data = joblib.load(data_file)

        poses, gt_locations, flow = [], [], []
        video_names, image_names, person_ids = [], [], []
        for sample in data:
            poses.append(sample['poses'])
            flow.append(sample['flow'])
            gt_locations.append(sample['gt_locations'])
            video_names.append(sample['video_names'][0])
            image_names.append(sample['image_names'][self.obs_len - 1])
            person_ids.append(sample['person_ids'][0])

        # convert to tensor
        poses = torch.tensor(poses, dtype=torch.float)                                          # ~ (num_samples, traj_len, keypoints*pose_feats)
        gt_locations = torch.tensor(gt_locations, dtype=torch.float)                            # ~ (num_samples, traj_len, 2)
        flow = torch.tensor(flow, dtype=torch.float)

        self.poses = poses[:, :, self.xy_indexes]
        self.flow = flow
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
            'flow': self.flow[index, :self.obs_len, :],
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
        sample['flow'] = std_normalize(sample['flow'], self.flow_mean, self.flow_var)

        return sample
