import joblib
import torch
import numpy
from torch.utils.data import Dataset
from utils.utils import calc_mean_variance, std_normalize


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(self, data_file, obs_len=10, pred_len=10, flip=False, image_width=1280):
        """
        Args:
            data_file: file name of train/val data. Data in data_file has the following structure:
            [{
                'video_names': [traj_len]
                'image_names': [traj_len]
                'person_ids': [traj_len]
                'poses': list ~[traj_len, 75]
                'locations': list ~[traj_len, 2]
                'bboxes':  list ~ [traj_len, 4]
             }]

        """
        super(TrajectoryDataset, self).__init__()

        self.data = joblib.load(data_file)   
        self.obs_len = obs_len
        self.pred_len = pred_len
        traj_len = obs_len + pred_len
        pose_features = 3              # x, y, c
        keypoints = 25                 # using openpose 25 keypoints


        poses, locations, bboxes = [], [], []
        video_names, image_names, person_ids = [], [], [] 
        for sample in self.data: 
            poses.append(sample['poses']) 
            locations.append(sample['locations'])
            bboxes.append(sample['bboxes'])
            video_names.append(sample['video_names'][0])
            image_names.append(sample['image_names'][obs_len-1])
            person_ids.append(sample['person_ids'][0])

        # convert to tensor
        poses_tensor = torch.tensor(poses, dtype=torch.float)               # ~ [num_samples, traj_len, keypoints*pose_features]
        locations_tensor = torch.tensor(locations, dtype=torch.float)       # ~ [num_samples, traj_len, 2]
        bboxes_tensor = torch.tensor(bboxes, dtype=torch.float)             # ~ [num_samples, traj_len, 4]


        # normalize data 
        self.loc_mean, self.loc_var = calc_mean_variance(locations_tensor)     # location mean, var ~ [2]
        self.pose_mean, self.pose_var = calc_mean_variance(poses_tensor)       # pose mean, var ~ [75]
        self.pose_center_mean, self.pose_center_var = self.pose_mean[3:5], self.pose_var[3:5]      # pose center has index 1,
                                                                                                       # thus x,y is at indexes 3,4
                                                                                                       # ~ [2]

        locations_tensor = std_normalize(locations_tensor, self.pose_center_mean, self.pose_center_var)
        poses_tensor= std_normalize(poses_tensor, self.pose_mean, self.pose_var)
        

        # convert poses_tensor to shape [samples, pose_features (3), traj_len (20), keypoints (25), instances (1)]
        num_samples = poses_tensor.shape[0]
        poses_tensor = poses_tensor.view(num_samples, traj_len, keypoints, pose_features) 
        poses_tensor = poses_tensor.permute(0, 3, 1, 2).contiguous() # ~[num_samples, pose_features, traj_len, keypoints]
        poses_tensor = poses_tensor.unsqueeze(4)                     # ~[num_samples, pose_features, traj_len, keypoints, instances]


        self.poses = poses_tensor
        self.locations = locations_tensor
        self.num_samples = num_samples
        self.video_names = video_names
        self.image_names = image_names
        self.person_ids = person_ids

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        """
            pose: size ~ [batch_size, pose_features, obs_len, keypoints, instances] or [N, C, T, V, M]
        """

        sample = [
            self.poses[index, :, :self.obs_len, :, :],
            self.locations[index, -self.pred_len:, :],
            self.video_names[index], 
            self.image_names[index], 
            self.person_ids[index]
        ]

        return sample


