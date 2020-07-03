import joblib
import torch
import numpy
from torch.utils.data import Dataset



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



        # convert to tensor
        poses, locations, bboxes = [], [], []
        video_names = [] 
        for sample in self.data: 
            poses.append(sample['poses']) 
            locations.append(sample['locations'])
            bboxes.append(sample['bboxes'])

        poses_tensor = torch.tensor(poses, dtype=torch.float)               # ~ [num_samples, traj_len, keypoints*pose_features]
        locations_tensor = torch.tensor(locations, dtype=torch.float)       # ~ [num_samples, traj_len, 2]
        bboxes_tensor = torch.tensor(bboxes, dtype=torch.float)             # ~ [num_samples, traj_len, 4]

        # convert poses_tensor to shape [samples, pose_features (3), traj_len (20), keypoints (25), instances (1)]
        num_samples = poses_tensor.shape[0]
        poses_tensor = poses_tensor.view(num_samples, traj_len, keypoints, pose_features) 
        poses_tensor = poses_tensor.permute(0, 3, 1, 2).contiguous() # ~[num_samples, pose_features, traj_len, keypoints]
        poses_tensor = poses_tensor.unsqueeze(4)                     # ~[num_samples, pose_features, traj_len, keypoints, instances]


        self.pose = poses_tensor
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        """
            pose: size ~ [batch_size, pose_features, obs_len, keypoints, instances] or [N, C, T, V, M]
        """

        sample = [
            self.pose[index, :, :self.obs_len, :, :]
        ]
        return sample
