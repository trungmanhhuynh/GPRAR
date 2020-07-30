import joblib
import torch
import numpy
from torch.utils.data import Dataset
from utils.utils import calc_mean_variance, std_normalize, std_denormalize


class TrajectoryDataset(Dataset):
	"""Dataloder for the Trajectory datasets"""
	def __init__(self, data_file, obs_len=10, pred_len=10, flip=False, reshape_pose = True, image_width=1280):
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

		self.data = joblib.load(data_file)   
		self.obs_len = obs_len
		self.pred_len = pred_len
		self.reshape_pose = reshape_pose
		traj_len = obs_len + pred_len
		pose_features = 3              # x, y, c
		keypoints = 25                 # using openpose 25 keypoints


		poses, imputed_poses, gt_locations, bboxes = [], [], [], []
		video_names, image_names, person_ids = [], [], [] 
		for sample in self.data: 
			poses.append(sample['poses']) 
			imputed_poses.append(sample['imputed_poses'])
			gt_locations.append(sample['locations'])
			bboxes.append(sample['bboxes'])
			video_names.append(sample['video_names'][0])
			image_names.append(sample['image_names'][obs_len-1])
			person_ids.append(sample['person_ids'][0])

		# convert to tensor
		poses = torch.tensor(poses, dtype=torch.float)                                          # ~ (num_samples, traj_len, keypoints*pose_features)
		imputed_poses =  torch.tensor(imputed_poses, dtype=torch.float)    						# ~ (num_samples, traj_len, keypoints*pose_features)
		gt_locations = torch.tensor(gt_locations, dtype=torch.float)                            # ~ (num_samples, traj_len, 2)
		bboxes = torch.tensor(bboxes, dtype=torch.float)                  						# ~ (num_samples, traj_len, 4)
		

		# calculate raw location and imputed location
		imputed_locations =  imputed_poses[:,:,24:26]    										# keypoint 8 ~ (num_samples, traj_len, 2)
		locations =  poses[:,:,24:26]    										# keypoint 8 ~ (num_samples, traj_len, 2)
																				# 


		# calculate mean/var
		self.loc_mean, self.loc_var = calc_mean_variance(gt_locations)                          # location mean, var ~ [2]
		self.pose_mean, self.pose_var = calc_mean_variance(poses)                               # pose mean, var ~ [75]


		# normalize data
		gt_locations = std_normalize(gt_locations, self.loc_mean, self.loc_var)          		   
		imputed_locations = std_normalize(imputed_locations, self.loc_mean, self.loc_var)        		
		locations = std_normalize(locations, self.loc_mean, self.loc_var)        																							
		poses = std_normalize(poses, self.pose_mean, self.pose_var)
		

		# convert poses to shape [samples, pose_features (3), traj_len (20), keypoints (25), instances (1)]
		num_samples = poses.shape[0]
		if(reshape_pose):
			poses = poses.view(num_samples, traj_len, keypoints, pose_features) 
			poses = poses.permute(0, 3, 1, 2).contiguous() 									# ~[num_samples, pose_features, traj_len, keypoints]
			poses = poses.unsqueeze(4)                    								    # ~[num_samples, pose_features, traj_len, keypoints, instances]


		self.poses = poses
		self.imputed_locations = imputed_locations
		self.locations = locations 
		self.gt_locations = gt_locations
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
		sample = {
			'locations': self.locations[index, : self.obs_len, :],
			'imputed_locations': self.imputed_locations[index, :self.obs_len, :],
			'gt_locations': self.gt_locations[index, -self.pred_len:, :],
			'video_names' : self.video_names[index], 
			'image_names' : self.image_names[index], 
			'person_ids' : self.person_ids[index]
		}


		if(self.reshape_pose):
			sample['poses'] = self.poses[index, :, :self.obs_len, :, :]
		else: 
			sample['poses'] = self.poses[index, :self.obs_len, :]


		
		return sample


