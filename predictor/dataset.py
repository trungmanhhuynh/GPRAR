import joblib
import torch
import numpy
import random
from torch.utils.data import Dataset
from utils.utils import calc_mean_variance, std_normalize, std_denormalize


class TrajectoryDataset(Dataset):
	"""Dataloder for the Trajectory datasets"""
	def __init__(self, data_file, obs_len=10, pred_len=10,
					flip=False, 
				    occl_ratio = 0, 
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
		print(self.poses.shape)

		# augment data
		if(flip):
			self.augment_flip_data()

		# make occlusion data if specified.
		if(occl_ratio > 0):
			self.create_occluded_pose_data(occl_ratio)			

		# normalize data 
		self.normalize_data() 

		# convert poses to shape [samples, pose_features (3), traj_len (20), keypoints (25), instances (1)]
		self.poses = self.poses.view(self.num_samples, self.traj_len, self.keypoints, self.pose_features) 
		self.poses = self.poses.permute(0, 3, 1, 2).contiguous() 									# ~[num_samples, pose_features, traj_len, keypoints]
		self.poses = self.poses.unsqueeze(4)                    								    # ~[num_samples, pose_features, traj_len, keypoints, instances]
		self.imputed_poses = self.imputed_poses.view(self.num_samples, self.traj_len, self.keypoints, self.pose_features) 
		self.imputed_poses = self.imputed_poses.permute(0, 3, 1, 2).contiguous() 									# ~[num_samples, pose_features, traj_len, keypoints]
		self.imputed_poses = self.imputed_poses.unsqueeze(4)                    								    # ~[num_samples, pose_features, traj_len, keypoints, instances]


	def __len__(self):
		return self.num_samples


	def read_data(self, data_file):
		''' read train/val data 
		'''

		data = joblib.load(data_file)   

		poses, imputed_poses, gt_locations, bboxes = [], [], [], []
		video_names, image_names, person_ids = [], [], [] 
		for sample in data: 
			poses.append(sample['poses']) 
			imputed_poses.append(sample['imputed_poses'])
			gt_locations.append(sample['locations'])
			bboxes.append(sample['bboxes'])
			video_names.append(sample['video_names'][0])
			image_names.append(sample['image_names'][self.obs_len-1])
			person_ids.append(sample['person_ids'][0])

		# convert to tensor
		poses = torch.tensor(poses, dtype=torch.float)                                          # ~ (num_samples, traj_len, keypoints*pose_features)
		imputed_poses =  torch.tensor(imputed_poses, dtype=torch.float)    						# ~ (num_samples, traj_len, keypoints*pose_features)
		gt_locations = torch.tensor(gt_locations, dtype=torch.float)                            # ~ (num_samples, traj_len, 2)
		bboxes = torch.tensor(bboxes, dtype=torch.float)                  						# ~ (num_samples, traj_len, 4)
		

		self.poses = poses
		self.imputed_poses = imputed_poses
		self.imputed_locations =  imputed_poses[:,:,24:26]    						# keypoint 8 ~ (num_samples, traj_len, 2)
		self.locations = poses[:,:,24:26]    										# keypoint 8 ~ (num_samples, traj_len, 2) 
		self.gt_locations = gt_locations
		self.video_names = video_names
		self.image_names = image_names
		self.person_ids = person_ids

		self.num_samples = poses.shape[0]


	def augment_flip_data(self):
		'''
			augment data by horizontal flip
		'''

		f_poses = self.poses.clone()
		f_poses[:,:,0::3]  = image_width - f_poses[:,:,0::3]
		f_imputed_poses = self.imputed_poses.clone()
		f_imputed_poses[:,:,0::3]  = image_width - f_imputed_poses[:,:,0::3]
		f_gt_locations = self.gt_locations.clone()
		f_gt_locations[:,:,0::3]  = image_width - f_gt_locations[:,:,0::3]
		f_bboxes = self.bboxes.clone()
		f_bboxes[:,:,0::3]  = image_width - f_bboxes[:,:,0::3]


		self.poses = torch.cat([poses, f_poses], dim=0)
		self.imputed_poses = torch.cat([imputed_poses, f_imputed_poses], dim=0)
		self.gt_locations = torch.cat([gt_locations, f_gt_locations], dim=0)
		self.bboxes = torch.cat([bboxes, f_bboxes], dim=0)


		self.video_names = self.video_names + self.video_names
		self.image_names = self.image_names + self.image_names
		self.person_ids = self.person_ids + self.person_ids


	def process_missing_part(self, missing_part):

		print("missing part is :", missing_part)
		if(missing_part!= None):
			if (missing_part  == "head"): missing_kpt = [0, 15, 16, 17, 18]
			elif (missing_part  == "body"): missing_kpt =  [1, 8, 9, 12]
			elif (missing_part  == "left_hand"): missing_kpt = [5, 6, 7]
			elif (missing_part  == "right_hand"): missing_kpt = [2, 3, 4]
			elif (missing_part  == "left_leg"): missing_kpt = [12, 14, 19, 20, 21]
			elif (missing_part  == "right_leg"): missing_kpt = [10, 11, 22, 23, 24]
			else: 	
				print("wrong missing part")

			for k in missing_kpt: 
				self.imputed_poses[:,:,k*3 : k*3 + 3] = 0


	def create_occluded_pose_data(self, occluded_rate):

		# set a random number of keypoints to zero vector using occluded_rate
		# this is for the study of seeing the impact of occlusion on prediction accuracy.

		random.seed(3)
		total_num_kpt = self.obs_len*self.keypoints 
		occ_t, occ_k = [], []

		occluded_kpt = random.choices(range(0,total_num_kpt), k = int(total_num_kpt*occluded_rate))
		print("number of occluded keypoints per target: {}/{}".format(len(occluded_kpt), total_num_kpt))
		for k in occluded_kpt:
			occ_t.append(int(k/self.keypoints ))
			occ_k.append(k%self.keypoints )

		for t in occ_t: 
			for k in occ_k:
				self.imputed_poses[:, t, k*3 : k*3 + 3] = 0
				self.poses[:, t, k*3 : k*3 + 3] = 0

	def create_occluded_location_data(self, occluded_rate):

		# set a random number of keypoints to zero vector using occluded_rate
		# this is for the study of seeing the impact of occlusion on prediction accuracy.
		random.seed(3)
		total_num_kpt = self.obs_len 
		occ_t, occ_k = [], []

		occluded_kpt = random.choices(range(0,total_num_kpt), k = int(total_num_kpt*occluded_rate))
		print("number of occluded locations per target: {}/{}".format(len(occluded_kpt), total_num_kpt))

		for t in occluded_kpt: 
			self.imputed_locations[:, t, :] = 0			# set keypoint 8 to 0

	def normalize_data(self):
		'''
			Calculate mean/var for each data feature and normalize
		'''

		# calculate mean/var
		self.loc_mean, self.loc_var = calc_mean_variance(self.gt_locations)                          # location mean, var ~ [2]
		self.pose_mean, self.pose_var = calc_mean_variance(self.poses)                               # pose mean, var ~ [75]
		self.imputed_pose_mean, self.imputed_pose_var = calc_mean_variance(self.imputed_poses)                               # pose mean, var ~ [75]

		# normalize data
		self.gt_locations = std_normalize(self.gt_locations, self.loc_mean, self.loc_var)          		   
		self.imputed_locations = std_normalize(self.imputed_locations, self.loc_mean, self.loc_var)        		
		self.locations = std_normalize(self.locations, self.loc_mean, self.loc_var)        																							
		self.poses = std_normalize(self.poses, self.pose_mean, self.pose_var)
		self.imputed_poses = std_normalize(self.imputed_poses, self.pose_mean, self.pose_var)



	def __getitem__(self, index):
		"""
			pose: size ~ [batch_size, pose_features, obs_len, keypoints, instances] or [N, C, T, V, M]
		"""
		sample = {
			'poses' : self.poses[index, :, :self.obs_len, :, :],
			'imputed_poses' :self.imputed_poses[index, :, :self.obs_len, :, :],
			'poses_gt': self.poses[index, :, -self.pred_len:, :, :],
			'imputed_poses_gt': self.imputed_poses[index, :,  -self.pred_len:, :, :],
			'locations': self.locations[index, : self.obs_len, :],
			'imputed_locations': self.imputed_locations[index, :self.obs_len, :],
			'gt_obs_locations': self.gt_locations[index, :self.obs_len, :],
			'gt_locations': self.gt_locations[index, -self.pred_len:, :],
			'video_names' : self.video_names[index], 
			'image_names' : self.image_names[index], 
			'person_ids' : self.person_ids[index]
		}

	
		return sample


