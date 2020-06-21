"""
Script to generate train and validation data 

author: Manh Huynh
last update: 06/21/2020

"""
import os
import json
import joblib
import random
import time
from itertools import islice


PROCESSED_LOCATION_DIR = "/home/manhh/github/Traj-STGCNN/processed_data/JAAD/location"
PROCESSED_POSE_ID_DIR = "/home/manhh/github/Traj-STGCNN/processed_data/JAAD/pose_id"
TRAIN_VAL_DIR = "/home/manhh/github/Traj-STGCNN/train_val_data/JAAD"

random.seed(1)


def chunks(data, traj_len = 20, slide = 1):
	#it = iter(data)
	for i in range(0, len(data), slide):
		yield list(islice(data, i, i + traj_len))



def generate_samples(video_data, video_name, traj_len = 20): 

	'''
		extract samples for each video. Each sample has the following dict structure: 
		
		{
			'video_names': [traj_len]
			'image_names': [traj_len]
			'person_ids': [traj_len]
			'poses': list ~[traj_len, 75]
			'locations': list ~[traj_len, 2]
			'bboxes':  list ~ [traj_len, 4]
		}

	'''
	video_samples = [] 

	# find list of pedestrian id 
	id_list = []
	for image_name in video_data:
		for person in video_data[image_name]["people"]:
			if(person['person_id'] not in id_list):
				id_list.append(person['person_id'])

	# print("person_id: ", id_list )
	
	# extract trajectory of each pedestrian
	for pid in id_list: 

		# extra whole trajector of a pedestrian 
		long_video_names = [] 
		long_image_names = [] 
		long_person_ids = [] 
		long_poses = [] 
		long_locations = [] 
		long_bboxes = [] 

		for image_name in video_data:
			for person in video_data[image_name]["people"]:
				if(person['person_id'] == pid):
					long_video_names.append(video_name)
					long_image_names.append(image_name)
					long_person_ids.append(pid)
					long_poses.append(person['pose'])
					long_locations.append(person['center'])
					long_bboxes.append(person['bbox'])


		# cut trajectories into chunk of pre-defined trajectory length 
		for video_names, image_names, person_ids, poses, locations, bboxes in \
									  zip(chunks(long_video_names, traj_len = traj_len, slide = 1), 
										  chunks(long_image_names, traj_len = traj_len, slide = 1),
										  chunks(long_person_ids, traj_len = traj_len, slide = 1),
										  chunks(long_poses, traj_len = traj_len, slide = 1), 
										  chunks(long_locations, traj_len = traj_len, slide = 1), 
										  chunks(long_bboxes, traj_len = traj_len, slide = 1)):  

			# skip if extracted trajectory is shorted thatn pre-defined one 
			if(len(locations) < traj_len): continue

			#---check if trajectory is continous by extracting frame number [-11:-4] from names
			# this applies for image names with format: 0000x.png  or {:05d}.png
			gap = abs(int(image_names[0][-9:-4]) - int(image_names[-1][-9:-4]))
			if(gap > traj_len): continue

			# add to sample list
			video_samples.append({
					'video_names': video_names, 
					'image_names': image_names, 
					'person_ids': person_ids, 
					'poses': poses, 
					'locations': locations,
					'bboxes': bboxes
				})

			# print(video_samples)
			# input("here")

	return video_samples


def generate_train_val_data(data_size):


	#---generate samples (list of features) for each pedestrian
	video_dir = os.path.join(PROCESSED_POSE_ID_DIR) 

	if(data_size == "mini_size"):

		num_videos = random.sample(os.listdir(video_dir), k = 30) 		
	else: 
		num_videos = os.listdir(video_dir)    						    # use all videos


	total_samples = [] 
	for video_name in num_videos: 

		'''
			generate data for a video with this dictionary structure
			'image_name'(str):{
				'people': [
					{
					'person_id': (str)
					'bbox':  [x1, y1, x2, y2]
					'center': [xc, yc]
					'pose'  : [1x75]  // [x,y,c] x 3
					}
				]
			}	

		'''

		print("processing video: ", video_name)
		video_data = {}

		for frame_number in range(len(os.listdir(os.path.join(PROCESSED_POSE_ID_DIR, video_name)))):

			with open(os.path.join(PROCESSED_POSE_ID_DIR, video_name, "{:05d}_keypoints.json".format(frame_number)) , "r") as f:
				pose_data = json.load(f) 

			with open(os.path.join(PROCESSED_LOCATION_DIR, video_name, "{:05d}_locations.json".format(frame_number)) , "r") as f:
				location_data = json.load(f) 


			image_name = "{:05d}.png".format(frame_number)
			video_data[image_name] = {}
			video_data[image_name]['people'] = []


			for ped in pose_data['people']:

				if(ped['person_id'][0] == -1):
					continue 

				temp = {}
				temp['person_id']  = ped['person_id']
				temp['pose'] = ped['pose_keypoints_2d']

				for p_in_loc in location_data['people']:
					if(ped['person_id'] == p_in_loc['person_id']):

						temp['bbox'] = p_in_loc['bbox']
						temp['center'] = p_in_loc['center']

				video_data[image_name]['people'].append(temp)

		# extract samples in a video
		video_samples = generate_samples(video_data, video_name)

		total_samples.append(video_samples)

	total_samples = sum(total_samples, []) 

	# split train/val data with ratio 80%/20%
	val_sample_indexes = random.sample(range(len(total_samples)), k = int(len(total_samples)*0.2))
	train_sample_indexes = list(set(range(len(total_samples))) - set(val_sample_indexes))

	val_samples = [total_samples[i] for i in val_sample_indexes]
	train_samples = [total_samples[i] for i in train_sample_indexes]

	print("number of video used: {}".format(len(num_videos)))
	print("total number of samples: {}".format(len(total_samples)))
	print("train/val samples = {}/{}".format(len(train_samples), len(val_samples)))

	return train_samples, val_samples


if __name__ == "__main__":


	stop_timer = time.time()
	data_size = "full_size"
	train_samples, val_samples = generate_train_val_data(data_size)


	#---dump to file 
	if not os.path.exists(os.path.join(TRAIN_VAL_DIR, data_size)):
		os.makedirs(os.path.join(TRAIN_VAL_DIR, data_size))

	train_data_file = os.path.join(TRAIN_VAL_DIR, data_size, "train_data.joblib")
	val_data_file = os.path.join(TRAIN_VAL_DIR, data_size, "val_data.joblib")

	joblib.dump(train_samples, train_data_file)
	joblib.dump(val_samples, val_data_file)
	start_timer = time.time()



	print("Dumped train data to file :", train_data_file)
	print("Dumped val data to file :", val_data_file)
	print("Processing time : {:.2f} (s)".format((start_timer - stop_timer)))