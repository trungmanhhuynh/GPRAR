"""
#
# assign_pose_id_jaad.py: By default, pose features resulted from openpose does not 
# have pedestrian id, which requires for prediction task. This script assign ped id for 
# each pose in each frame using ground-truth pedestrian locations. The Hungarian matching 
# algorithm is used for the assignement. 
# Author: Manh Huynh
# Last Update: 06/18/2020

"""
import os 
import json
from munkres import Munkres, print_matrix
from scipy.spatial import distance
import numpy as np


PROCESSED_DATA_DIR = "/home/manhh/github/Traj-STGCNN/processed_data/JAAD"
PROCESSED_POSE_DIR = "/home/manhh/github/Traj-STGCNN/processed_data/JAAD/pose"
PROCESSED_LOCATION_DIR = "/home/manhh/github/Traj-STGCNN/processed_data/JAAD/location"
OUTPUT_POSE_ID_DIR = "/home/manhh/github/Traj-STGCNN/processed_data/JAAD/pose_id"

def generate_cost_matrix(pose_data, location_data):


	# Extract center position of all pedestrian within a frame
	# from pose data and location data
	pose_center = [] 
	for ped in pose_data["people"]: 

		pose_center_x = ped['pose_keypoints_2d'][24]   # mid hip - index 8 in keypoints
		pose_center_y = ped['pose_keypoints_2d'][25]       
		pose_center.append([pose_center_x, pose_center_y])


	location_center = [] 
	for ped_id in location_data['ped_annotations']:

		loc_center_x  = 0.5*(location_data['ped_annotations'][ped_id]['bbox'][0] \
						 + location_data['ped_annotations'][ped_id]['bbox'][2])

		loc_center_y = 0.5*(location_data['ped_annotations'][ped_id]['bbox'][1] \
						 + location_data['ped_annotations'][ped_id]['bbox'][3])

		location_center.append([loc_center_x, loc_center_y])


	# Build cost matrix as a numpy array ~ (num_pose, num_gt_ped)
	num_pose = len(pose_center)
	num_gt_ped = len(location_center)
	cost_matrix = np.zeros((num_pose, num_gt_ped))           		# could be rectangular matrix
																	# pose is on row
																	# gt people is on col
	for i in range(cost_matrix.shape[0]):
		for j in range(cost_matrix.shape[1]):
			cost_matrix[i][j] =  distance.euclidean(pose_center[i], location_center[j])

	return cost_matrix.tolist()

def assign_pose_id():

	"""
		function to assign 'person_id' for each pose 
	"""

	video_dir = os.path.join(PROCESSED_POSE_DIR) 
	for video_name in os.listdir(video_dir): 

		print("processing video :", video_name)

		if not os.path.exists(os.path.join(OUTPUT_POSE_ID_DIR, video_name)):
			os.makedirs(os.path.join(OUTPUT_POSE_ID_DIR, video_name))


		# Assign person_id for each pose in a frame
		for frame_number in range(len(os.listdir(os.path.join(PROCESSED_POSE_DIR, video_name)))):

			with open(os.path.join(PROCESSED_POSE_DIR, video_name, "{:05d}_keypoints.json".format(frame_number)) , "r") as f:
				pose_data = json.load(f) 

			with open(os.path.join(PROCESSED_LOCATION_DIR, video_name, "{:05d}_locations.json".format(frame_number)) , "r") as f:
				location_data = json.load(f) 


			if not pose_data["people"]: 
				continue 

			cost_matrix = generate_cost_matrix(pose_data, location_data) 
			#print("num_pose={}, num_gt_peds ={}".format(len(cost_matrix), len(cost_matrix[0])))
			#print_matrix(cost_matrix, msg='Lowest cost through this matrix:')

			# run matching 
			m = Munkres()
			indexes = m.compute(cost_matrix)             #(pose_idx, actual_person_id)


			for row, col in indexes:
				# ped_id of pose at row matched with location at col
				pose_data["people"][row]['person_id'] = list(location_data['ped_annotations'])[col]

			# save to file
			outfile = os.path.join(OUTPUT_POSE_ID_DIR, video_name, "{:05d}_keypoints.json".format(frame_number))
			with open(outfile, 'w') as f:
				json.dump(pose_data, f)

	print("processing done")


def test_munkres():
	"""
		Testing munkres algorithm 
	"""

	# test square matrix 
	print("Test square matrix")
	matrix = [[5, 9, 1],
		  	  [10, 3, 2],
			  [8, 7, 4]]
	m = Munkres()
	indexes = m.compute(matrix)
	print_matrix(matrix, msg='Lowest cost through this matrix:')
	total = 0
	for row, column in indexes:
		value = matrix[row][column]
		total += value
		print(f'({row}, {column}) -> {value}')
	print(f'total cost: {total}')

	print("Test non-square matrix 1")
	matrix = [[5,  1],
		  	  [10,  2],
			  [8,  4]]
	m = Munkres()
	indexes = m.compute(matrix)
	print_matrix(matrix, msg='Lowest cost through this matrix:')
	total = 0
	for row, column in indexes:
		value = matrix[row][column]
		total += value
		print(f'({row}, {column}) -> {value}')
	print(f'total cost: {total}')


	print("Test non-square matrix 2")
	matrix = [[5, 10, 1],
		  	  [1, 4, 2]]
	m = Munkres()
	indexes = m.compute(matrix)
	print_matrix(matrix, msg='Lowest cost through this matrix:')
	total = 0
	for row, column in indexes:
		value = matrix[row][column]
		total += value
		print(f'({row}, {column}) -> {value}')
	print(f'total cost: {total}')




if __name__ == "__main__":


	# test munkres 
	#test_munkres()

	assign_pose_id()