'''
generate_pose_data_jaad.py 

Author: Manh Huynh
Last Update: 06/15/2020
'''

import os


JAAD_DATASET_DIR = "/home/manhh/github/datasets/JAAD"
PROCESSED_DATA_DIR = "/home/manhh/github/Traj-STGCNN/processed_data/JAAD"
OPENPOSE_DIR =  "/home/manhh/github/openpose/build/examples/openpose"


def generate_pose_data_jaad():
	'''
		generate pose feature for jaad datasets using openpose. The script run the following command line
		for each video in the dataset. 
		#./build/examples/openpose/openpose.bin --image_dir /home/manhh/github/datasets/JAAD/images \
		--write_json processed_pose_dir--display 0 --render_pose 0 

		Read README.md for output directory structure of the processed data


		Look https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
		for dictionary structure of each frame  data 		


	'''

	images_dir = os.path.join(JAAD_DATASET_DIR, "images") 
	for video_name in os.listdir(images_dir): 

		print("Generate pose data for : ", video_name)

		processed_pose_dir = os.path.join(PROCESSED_DATA_DIR, "pose", video_name)
		if not os.path.exists(processed_pose_dir):
			os.makedirs(processed_pose_dir)

		cmd = "cd /home/manhh/github/openpose && {0} --image_dir {1} --write_json {2} --display 0 --render_pose 0".format(
			os.path.join(OPENPOSE_DIR, "openpose.bin"), 
			os.path.join(images_dir, video_name), 
			processed_pose_dir) 

		os.system(cmd)


if __name__ == '__main__':


	generate_pose_data_jaad() 