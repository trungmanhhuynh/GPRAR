'''

Author: Manh Huynh
Last Update: 06/03/2020
'''

import sys
import os

#from datasets.JAAD.jaad_data import JAAD
#sys.path.append('..')


JAAD_DATASET_DIR = "/home/manhh/github/datasets/JAAD"
PROCESSED_DATA_DIR = "/home/manhh/github/Traj-STGCNN/processed_data/JAAD"
OPENPOSE_DIR =  "/home/manhh/github/openpose/build/examples/openpose"


def generate_pose_data():

	#./build/examples/openpose/openpose.bin --image_dir ~/github/datasets/eth_mobile/images/eth_bahnhof/--write_json ./mytest/ --display 0 --render_pose 0 


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


def generate_location_data():


	 # 1. Read ground truth pedestrian information given by JAAD dataset. 
	imdb = JAAD(data_path=JAAD_DATASET_PATH)
	jaad_data = imdb.generate_database()

	print(jaad_data)
	input("jere")




if __name__ == '__main__':



	generate_pose_data() 