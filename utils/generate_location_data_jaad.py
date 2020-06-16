'''
generate_location_data_jaad.py 

Author: Manh Huynh
Last Update: 06/15/2020
'''


import sys
import os 


JAAD_DATASET_DIR = "/home/manhh/github/datasets/JAAD"
PROCESSED_DATA_DIR = "/home/manhh/github/Traj-STGCNN/processed_data/JAAD"
OPENPOSE_DIR =  "/home/manhh/github/openpose/build/examples/openpose"


sys.path.append(JAAD_DATASET_DIR)
from jaad_data import JAAD 


def generate_location_data():


	# Read ground truth pedestrian information given by JAAD dataset. 
	imdb = JAAD(data_path=JAAD_DATASET_DIR)
	jaad_data = imdb.generate_database()

	print(jaad_data)
	input("jere")


if __name__ == '__main__':


	generate_location_data() 