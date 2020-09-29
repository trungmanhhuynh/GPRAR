

## JAAD dataset (http://data.nvision2.eecs.yorku.ca/JAAD_dataset/). 
#### 1. Description 

  + The dataset contains 346 videos with 82032 frames, where the videos are the
    recorded with front-view camera under various scenes,
    weathers, and lighting conditions
 
  + Our dataset contains 88K frames with 2793 unique pedestrians labeled with over 390K bounding boxes. Occlusion tags are provided for each bounding box. ~57K (13%) of bounding boxes are tagged with partial occlusion and ~48K (12%) with heavy occlusion.

  Make sure JAAD dataset available in a directory (mine is `~/github/datasets/`. This can be done by
       running the following steps
       ```
       >> cd ~/github/datasets/
       >> git clone https://github.com/ykotseruba/JAAD.git 
       >> cd JAAD 
       >> sh download_clips.sh                 # download the JAAD clips 
       >> sh split_clips_to_frames.sh          # parse clips into images 
       ```

#### 2. Generate processed data
Create the processed features for JAAD dataset, and make sure the structure of data is at follow:  
```
Traj-STGCNN
└── processed_data
    └── JAAD
        └── pose
            ├── video_0001
            │   └── 00000_keypoints.json
            │   └── ...
            ├── ...
        └── location
            ├── video_0001
            │   └── 00000_locations.json
            │   └── ...     
            ├── ...
        └── grid_flow
            ├── video_0001
            │   └── 00000_gridflow.json
            │   └── ...     
            ├── ...
```

#### 2.1 Generate pose data
To generate pose data, specify `JAAD_DATASET_DIR`, `PROCESSED_DATA_DIR`, `OPENPOSE_DIR` in `generate_pose_data_jaad.py`, then run:
```
$ python utils/generate_pose_data_jaad.py
```
#### 2.2 Generate location data
To generate location data, specify `JAAD_DATASET_DIR` and `PROCESSED_DATA_DIR` in `generate_location_data_jaad.py`, then run:  
```
$ python utils/generate_location_data_jaad.py
```
#### 2.3 Generate grid optical flow data

```
$ conda activate flownet2_env
$ python data_utils/JAAD/generate_flow_data_jaad.py          # generate optical flow data for each image
$ python data_utils/JAAD/generate_gridflow_data_jaad.py
```
#### 2.4 Generate pedestrian id for pose 
```
$ python utils/assign_pose_id_jaad.py
```
## 3.Visualize

#### 3.1 Visualize Pose + Location + Bounding Box
To visualize processed features, specify `video_name`, `IMAGES_DIR`, `LOCATION_DATA_DIR`, `POSE_DATA_DIR`, `OUTPUT_IMAGES_DIR`, then run:
```
python utils/visualize_inputs.py
```
By default, this script plots both pose and bounding box on the same images. 

##### 3.2 Visualize Optical Flow 

Example below visualize optical flow to images of JAAD/video0001. 
```
$ conda activate flownet2_env
$ python data_utils/JAAD/visualize_flow_data.py --input_flow_dir processed_data/JAAD/flow/video_0001/ --output_image_dir video0001
```


#### Download Kinetics-skeleton dataset
```
gdown https://drive.google.com/uc?id=1PvIM_FjDRVSKh_kRDiWBpKkNNp3fUH3o
```

To generate train/val data with medium size (30% of data), run 
```
python data_utils/kinetics_skeleton/kinetics_gendata.py --data_path ../datasets/features/kinetics-skeleton/ --out_folder data/kinetics-skeleton/ --dsize medium
```

