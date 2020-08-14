

## Goal 

 - In this project, we study how pose features impact on human future location prediction. We strongly focus handling the scenarios that 
 the human pose estimation are often imperfect (e.g. missing detected human keypoints). 
 - The current method is based on spatio-temporal graph convolutional neural networks (st-gcnn), which we will exploit its capability to "attend" to visible human keypoints, 
 give higher importance weights to human parts that are related to prediction tasks. 


## Download Datasets

JAAD dataset (http://data.nvision2.eecs.yorku.ca/JAAD_dataset/). 
Description 
    + The dataset contains 346 videos with 82032 frames, where the videos are the
    recorded with front-view camera under various scenes,
    weathers, and lighting conditions
 
    + Our dataset contains 88K frames with 2793 unique pedestrians labeled with over 390K bounding boxes. Occlusion tags are provided for each bounding box. ~57K (13%) of bounding boxes are tagged with partial occlusion and ~48K (12%) with heavy occlusion.

    1. Make sure JAAD dataset available in a directory (mine is `~/github/datasets/`. This can be done by
       running the following steps
       ```
       >> cd ~/github/datasets/
       >> git clone https://github.com/ykotseruba/JAAD.git 
       >> cd JAAD 
       >> sh download_clips.sh                 # download the JAAD clips 
       >> sh split_clips_to_frames.sh          # parse clips into images 
       ```

## Generate processed data
  
Create the processed features for JAAD dataset, and make sure the structure of raw data is at follow:

```
Traj-STGCNN
└── processed_data
    └── JAAD
        └── pose
            ├── video_0001
            │   └── 00000_keypoints.json
            │   └── ...
            │   └── 0000y_keypoints.json    
            │       
            ├── ...
            └── video_000y
        └── location
            ├── video_0001
            │   └── 00000_locations.json
            │   └── ...
            │   └── 0000x_locations.json
            │       
            ├── ...
            └── video_000y
```


To generate pose data, specify `JAAD_DATASET_DIR`, `PROCESSED_DATA_DIR`, `OPENPOSE_DIR`, then run:
```

$ python utils/generate_pose_data_jaad.py

```

To generate location data, specify `JAAD_DATASET_DIR` and `PROCESSED_DATA_DIR` in `generate_location_data_jaad.py`, then run:

```
$ python utils/generate_pose_data_jaad.py

```

To visualize processed features, specify `video_name`, `IMAGES_DIR`, `LOCATION_DATA_DIR`, `POSE_DATA_DIR`, `OUTPUT_IMAGES_DIR`, then run:
```
python utils/visualize_inputs.py
```

By default, this script plots both pose and bounding box on the same images. 


## Generate train/val data

  To generate train/val data, run the following: 
```

```

  Please double check and make sure the processed data folder is as follows: 

```
Traj-STGCNN
└── train_val_data
    └── JAAD
        └── mini_size
              └── train_data.joblib
              └── val_data.joblib
        └── medium_size
              └── train_data.joblib
              └── val_data.joblib
        └── full_size
              └── train_data.joblib
              └── val_data.joblib  
```



## Train 

Specify `args.train_data` and `args.val_data`, then run

```
python train.py --save_dir save/traj-stgcnn/full_size/
```

## Test

Specify `args.test_data` and `args.test_data`, then run

```
python test.py --python test.py --resume path/to/checkpoint

```



## Pose reconstruction 
```
 python train.py --reconstruct_pose --occl_ratio 0.2 --train_data train_val_data/JAAD/full_size/train_pose_reconstruction.joblib --val_data train_val_data/JAAD/full_size/val_pose_reconstruction.joblib --save_dir save/model4/reconstruct --nepochs 100
```


## Plan 
1. Pre-processes the datasets (done)
    + Extract sample from the dataset. Each sample includes these features
        -- video name 
        -- frame number
        -- pid 
        -- pose (using openpose) (x,y,c)
        -- optical_flow
        -- occludede ?

2. Build the network  (in-progress)
    + using st-gcn as decoder and encoder. 


3. Compare the results with other methods. 
    

4. Ablation study


## Comparisons
1. LSTM 
2. FPL (https://github.com/takumayagi/fpl)
2. Social-STGCNN (adapted to dynamic scenes).


## Analysis

``
python utils/analysis.py --traj_file save/trajs.json --test_data train_val_data/JAAD/mini_size/val_data.joblib
```







