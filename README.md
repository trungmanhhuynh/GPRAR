

## Goal 

Use ST-GCN to generate a better pose features, which helps improve trajectory prediction accuracy. 

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

## Generate raw data
  
  Create the raw data for JAAD dataset, and make sure the structure of raw data is at follow:

```
Traj-STGCNN
└── raw_data
    └── JAAD
        └── pose_data
            ├── video1
            │   └── frame01
            │   └── ...
            │   └── framen    
            │       
            ├── ...
            └── videon
        └── location_data
            ├── video1
            │   └── frame01
            │   └── ...
            │   └── framen    
            │       
            ├── ...
            └── videon
        └── opticalflow_data
            ├── video1
            │   └── frame01
            │   └── ...
            │   └── framen    
            │       
            ├── ...
            └── videon

```


## Generate processed data 

  To generate processed data, run the following: 


  Please double check and make sure the processed data folder is as follows: 

```
Traj-STGCNN
└── processed_data
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



2. 


## Plan 
1. Pre-processes the datasets 
    + Extract sample from the dataset. Each sample includes these features
        -- video name 
        -- frame number
        -- pid 
        -- pose (using openpose) (x,y,c)
        -- optical_flow
        -- occludede ?

2. Build the network  
    + using tcn as decoder and encoder. 


3. Compare the results with other methods. 
    

4. Ablation sutdy


## Comparisons
1. LSTM 
2. FPL 
2. Social-STGCNN (adapted to dynamic scenes).

    







