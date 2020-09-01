

## Goal 

 - In this project, we study how pose features impact on human future location prediction. We strongly focus handling the scenarios that 
 the human pose estimation are often imperfect (e.g. missing detected human keypoints). 
 - The current method is based on spatio-temporal graph convolutional neural networks (st-gcnn), which we will exploit its capability to "attend" to visible human keypoints, 
 give higher importance weights to human parts that are related to prediction tasks. 


## Generate processed data

Please read [PRE_PROCESS.MD](PRE_PROCESS.MD)


## Generate train/val data

#### Generate train/val data for reconstructor
If use high-confident poses and large data size 
```
  python generate_data_reconstructor.py --hc_poses --d_size large
```

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

1. Train Reconstructor 

```
python train.py --mode reconstructor 
```
By default, trained models are save at `save/reconstructor`

2. Train Predictor 

The Reconstructor's weights are freezed during this training phase. 
```
python train.py --mode predictor --resume save/reconstructor/model/model_epoch_50.py
```
By default, trained models are save at `save/predictor`






## Test

Specify `args.test_data` and `args.test_data`, then run

```
python test.py --python test.py --resume path/to/checkpoint

```



## Pose Reconstructor 
#### 1. Generate train/val data for reconstructor
```
python reconstructor/generate_train_val_data.py --data_size small
```

#### 2. Train pose reconstuctor 
```
python reconstructor/train.py --save_dir reconstructor/save/small
```


For modifying other parameters, please look into the script

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







