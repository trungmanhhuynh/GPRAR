

## Goal 

 - In this project, we study how pose features impact on human future location prediction. We strongly focus handling the scenarios that 
 the human pose estimation are often imperfect (e.g. missing detected human keypoints). 
 - The current method is based on spatio-temporal graph convolutional neural networks (st-gcnn), which we will exploit its capability to "attend" to visible human keypoints, 
 give higher importance weights to human parts that are related to prediction tasks. 


## Generate processed data  
Please read [PRE_PROCESS.MD](PRE_PROCESS.MD)


## Reconstructor 
#### Generate train/val data 
We use high confident poses to train the reconstructor. The train/val data is generated
using the following command:
```
  python reconstructor/generate_data.py --d_size large --hc_poses 
```
The result files are `train_large_hcposes.joblib` and `val_large_hcposes.joblib` saved in `train_val_data/JAAD/reconstructor/`

#### Train
```
python reconstructor/train.py --train_data train_val_data/JAAD/reconstructor/train_large_hcposes.joblib --val_data train_val_data/JAAD/reconstructor/val_large_hcposes.joblib --add_noise
```
By default, trained models are save at `save/reconstructor/model`

#### Test
To valdiate on high-confident poses + random keypoint drop (same validation set as used in training)
```
python reconstructor/test.py --resume save/reconstructor/model/model_epoch_50.pt --val_data train_val_data/JAAD/reconstructor/val_large_hcposes.joblib --add_noise
```

To valdidate on the save
```
python reconstructor/test.py --resume save/reconstructor/model/model_epoch_50.pt --d_size large 
```

## Predictor
#### 1. Generate train/val data
```
python predictor/generate_data.py --d_size small
```

#### Train
```
python predictor/train.py --train_data 
```

#### Test


## Analysis

``
python utils/analysis.py --traj_file save/trajs.json --test_data train_val_data/JAAD/mini_size/val_data.joblib
```







