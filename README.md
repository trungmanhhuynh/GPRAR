

## Goal 

 - In this project, we study how pose features impact on human future location prediction. We strongly focus handling the scenarios that 
 the human pose estimation are often imperfect (e.g. missing detected human keypoints). 
 - The current method is based on spatio-temporal graph convolutional neural networks (st-gcnn), which we will exploit its capability to "attend" to visible human keypoints, 
 give higher importance weights to human parts that are related to prediction tasks. 


## Generate processed data  
Please read [PRE_PROCESS.md](PRE_PROCESS.md)


## Reconstructor 
#### 1. Generate train/val data 
Generate confident poses:
```
$ python reconstructor/generate_data.py --d_size large
```
By default, The script generates `train_val_data/JAAD/reconstructor/train_$d_size.joblib` and `train_val_data/JAAD/reconstructor/val_$d_size.joblib`

#### 2. Train
```
python reconstructor/train.py --dset JAAD --dsize large --add_noise --save_dir save/reconstructor/
```

#### 3. Test

To valdiate on high-confident poses + random keypoint drop (same validation set as used in training)
```
python reconstructor/test.py --test_data train_val_data/JAAD/reconstructor/val_large.joblib --add_noise --resume save/reconstructor/model/model_best.py
```
To test on pose used in prediction step. Dont specify `--add_noise` because it already exists
```
python reconstructor/test.py --test_data train_val_data/JAAD/predictor/val_large.joblib --resume save/reconstructor/model/model_best.py
```

## Predictor
#### Generate train/val data
```
python predictor/generate_data.py --d_size small
```
#### Train without using reconstructor
```
$ python predictor/train.py --train_data train_val_data/JAAD/predictor/train_medium.joblib --val_data train_val_data/JAAD/predictor/val_medium.joblib --save_dir save_temp/predictor_medium_withflow_norc
```

#### Train using reconstructor
```
python predictor/train.py --train_data train_val_data/JAAD/predictor/train_medium.joblib --val_data train_val_data/JAAD/predictor/val_medium.joblib --save_dir save_temp/predictor_medium_withflow_withrc --resume save/reconstructor/model/model_best.pt
```
#### Test

## Analysis

```
python utils/analysis.py --traj_file save/trajs.json --test_data train_val_data/JAAD/mini_size/val_data.joblib
```




