
#### Social-STGCNN 
Download model 
```
$ cd .. 
$ git clone https://github.com/abduallahmohamed/Social-STGCNN.git
```

Generate train/val data 
JAAD dataset
```
$ python data_processing/prediction/generate_data_jaad_social-stgcnn.py --obs_type gt
$ python data_processing/prediction/generate_data_jaad_social-stgcnn.py --obs_type raw
$ python data_processing/prediction/generate_data_jaad_social-stgcnn.py --obs_type impute
```
TITAN


Modifications:  
In utils.py: add the below codes 
```
# add the next 2 lines for handling the size inconsistency
if curr_ped_seq.size != curr_seq[_idx, :, pad_front:pad_end].size:
   continue
```
In test.py: 
```
paths = ['./checkpoint/*jaad*'] (line 130)
data_set+'val/', (line 169)
```
Change function `ade()` and `fde()` to `ade_pixel()`
and `fde_pixel()`. Implement these functions in `metric.py` 


Train JAAD model:
```
$ python train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset jaad_gt --tag jaad_gt --use_lrschd --num_epochs 50 --obs_seq_len 10 --pred_seq_len 10
$ python train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset jaad_gt --tag jaad_impute --use_lrschd --num_epochs 50 --obs_seq_len 10 --pred_seq_len 10
$ python train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset jaad_gt --tag jaad_raw --use_lrschd --num_epochs 50 --obs_seq_len 10 --pred_seq_len 10
```
Train TITAN model:
```
$ python train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset titan_gt --tag titan_gt --use_lrschd --num_epochs 50 --obs_seq_len 10 --pred_seq_len 10
$ python train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset titan_impute --tag titan_impute --use_lrschd --num_epochs 50 --obs_seq_len 10 --pred_seq_len 10
$ python train.py --lr 0.01 --n_stgcnn 1 --n_txpcnn 5  --dataset titan_raw --tag titan_raw --use_lrschd --num_epochs 50 --obs_seq_len 10 --pred_seq_len 10
```

Test model: 
```
$ python test.py 
```

## Results
1. Results (loss) (50 epochs):  
Noisy Observation: ADE: 0.10726554634134752  FDE: 0.09081430786608759
Pre-processed Observation: ADE: 0.07415644297706861  FDE: 0.0679991089046919   
Ground-truth observation: ADE: 0.07381236445378819  FDE: 0.05483382429607957
2. Results (pixels) (50 epochs):  
3. Results (loss) (250 epochs):  
4. Results (pixels) (250 epochs):  
