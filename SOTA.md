


## TCNN model
**Train**  

```
python sota/tcnn/train_pytorch.py --nepochs 50 --pred_len 10 --model tcnn --save_dir sota/tcnn/save/tcnn/
```

by default, the trained network is saved at sota/tcnn/save/model

**Test**
```
python sota/tcnn/test_pytorch.py --model tcnn --resume sota/tcnn/save/tcnn/model/model_epoch_49.pt --save_dir sota/tcnn/save/tcnn/
```


## TCNN_POSE model
**Train**  

```
python sota/tcnn/train_pytorch.py --nepochs 50 --pred_len 10 --model tcnn --save_dir sota/tcnn/save/tcnn/ 
```

by default, the trained network is saved at sota/tcnn/save/model

**Test**
```
python sota/tcnn/test_pytorch.py --model tcnn_pose --resume sota/tcnn/save/tcnn_pose/model/model_epoch_49.pt --save_dir sota/tcnn/save/tcnn_pose/
```

## LSTM model
**Train**
```
 python sota/lstm/train.py --nepochs 50 --pred_len 10 --save_dir sota/lstm/save/lstm
```
**Test**
```
python sota/lstm/test.py --resume sota/lstm/save/lstm/model/model_epoch_49.pt --save_dir sota/lstm/save/lstm/

```

