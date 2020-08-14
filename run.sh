# echo "Running LSTM model" 
# python sota/lstm/test.py --save_dir sota/lstm/save/ --resume sota/lstm/save/model/model_epoch_49.pt --occluded_rate 0.0 --occluded_type locations  
# python sota/lstm/test.py --save_dir sota/lstm/save/ --resume sota/lstm/save/model/model_epoch_49.pt --occluded_rate 0.1 --occluded_type locations
# python sota/lstm/test.py --save_dir sota/lstm/save/ --resume sota/lstm/save/model/model_epoch_49.pt --occluded_rate 0.2 --occluded_type locations
# python sota/lstm/test.py --save_dir sota/lstm/save/ --resume sota/lstm/save/model/model_epoch_49.pt --occluded_rate 0.3 --occluded_type locations
# python sota/lstm/test.py --save_dir sota/lstm/save/ --resume sota/lstm/save/model/model_epoch_49.pt --occluded_rate 0.4 --occluded_type locations
# python sota/lstm/test.py --save_dir sota/lstm/save/ --resume sota/lstm/save/model/model_epoch_49.pt --occluded_rate 0.5 --occluded_type locations

echo "Running TCNN_POSE model"
python sota/tcnn/test_pytorch.py --save_dir sota/tcnn/save/tcnn_pose --model tcnn_pose --resume sota/tcnn/save/tcnn_pose/model/model_epoch_49.pt --occluded_rate 0.0 --occluded_type pose
python sota/tcnn/test_pytorch.py --save_dir sota/tcnn/save/tcnn_pose --model tcnn_pose --resume sota/tcnn/save/tcnn_pose/model/model_epoch_49.pt --occluded_rate 0.01 --occluded_type pose
python sota/tcnn/test_pytorch.py --save_dir sota/tcnn/save/tcnn_pose --model tcnn_pose --resume sota/tcnn/save/tcnn_pose/model/model_epoch_49.pt --occluded_rate 0.02 --occluded_type pose
python sota/tcnn/test_pytorch.py --save_dir sota/tcnn/save/tcnn_pose --model tcnn_pose --resume sota/tcnn/save/tcnn_pose/model/model_epoch_49.pt --occluded_rate 0.03 --occluded_type pose
python sota/tcnn/test_pytorch.py --save_dir sota/tcnn/save/tcnn_pose --model tcnn_pose --resume sota/tcnn/save/tcnn_pose/model/model_epoch_49.pt --occluded_rate 0.04 --occluded_type pose
python sota/tcnn/test_pytorch.py --save_dir sota/tcnn/save/tcnn_pose --model tcnn_pose --resume sota/tcnn/save/tcnn_pose/model/model_epoch_49.pt --occluded_rate 0.05 --occluded_type pose
python sota/tcnn/test_pytorch.py --save_dir sota/tcnn/save/tcnn_pose --model tcnn_pose --resume sota/tcnn/save/tcnn_pose/model/model_epoch_49.pt --occluded_rate 0.06 --occluded_type pose
python sota/tcnn/test_pytorch.py --save_dir sota/tcnn/save/tcnn_pose --model tcnn_pose --resume sota/tcnn/save/tcnn_pose/model/model_epoch_49.pt --occluded_rate 0.07 --occluded_type pose
python sota/tcnn/test_pytorch.py --save_dir sota/tcnn/save/tcnn_pose --model tcnn_pose --resume sota/tcnn/save/tcnn_pose/model/model_epoch_49.pt --occluded_rate 0.08 --occluded_type pose
python sota/tcnn/test_pytorch.py --save_dir sota/tcnn/save/tcnn_pose --model tcnn_pose --resume sota/tcnn/save/tcnn_pose/model/model_epoch_49.pt --occluded_rate 0.09 --occluded_type pose
python sota/tcnn/test_pytorch.py --save_dir sota/tcnn/save/tcnn_pose --model tcnn_pose --resume sota/tcnn/save/tcnn_pose/model/model_epoch_49.pt --occluded_rate 0.1 --occluded_type pose
python sota/tcnn/test_pytorch.py --save_dir sota/tcnn/save/tcnn_pose --model tcnn_pose --resume sota/tcnn/save/tcnn_pose/model/model_epoch_49.pt --occluded_rate 0.2 --occluded_type pose
python sota/tcnn/test_pytorch.py --save_dir sota/tcnn/save/tcnn_pose --model tcnn_pose --resume sota/tcnn/save/tcnn_pose/model/model_epoch_49.pt --occluded_rate 0.3 --occluded_type pose
python sota/tcnn/test_pytorch.py --save_dir sota/tcnn/save/tcnn_pose --model tcnn_pose --resume sota/tcnn/save/tcnn_pose/model/model_epoch_49.pt --occluded_rate 0.4 --occluded_type pose
python sota/tcnn/test_pytorch.py --save_dir sota/tcnn/save/tcnn_pose --model tcnn_pose --resume sota/tcnn/save/tcnn_pose/model/model_epoch_49.pt --occluded_rate 0.5 --occluded_type pose

