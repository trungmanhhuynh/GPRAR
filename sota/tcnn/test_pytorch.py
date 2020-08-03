"""
Train module - TCNN
- Re-use the simmilar model (using chainer) given by the author

Author: Manh Huynh
Last Update: 06/23/2020

"""
import os
import time
import json
import argparse
import torch
import sys
sys.path.append("/home/manhh/github/Traj-STGCNN")


import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import TrajectoryDataset
from utils.utils import calculate_ade_fde
from sota.tcnn.model_pytorch import TCNN, TCNN_POSE
from utils.utils import calculate_ade_fde, save_traj_json


parser = argparse.ArgumentParser()
parser.add_argument('--obs_len', type=int, default=10)
parser.add_argument('--pred_len', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128, 
                    help='minibatch size')
parser.add_argument('--test_data', type=str,  default="train_val_data/JAAD/mini_size/val_data.joblib", 
                    help='file used for testing')
parser.add_argument('--use_cuda', action='store_true', default= True, 
                    help = 'use gpu')
parser.add_argument('--save_dir', type=str, default='sota/tcnn/save/tcnn',
                     help='save directory')
parser.add_argument('--resume', type=str, default="",
                     help='resume a trained model?')
parser.add_argument('--model', type=str, default="tcnn_pose",
                     help='supporting tcnn or tcnn_pose')

args = parser.parse_args()

args.save_model_dir = os.path.join(args.save_dir, "model")
args.save_log_dir = os.path.join(args.save_dir, "log")
if not os.path.exists(args.save_model_dir ):
    os.makedirs(args.save_model_dir )
if not os.path.exists(args.save_log_dir ):
    os.makedirs(args.save_log_dir )
print(args)


# Fixed the seed
np.random.seed(1)
torch.manual_seed(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 1.prepare data
dset_test = TrajectoryDataset(
        args.test_data,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        flip = False,
        reshape_pose = False
        )

loader_test = DataLoader(
        dset_test,
        batch_size=args.batch_size, 
        shuffle =False,
        num_workers=0)

print("test datasize = {}".format(len(dset_test)))


# 2.load model
if(args.model == 'tcnn_pose'):
    model = TCNN_POSE(pred_len = args.pred_len)
if(args.model == 'tcnn'):
    model = TCNN(pred_len = args.pred_len)
    
if(args.use_cuda) : model = model.cuda() 
#print(model)
#model.apply(weights_init)


# 3. define loss function 
mse_loss = torch.nn.MSELoss()  


# 5. load check points ?
resume_epoch = 0 
if(args.resume != ""):
    resume_dict = torch.load(args.resume)
    model.load_state_dict(resume_dict['state_dict'])
    resume_epoch = resume_dict['e']


# 6.2 validate
print("---validate---")
start_time = time.time()

test_loss = 0 ;  test_ade = 0 ; test_fde = 0 
traj_dict = {'video_names': [], 'image_names': [],  'person_ids': [], 'traj_gt': [], 'traj_pred': [], 'pose': []}
model.eval()
for test_it, samples in enumerate(loader_test):
    
    locations = Variable(samples['locations'])              # pose ~ [batch_size, pose_features, obs_len, keypoints, instances]   
    poses = Variable(samples['poses'])                        # pose ~ [batch_size, pose_features, obs_len, keypoints, instances]                                                   
    gt_locations =  Variable(samples['gt_locations'])               # gt_locations ~ [batch_size, pred_len, 2]

    if(args.use_cuda): 
        poses, gt_locations= poses.cuda(), gt_locations.cuda()
        locations = locations.cuda()

    #forward
    if(args.model == 'tcnn'):
        pred_locations = model(locations)                                       # pred_locations ~ [batch_size, pred_len, 2]

    elif(args.model == 'tcnn_pose'):
        pred_locations = model(locations, poses)                                # pred_locations ~ [batch_size, pred_len, 2]
                                                                                # pred_locations ~ [batch_size, pred_len, 2]
    test_loss +=  mse_loss(pred_locations, gt_locations).item()


    # calculate ade/fde
    ade, fde = calculate_ade_fde(gt_locations, pred_locations, dset_test.loc_mean, dset_test.loc_var)
    test_ade += ade 
    test_fde += fde

    # get trajectories
    traj_dict = save_traj_json(traj_dict, pred_locations, samples['video_names'], samples['image_names'], samples['person_ids'], \
                               dset_test.loc_mean, dset_test.loc_var)




test_loss /= len(loader_test)
test_ade /=  len(loader_test)
test_fde /= len(loader_test)
stop_time = time.time()


print("epoch:{} test_loss:{:.5f} test_ade:{:.2f} test_fde:{:.2f} time(ms):{:.2f}".format(
     resume_epoch, test_loss, test_ade, test_fde, (stop_time - start_time)*1000))


# save trajectories to file
for key in traj_dict:
    traj_dict[key] = sum(traj_dict[key], [] )           # size  == size of equal num_samples == len(loader_test)

traj_file = os.path.join(args.save_dir, "trajs.json")
with open(traj_file, 'w') as f:
    json.dump(traj_dict, f)
print("Saved traj file: {}".format(traj_file))