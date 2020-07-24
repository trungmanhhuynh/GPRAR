
"""
Test module - Const-Vel model

Author: Manh Huynh
Last Update: 06/23/2020

"""
import os
import time
import json
import argparse
import torch
import numpy as np
import sys
sys.path.append("/home/manhh/github/Traj-STGCNN")

from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataset import TrajectoryDataset
from utils.utils import calculate_ade_fde, save_traj_json
from sota.const_vel.model import ConstVel


parser = argparse.ArgumentParser()
parser.add_argument('--obs_len', type=int, default=10)
parser.add_argument('--pred_len', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128, 
                    help='minibatch size')
parser.add_argument('--test_data', type=str,  default="train_val_data/JAAD/mini_size/val_data.joblib", 
                    help='file used for testing')
parser.add_argument('--use_cuda', action='store_true', default= True, 
                    help = 'use gpu')
parser.add_argument('--save_dir', type=str, default='./save',
                     help='save directory')

args = parser.parse_args()
args.save_model_dir = os.path.join(args.save_dir, "model")
if not os.path.exists(args.save_model_dir ):
    os.makedirs(args.save_model_dir )

print(args)


# Fixed the seed
np.random.seed(1)
torch.manual_seed(1)


# 1.prepare data
dset_test = TrajectoryDataset(
        args.test_data,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        flip = False
        )

loader_test = DataLoader(
        dset_test,
        batch_size=128, 
        shuffle =False,
        num_workers=0)

print("test datasize = {}".format(len(dset_test)))


# 2. create model
model = ConstVel(pred_len=args.pred_len)


# 3. define loss function 
mse_loss = torch.nn.MSELoss()  

# 4. Test
print("---testing---")
start_time = time.time()

test_loss = 0 ;  test_ade = 0 ; test_fde = 0 
traj_dict = {'video_names': [], 'image_names': [],  'person_ids': [], 'traj_pred': []}
for test_it, samples in enumerate(loader_test):
    
    pose_locations = Variable(samples[1])                        # pose ~ [batch_size, pose_features, obs_len, keypoints, instances]   
                                                       # e.g. ~ [128, 3, 10, 25, 1]
    gt_locations =  Variable(samples[2])               # gt_locations ~ [batch_size, pred_len, 2]

    if(args.use_cuda): 
        pose_locations, gt_locations= pose_locations.cuda(), gt_locations.cuda()

    #forward
    pred_locations = model(pose_locations)  
    print(pred_locations.shape)                                    # pred_locations ~ [batch_size, pred_len, 2]

    test_loss +=  mse_loss(pred_locations, gt_locations).item()

    # calculate ade/fde
    ade, fde = calculate_ade_fde(gt_locations.data.cpu(), pred_locations.data.cpu(), dset_test.loc_mean, dset_test.loc_var)
    test_ade += ade 
    test_fde += fde

    # get trajectories
    traj_dict = save_traj_json(traj_dict, pred_locations, samples[3], samples[4], samples[5], dset_test.loc_mean, dset_test.loc_var)


test_loss /= len(loader_test)
test_ade /=  len(loader_test)
test_fde /= len(loader_test)
stop_time = time.time()


print("test_loss:{:.5f} test_ade:{:.2f} test_fde:{:.2f} time(ms):{:.2f}".format(
    test_loss, test_ade, test_fde, (stop_time - start_time)*1000))

# save trajectories to file
for key in traj_dict:
    traj_dict[key] = sum(traj_dict[key], [] )           # size  == size of equal num_samples == len(loader_test)

traj_file = os.path.join(args.save_dir, "trajs.json")
with open(traj_file, 'w') as f:
    json.dump(traj_dict, f)