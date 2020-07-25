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
import sys
sys.path.append("/home/manhh/github/Traj-STGCNN")

import numpy as np

import chainer
import chainer.functions as F
from chainer import optimizers, Variable, cuda, serializers

import torch
from torch.utils.data import DataLoader

from dataset import TrajectoryDataset
from utils.utils import calculate_ade_fde, save_traj_json
from sota.tcnn.model import TCNN

parser = argparse.ArgumentParser()
parser.add_argument('--obs_len', type=int, default=10)
parser.add_argument('--pred_len', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128, 
                    help='minibatch size')
parser.add_argument('--test_data', type=str,  default="train_val_data/JAAD/mini_size/val_data.joblib", 
                    help='file used for testing')
parser.add_argument('--use_cuda', action='store_true', default= True, 
                    help = 'use gpu')
parser.add_argument('--save_dir', type=str, default='sota/tcnn/save',
                     help='save directory')
parser.add_argument('--resume', type=str, default="",
                     help='resume a trained model?')

# TCNN arguments
parser.add_argument('--inter_list', type=int, nargs="*", default=[256])
parser.add_argument('--last_list', type=int, nargs="*", default=[])
parser.add_argument('--channel_list', type=int, nargs="*", default=[32, 64, 128, 128])
parser.add_argument('--deconv_list', type=int, nargs="*", default=[256, 128, 64, 32])
parser.add_argument('--ksize_list', type=int, nargs="*", default=[3, 3, 3, 3])
parser.add_argument('--dc_ksize_list', type=int, nargs="*", default=[])
parser.add_argument('--pad_list', type=int, nargs="*", default=[])


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


# 2.load model
model = TCNN(args.channel_list, args.deconv_list, args.ksize_list, \
                        args.dc_ksize_list, args.inter_list, args.last_list, args.pad_list)
if(args.use_cuda) : 
    model.to_gpu(0)

# 5. load check points ?
if(args.resume != ""):
        serializers.load_npz(args.resume, model)             # currently has to start from begining, thus
                                                             # currently has to start from begining, thus
                                                             # only work for testing, not work for actual resuming
                                                             # from a specific epoch
# 6.2 validate
print("---validate---")
start_time = time.time()

chainer.config.train = False
chainer.config.enable_backprop = False
test_loss = 0 ;  test_ade = 0 ; test_fde = 0 
traj_dict = {'video_names': [], 'image_names': [],  'person_ids': [], 'traj_gt': [], 'traj_pred': [], 'pose': []}
for test_it, samples in enumerate(loader_test):
    
    pose_locations = Variable(samples[1].numpy())                        # pose ~ [batch_size, pose_features, obs_len, keypoints, instances]   
                                                       # e.g. ~ [128, 3, 10, 25, 1]
    gt_locations =  Variable(samples[2].numpy())               # gt_locations ~ [batch_size, pred_len, 2]

    if(args.use_cuda): 
        pose_locations.to_gpu(0), gt_locations.to_gpu(0)

    #forward
    pred_locations = model(pose_locations)                                      # pred_locations ~ [batch_size, pred_len, 2]
    loss = F.mean_squared_error(pred_locations, gt_locations)
    test_loss += loss.item()

    # calculate ade/fde
    ade, fde = calculate_ade_fde(torch.tensor(cuda.to_cpu(gt_locations.data)), torch.tensor(cuda.to_cpu(pred_locations.data)), dset_test.loc_mean, dset_test.loc_var)
    test_ade += ade 
    test_fde += fde

    # get trajectories
    traj_dict = save_traj_json(traj_dict, torch.tensor(cuda.to_cpu(pred_locations.data)), samples[3], samples[4], samples[5], dset_test.loc_mean, dset_test.loc_var)

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