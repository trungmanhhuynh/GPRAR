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
from utils.utils import calculate_ade_fde
from sota.tcnn.model import TCNN

parser = argparse.ArgumentParser()
parser.add_argument('--obs_len', type=int, default=10)
parser.add_argument('--pred_len', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128, 
                    help='minibatch size')
parser.add_argument('--train_data', type=str, default="train_val_data/JAAD/mini_size/train_data.joblib", 
                    help='file used for training')
parser.add_argument('--val_data', type=str,  default="train_val_data/JAAD/mini_size/val_data.joblib", 
                    help='file used for validation')
parser.add_argument('--learning_rate', type=float, default=0.0001, 
                    help='learning rate')
parser.add_argument('--optim', type=str, default='Adam', 
                    help="ctype of optimizer: 'rmsprop' 'adam'")
parser.add_argument('--grad_clip', type=float, default=10., 
                    help='clip gradients to this magnitude')
parser.add_argument('--nepochs', type=int, default= 50, 
                    help='number of epochs')
parser.add_argument('--info_fre', type=int, default= 10, 
                    help='print out log every x interations')
parser.add_argument('--save_fre', type=int, default= 5, 
                    help='save model every x epochs')
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
dset_train = TrajectoryDataset(
        args.train_data,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        flip = False
        )

loader_train = DataLoader(
        dset_train,
        batch_size=128, 
        shuffle =True,
        num_workers=0)

dset_val = TrajectoryDataset(
        args.val_data,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        flip = False
        )

loader_val = DataLoader(
        dset_val,
        batch_size=128, 
        shuffle =False,
        num_workers=0)
print("train/val datasize = {}/{}".format(len(dset_train), len(dset_val)))


# 2.load model
model = TCNN(args.channel_list, args.deconv_list, args.ksize_list, \
                        args.dc_ksize_list, args.inter_list, args.last_list, args.pad_list)
if(args.use_cuda) : 
    model.to_gpu(0)
#model.apply(weights_init)


# 4. train settings
optimizer = optimizers.Adam(alpha = args.learning_rate)
optimizer.setup(model)
optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))


# 5. load check points ?
resume_epoch = 0 
if(args.resume != ""):
    serializers.load_npz(args.resume, model)                  # currently has to start from begining, thus
                                                             # only work for testing, not work for actual resuming
                                                             # from a specific epoch


    #resume_epoch = resume_dict['e']

# 6. start training
log_dict = {'epoch': [], 'train_loss': [], 'val_loss': []} 
for e in range(resume_epoch, args.nepochs):

    # 6.1 train 
    start_time = time.time()
    train_loss = 0 

    for train_it, samples in enumerate(loader_train):
        
        chainer.config.train = True
        chainer.config.enable_backprop = True

        pose_locations = Variable(samples[1].numpy())              # pose ~ [batch_size, pose_features, obs_len, keypoints, instances]   
                                                           # e.g. ~ [128, 3, 10, 25, 1]
        gt_locations =  Variable(samples[2].numpy())               # gt_locations ~ [batch_size, pred_len, 2]

        if(args.use_cuda): 
            pose_locations.to_gpu(0), gt_locations.to_gpu(0)

        #forward
        model.cleargrads()
        pred_locations = model(pose_locations)                                      # pred_locations ~ [batch_size, pred_len, 2]
        loss = F.mean_squared_error(pred_locations, gt_locations)
        train_loss += loss.item()

        # backward
        loss.backward()
        loss.unchain_backward()
        optimizer.update()

        # print out
        if((e*len(loader_train) + train_it)  % args.info_fre == 0):
            print("iter:{}/{} train_loss:{:.5f} ".format(e*len(loader_train) + train_it , 
                                                          args.nepochs*len(loader_train), 
                                                          train_loss/(train_it+1)))

    train_loss /= len(loader_train)

    # 6.2 validate
    print("---validate---")
    chainer.config.train = False
    chainer.config.enable_backprop = False
    val_loss = 0 ;  val_ade = 0 ; val_fde = 0 
    for val_it, samples in enumerate(loader_val):
        
        pose_locations = Variable(samples[1].numpy())                        # pose ~ [batch_size, pose_features, obs_len, keypoints, instances]   
                                                           # e.g. ~ [128, 3, 10, 25, 1]
        gt_locations =  Variable(samples[2].numpy())               # gt_locations ~ [batch_size, pred_len, 2]

        if(args.use_cuda): 
            pose_locations.to_gpu(0), gt_locations.to_gpu(0)

        #forward
        pred_locations = model(pose_locations)                                      # pred_locations ~ [batch_size, pred_len, 2]
        loss = F.mean_squared_error(pred_locations, gt_locations)
        val_loss += loss.item()

        # calculate ade/fde
        ade, fde = calculate_ade_fde(torch.tensor(cuda.to_cpu(gt_locations.data)), torch.tensor(cuda.to_cpu(pred_locations.data)), dset_val.loc_mean, dset_val.loc_var)
        val_ade += ade 
        val_fde += fde


    val_loss /= len(loader_val)
    val_ade /=  len(loader_val)
    val_fde /= len(loader_val)
    stop_time = time.time()


    print("epoch:{} train_loss:{:.5f} val_loss:{:.5f} val_ade:{:.2f} val_fde:{:.2f} time(ms):{:.2f}".format(
        e, train_loss, val_loss, val_ade, val_fde, (stop_time - start_time)*1000))


    # 7. save model 
    if((e  +1) % args.save_fre  == 0):
        model_file = os.path.join(args.save_model_dir, "model_epoch_{}.pt".format(e))
        serializers.save_npz(model_file, model)
        print("saved model to file:", model_file)

    # 8.1 save log as json  
    log_dict['epoch'].append(e)
    log_dict['train_loss'].append(train_loss)
    log_dict['val_loss'].append(val_loss)

# 8.2 save log as json  
logfile = os.path.join(args.save_log_dir,"log.json")
with open(logfile, 'w') as f:
    json.dump(log_dict, f)