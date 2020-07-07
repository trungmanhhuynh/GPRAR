"""
Train module

Author: Manh Huynh
Last Update: 06/23/2020

"""
import os
import time
import argparse
import torch
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from net.model import Model
from dataset import TrajectoryDataset



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
parser.add_argument('--save_dir', type=str, default='./save',
                     help='save directory')
parser.add_argument('--resume', type=str, default="",
                     help='resume a trained model?')

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
model = Model()
if(args.use_cuda) : model = model.cuda() 
#model.apply(weights_init)


# 3. define loss function 
mse_loss = torch.nn.MSELoss()  

# 4. train settings
optimizer = getattr(optim, args.optim)(model.parameters(), lr = args.learning_rate)


# 5. load check points ?
resume_epoch = 0 
if(args.resume != ""):
    resume_dict = torch.load(args.resume)
    model.load_state_dict(resume_dict['state_dict'])
    resume_epoch = resume_dict['e']

# 6. start training
for e in range(resume_epoch, args.nepochs):


    # 6.1 train 
    start_time = time.time()
    train_loss = 0 

    for train_it, samples in enumerate(loader_train):
        
        model.train()
        optimizer.zero_grad()    

        pose = Variable(samples[0])                        # pose ~ [batch_size, pose_features, obs_len, keypoints, instances]   
                                                           # e.g. ~ [128, 3, 10, 25, 1]
        gt_locations =  Variable(samples[1])               # gt_locations ~ [batch_size, pred_len, 2]

        if(args.use_cuda): 
            pose, gt_locations= pose.cuda(), gt_locations.cuda()

        #forward
        pred_locations = model(pose)                                      # pred_locations ~ [batch_size, pred_len, 2]
        loss = mse_loss(pred_locations, gt_locations)
        train_loss += loss.item()
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # print out
        if((e*len(loader_train) + train_it)  % args.info_fre == 0):
            print("iter:{}/{} train_loss:{:.5f} ".format(e*len(loader_train) + train_it , 
                                                          args.nepochs*len(loader_train), 
                                                          train_loss/(train_it+1)))

    train_loss /= len(loader_train)

    # 6.2 validate
    print("---validate---")
    model.eval()
    val_loss = 0; 
    for val_it, samples in enumerate(loader_val):
        
        pose = Variable(samples[0])                        # pose ~ [batch_size, pose_features, obs_len, keypoints, instances]   
                                                           # e.g. ~ [128, 3, 10, 25, 1]
        gt_locations =  Variable(samples[1])               # gt_locations ~ [batch_size, pred_len, 2]

        if(args.use_cuda): 
            pose, gt_locations= pose.cuda(), gt_locations.cuda()

        #forward
        pred_locations = model(pose)                                      # pred_locations ~ [batch_size, pred_len, 2]
        val_loss +=  mse_loss(pred_locations, gt_locations).item()


    val_loss /= len(loader_val)
    stop_time = time.time()


    print("epoch:{} train_loss:{:.5f} val_loss:{:.5f} time(ms):{:.2f}".format(
        (e + 1), train_loss, val_loss, (stop_time - start_time)*1000))


    # 7. save model 
    if( e % args.save_fre == 0):
        model_file = os.path.join(args.save_model_dir, "model_epoch_{}.pt".format(e))
        torch.save({'state_dict': model.state_dict(), 'e' :e}, model_file)
        print("saved model to file:", model_file)

    # 8. save log as json  
    
