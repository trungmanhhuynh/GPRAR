"""
Train module of reconstructor

Author: Manh Huynh

"""
import os
import time
import json
import argparse
import torch
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from net.reconstructor import Reconstructor
from reconstructor.dataset import TrajectoryDataset
from reconstructor.utils import std_denormalize, calculate_ade, plot_results


# read arguments
parser = argparse.ArgumentParser()
parser.add_argument('--obs_len', type=int, default=10)
parser.add_argument('--pred_len', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128,
                    help='minibatch size')
parser.add_argument('--train_data', type=str, default="reconstructor/train_val_data/JAAD/small_train_data.joblib",
                    help='file used for training')
parser.add_argument('--val_data', type=str, default="reconstructor/train_val_data/JAAD/small_val_data.joblib",
                    help='file used for validation')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--optim', type=str, default='Adam',
                    help="ctype of optimizer: 'rmsprop' 'adam'")
parser.add_argument('--grad_clip', type=float, default=10.,
                    help='clip gradients to this magnitude')
parser.add_argument('--nepochs', type=int, default=100,
                    help='number of epochs')
parser.add_argument('--info_fre', type=int, default=10,
                    help='print out log every x interations')
parser.add_argument('--save_fre', type=int, default=5,
                    help='save model every x epochs')
parser.add_argument('--use_cuda', action='store_true', default=True,
                    help='use gpu')
parser.add_argument('--save_dir', type=str, default='reconstructor/save/small/',
                    help='save directory')
parser.add_argument('--resume', type=str, default="",
                    help='resume a trained model?')
parser.add_argument('--plot_sample', action='store_true', default=False,
                    help='plot val results?')
parser.add_argument('--image_dir', type=str, default="/home/manhh/github/datasets/JAAD/images",
                    help='must be specified if plot_sample is true')

args = parser.parse_args()

args.save_model_dir = os.path.join(args.save_dir, "model")
args.save_log_dir = os.path.join(args.save_dir, "log")
args.result_dir = os.path.join(args.save_dir, "result")

if not os.path.exists(args.save_model_dir):
    os.makedirs(args.save_model_dir)
if not os.path.exists(args.save_log_dir):
    os.makedirs(args.save_log_dir)
if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)
print(args)


# Fixed random seed
np.random.seed(1)
torch.manual_seed(1)


# prepare data
dset_train = TrajectoryDataset(
    args.train_data,
    obs_len=args.obs_len,
    pred_len=args.pred_len,
    flip=False,
)

loader_train = DataLoader(
    dset_train,
    batch_size=128,
    shuffle=True,
    num_workers=0)

dset_val = TrajectoryDataset(
    args.val_data,
    obs_len=args.obs_len,
    pred_len=args.pred_len,
    flip=False,
)

loader_val = DataLoader(
    dset_val,
    batch_size=128,
    shuffle=False,
    num_workers=0)

print("train/val datasize = {}/{}".format(len(dset_train), len(dset_val)))


# load model
model = Reconstructor(batch_size=args.batch_size,
                      in_channels=3,
                      out_channels=3,
                      obs_len=args.obs_len,
                      pred_len=args.pred_len,
                      num_keypoints=25,
                      edge_importance_weighting=False)
if(args.use_cuda):
    model = model.cuda()

# define loss function
mse_loss = torch.nn.MSELoss()

# train settings
optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.learning_rate)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)


# load check points ?
resume_epoch = 0
if(args.resume != ""):
    resume_dict = torch.load(args.resume)
    model.load_state_dict(resume_dict['state_dict'])
    resume_epoch = resume_dict['e']

# start training
log_dict = {'epoch': [], 'train_loss': [], 'val_loss': []}
for e in range(resume_epoch, args.nepochs):

    # 6.1 train
    start_time = time.time()
    train_loss = 0

    model.train()
    for train_it, samples in enumerate(loader_train):

        poses = Variable(samples['poses'])                        # pose ~ [batch_size, pose_features, obs_len, keypoints, instances]
        gt_poses = Variable(samples['poses_gt'])                        # pose ~ [batch_size, pose_features, obs_len, keypoints, instances]

        if(args.use_cuda):
            poses, gt_poses = poses.cuda(), gt_poses.cuda(),

        # forward
        optimizer.zero_grad()
        pred_poses = model(poses)                                      # pred_poses ~ [batch_size, pred_len, 2] or pose ~ (batch_size, pred_len, 75)

        loss = mse_loss(pred_poses, gt_poses)
        train_loss += loss.item()

        # backward
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # print out
        if((e * len(loader_train) + train_it) % args.info_fre == 0):
            print("iter:{}/{} train_loss:{:.5f} ".format(e * len(loader_train) + train_it,
                                                         args.nepochs * len(loader_train),
                                                         train_loss / (train_it + 1)))
    train_loss /= len(loader_train)

    # validate
    print("---validate---")
    val_loss, val_ade = 0, 0
    # model.eval()
    for val_it, samples in enumerate(loader_val):

        poses = Variable(samples['poses'])                        # pose ~ [batch_size, obs_len, num_keypoints*num_feats]
        gt_poses = Variable(samples['poses_gt'])

        if(args.use_cuda):
            poses, gt_poses = poses.cuda(), gt_poses.cuda(),

        # forward
        pred_poses = model(poses)                                      # pred_poses ~ [batch_size, pred_len, 2]
        loss = mse_loss(pred_poses, gt_poses)
        val_loss += loss.item()

        # de-normalize
        gt_poses = std_denormalize(gt_poses, dset_val.pose_mean, dset_val.pose_var)
        pred_poses = std_denormalize(pred_poses, dset_val.pose_mean, dset_val.pose_var)

        # calculate mse error in pixels
        ade = calculate_ade(gt_poses, pred_poses)
        val_ade += ade

        # plot samples
        if(args.plot_sample and val_it == 0):
            # plot results at iteration val_it
            plot_results(gt_poses=gt_poses,
                         pred_poses=pred_poses,
                         bboxes=samples['bboxes'],
                         epoch=e,
                         video_name=samples['video_names'],
                         image_name=samples['image_names'],
                         image_dir=args.image_dir,
                         result_dir=args.result_dir)

    val_loss /= len(loader_val)
    val_ade /= len(loader_val)
    stop_time = time.time()
    scheduler.step()    # update learning rate

    print("epoch:{} lr:{} train_loss:{:.5f} val_loss:{:.5f} val_ade:{:.2f} time(ms):{:.2f}".format(
        e, scheduler.get_lr(), train_loss, val_loss, val_ade, (stop_time - start_time) * 1000))

    # 7. save model
    if((e + 1) % args.save_fre == 0):
        model_file = os.path.join(args.save_model_dir, "model_epoch_{}.pt".format(e))
        torch.save({'state_dict': model.state_dict(), 'e': e}, model_file)
        print("saved model to file:", model_file)

    # 8.1 save log as json
    log_dict['epoch'].append(e)
    log_dict['train_loss'].append(train_loss)
    log_dict['val_loss'].append(val_loss)

# 8.2 save log as json
logfile = os.path.join(args.save_log_dir, "log.json")
with open(logfile, 'w') as f:
    json.dump(log_dict, f)
