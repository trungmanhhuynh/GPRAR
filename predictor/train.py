"""
Train Predictor
Author: Manh Huynh
"""
import os
import time
import json
import torch
import random
import numpy as np
import copy
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from net.traj_stgcnn import Traj_STGCNN
from predictor.load_dataset import TrajectoryDataset
from common.utils import calculate_ade_fde
from predictor.config import read_args_predictor


def load_datasets(args):
    # 1.prepare data
    dset_train = TrajectoryDataset(
        args.train_data,
        args)

    loader_train = DataLoader(
        dset_train,
        batch_size=128,
        shuffle=True,
        num_workers=0)

    dset_val = TrajectoryDataset(
        args.val_data,
        args,
        mean=dset_train.mean,
        var=dset_train.var)

    loader_val = DataLoader(
        dset_val,
        batch_size=128,
        shuffle=False,
        num_workers=0)

    return dset_train, loader_train, dset_val, loader_val


def load_model(args):
    '''
        load: models
        define loss, optimizer, scheduler
    '''
    model = Traj_STGCNN("predictor")
    if(args.use_cuda):
        model = model.cuda()

    mse_loss = torch.nn.MSELoss()
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.5)

    return model, mse_loss, optimizer, scheduler

def train(args, model, mse_loss, optimizer, scheduler, loader_train, epoch):

    model.train()
    train_loss = 0
    for train_it, samples in enumerate(loader_train):

        poses = Variable(samples['poses'])                               # pose ~ [batch_size, pose_features, obs_len, keypoints, instances]
        gt_locations = Variable(samples['gt_locations'])                 # gt_locations ~ [batch_size, pred_len, 2]
        obs_locations = Variable(samples['obs_locations'])
        missing_keypoints = samples['missing_keypoints']
        flow = Variable(samples['flow'])

        if(args.use_cuda):
            poses, gt_locations = poses.cuda(), gt_locations.cuda()
            obs_locations = obs_locations.cuda()
            flow = flow.cuda()
            missing_keypoints = missing_keypoints.cuda()

        # forward
        optimizer.zero_grad()
        pred_locations = model(pose_in=poses, flow_in=flow, missing_keypoints=missing_keypoints)                                      # output ~ [batch_size, pred_len, 2]
        loss = mse_loss(pred_locations, gt_locations)
        train_loss += loss.item()

        # backward
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        # print out
        if((epoch * len(loader_train) + train_it) % args.info_fre == 0):
            print("iter:{}/{} train_loss:{:.5f} ".format(epoch * len(loader_train) + train_it,
                                                         args.nepochs * len(loader_train),
                                                         train_loss / (train_it + 1)))
    scheduler.step()    # update learning rate
    return train_loss / len(loader_train)

def validate(args, model, mse_loss, dset_val, loader_val):

    # 6.2 validate
    val_loss, val_ade, val_fde = 0, 0, 0
    model.eval()
    for val_it, samples in enumerate(loader_val):

        poses = Variable(samples['poses'])                        # pose ~ (batch_size, obs_len, pose_features)
        gt_locations = Variable(samples['gt_locations'])          # gt_locations ~ (batch_size, pred_len, 2)
        missing_keypoints = samples['missing_keypoints']
        obs_locations = Variable(samples['obs_locations'])
        flow = Variable(samples['flow'])

        if(args.use_cuda):
            poses, gt_locations = poses.cuda(), gt_locations.cuda()
            obs_locations = obs_locations.cuda()
            flow = flow.cuda()
            missing_keypoints = missing_keypoints.cuda()

        # forward
        pred_locations = model(pose_in=poses, flow_in=flow, missing_keypoints=missing_keypoints)                                      # output ~ [batch_size, pred_len, 2]
        loss = mse_loss(pred_locations, gt_locations)
        val_loss += loss.item()

        # calculate ade/fde
        ade, fde = calculate_ade_fde(gt_locations, pred_locations, dset_val.loc_mean, dset_val.loc_var)
        val_ade += ade
        val_fde += fde

    return val_loss / len(loader_val), val_ade / len(loader_val), val_fde / len(loader_val)


def save_model(args, model, mean, var, epoch, best_model=False):

    if(best_model):
        model_file = os.path.join(args.save_model_dir, "model_best.pt")
    else:
        model_file = os.path.join(args.save_model_dir, "model_epoch_{}.pt".format(epoch))

    torch.save({'state_dict': model.state_dict(),
                'epoch': epoch,
                'mean': mean,
                'var': var,
                }, model_file)

    print("saved model to file:", model_file)


def resume_model(args, resumed_epoch, model):
    resume_dict = torch.load(args.resume)
    model.load_state_dict(resume_dict['state_dict'])
    resumed_epoch = resume_dict['epoch']

    print("resumed model:", resume_dict['epoch'])

    return resumed_epoch, model

def save_log(args, log_dict, train_loss, val_loss, epoch):

    # 8.1 save log as json
    log_dict['epoch'].append(epoch)
    log_dict['train_loss'].append(train_loss)
    log_dict['val_loss'].append(val_loss)


if __name__ == "__main__":

    # 1. read argurments
    args = read_args_predictor()

    # 2. fixed randomization
    np.random.seed(1)
    torch.manual_seed(1)
    random.seed(1)
    # 3. load dataset
    dset_train, loader_train, dset_val, loader_val = load_datasets(args)
    print("train/val datasize = {}/{}".format(len(dset_train), len(dset_val)))

    # 4. load model
    model, mse_loss, optimizer, scheduler = load_model(args)

    # 5. resume model
    resumed_epoch = 1
    if(args.resume != ""):
        _, model = resume_model(args, resumed_epoch, model)

    log_dict = {'epoch': [], 'train_loss': [], 'val_loss': []}
    best_epoch, best_val_ade, best_val_fde = 0, 10000, 10000
    for epoch in range(1, args.nepochs + 1):

        start_time = time.time()

        # 6. train
        train_loss = train(args, model, mse_loss, optimizer, scheduler, loader_train, epoch)

        # 7. validate
        val_loss, val_ade, val_fde = validate(args, model, mse_loss, dset_val, loader_val)

        # Save best model
        if(val_ade < best_val_ade):
            best_val_ade, best_val_fde, best_epoch = val_ade, val_fde, epoch
            save_model(args, model, dset_train.mean, dset_train.var, best_epoch, best_model=True)

        # 8. save model after args.save_fre
        if(epoch % args.save_fre == 0):
            save_model(args, model, dset_train.mean, dset_train.var, epoch)

        # 9. logging
        save_log(args, log_dict, train_loss, val_loss, epoch)

        print("epoch:{} lr:{} train_loss:{:.5f} val_loss:{:.5f} val_ade:{:.2f} val_fde:{:.2f} time(ms):{:.2f}".format(
            epoch, scheduler.get_lr(), train_loss, val_loss, val_ade, val_fde, (time.time() - start_time) * 1000))
        print("best_epoch:{} best_val_ade:{:.2f} best_val_fde:{:.2f}".format(
            best_epoch, best_val_ade, best_val_fde))

    # 10. save log file
    logfile = os.path.join(args.save_log_dir, "log.json")
    print("Written log file to ", logfile)
    with open(logfile, 'w') as f:
        json.dump(log_dict, f)
