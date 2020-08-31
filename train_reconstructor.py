"""
Train Reconstructor module
Author: Manh Huynh

"""
import os
import time
import json
import torch
import random
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from net.traj_stgcnn import Traj_STGCNN
from load_dataset_reconstructor import PoseDataset
from common.utils import calculate_pose_ade
from config import read_args_constructor


def load_datasets(args):
    # 1.prepare data
    dset_train = PoseDataset(
        args.train_data,
        generate_noisy_pose=args.add_noise,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        flip=False
    )

    loader_train = DataLoader(
        dset_train,
        batch_size=128,
        shuffle=True,
        num_workers=0)

    dset_val = PoseDataset(
        args.val_data,
        generate_noisy_pose=args.add_noise,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        flip=False
    )

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
    model = Traj_STGCNN(mode="reconstructor")
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

        noisy_poses = Variable(samples['noisy_poses'])
        poses_gt = Variable(samples['poses_gt'])                         # pose ~ [batch_size, pose_features, obs_len, keypoints, instances]
        # present_idx = samples['present_idx']

        if(args.use_cuda):
            noisy_poses, poses_gt = noisy_poses.cuda(), poses_gt.cuda()

        # forward
        optimizer.zero_grad()
        pred_pose = model(noisy_poses)
        loss = mse_loss(pred_pose, poses_gt)
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
    val_loss, val_ade = 0, 0
    model.eval()
    for val_it, samples in enumerate(loader_val):

        noisy_poses = Variable(samples['noisy_poses'])
        poses_gt = Variable(samples['poses_gt'])                         # pose ~ [batch_size, pose_features, obs_len, keypoints, instances]
        # present_idx = samples['present_idx']

        if(args.use_cuda):
            noisy_poses, poses_gt = noisy_poses.cuda(), poses_gt.cuda()

        # forward
        pred_poses = model(noisy_poses)
        loss = mse_loss(poses_gt, pred_poses)
        ade = calculate_pose_ade(poses_gt, pred_poses, dset_val.pose_mean, dset_val.pose_var)

        val_ade += ade
        val_loss += loss.item()

    return val_loss / len(loader_val), val_ade / len(loader_val)


def save_model(args, model, epoch):

    model_file = os.path.join(args.save_model_dir, "model_epoch_{}.pt".format(epoch))
    torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_file)
    print("saved model to file:", model_file)


def resume_model(args, resumed_epoch, model):
    resume_dict = torch.load(args.resume)
    model.load_state_dict(resume_dict['state_dict'])
    resumed_epoch = resume_dict['epoch']

    return resumed_epoch, model

def save_log(args, log_dict, train_loss, val_loss, epoch):

    # 8.1 save log as json
    log_dict['epoch'].append(epoch)
    log_dict['train_loss'].append(train_loss)
    log_dict['val_loss'].append(val_loss)


if __name__ == "__main__":

    # 1. read argurments
    args = read_args_constructor()

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
        resumed_epoch, model = resume_model(args, resumed_epoch, model)

    log_dict = {'epoch': [], 'train_loss': [], 'val_loss': []}
    for epoch in range(1, args.nepochs + 1):

        start_time = time.time()

        # 6. train
        train_loss = train(args, model, mse_loss, optimizer, scheduler, loader_train, epoch)

        # 7. validate
        val_loss, val_ade = validate(args, model, mse_loss, dset_val, loader_val)

        # 8. save model
        if(epoch % args.save_fre == 0):
            save_model(args, model, epoch)

        # 9. logging
        save_log(args, log_dict, train_loss, val_loss, epoch)

        print("epoch:{} lr:{} train_loss:{:.5f} val_loss:{:.5f} val_ade:{:.2f} time(ms):{:.2f}".format(
            epoch, scheduler.get_lr(), train_loss, val_loss, val_ade, (time.time() - start_time) * 1000))

    # 10. save log file
    logfile = os.path.join(args.save_log_dir, "log.json")
    print("Written log file to ", logfile)
    with open(logfile, 'w') as f:
        json.dump(log_dict, f)
