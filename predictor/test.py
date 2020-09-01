"""
Test Predictor
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
from load_dataset_predictor import TrajectoryDataset
from common.utils import calculate_ade_fde, save_traj
from config import read_args_predictor

def load_datasets(args):

    dset_val = TrajectoryDataset(
        args.val_data,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        flip=False
    )

    loader_val = DataLoader(
        dset_val,
        batch_size=128,
        shuffle=False,
        num_workers=0)

    return dset_val, loader_val


def load_model(args):
    '''
        load: models
        define loss, optimizer, scheduler
    '''
    model = Traj_STGCNN(mode="predictor")
    if(args.use_cuda):
        model = model.cuda()

    mse_loss = torch.nn.MSELoss()
    optimizer = getattr(optim, args.optim)(model.parameters(), lr=args.learning_rate)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    return model, mse_loss, optimizer, scheduler

def test(args, model, epoch, mse_loss, dset_val, loader_val):

    # 6.2 validate
    val_loss, val_ade, val_fde = 0, 0, 0
    model.eval()
    traj_dict = {'video_names': [], 'image_names': [], 'person_ids': [], 'traj_gt': [], 'traj_pred': [], 'pose': []}
    for val_it, samples in enumerate(loader_val):

        poses = Variable(samples['poses'])                        # pose ~ (batch_size, obs_len, pose_features)
        gt_locations = Variable(samples['gt_locations'])          # gt_locations ~ (batch_size, pred_len, 2)

        # read auxilary data
        video_names = samples['video_names']
        image_names = samples['image_names']
        person_ids = samples['person_ids']

        if(args.use_cuda):
            poses, gt_locations = poses.cuda(), gt_locations.cuda()

        # forward
        predicted_locations = model(poses)                                      # output ~ [batch_size, pred_len, 2]
        loss = mse_loss(predicted_locations, gt_locations)
        val_loss += loss.item()

        # calculate ade/fde
        ade, fde = calculate_ade_fde(gt_locations, predicted_locations, dset_val.loc_mean, dset_val.loc_var)
        val_ade += ade
        val_fde += fde

        # save predicted trajectories
        save_traj(traj_dict, predicted_locations, video_names, image_names, person_ids,
                  dset_val.loc_mean, dset_val.loc_var)

    # save predicted trajectories to file
    for key in traj_dict:
        traj_dict[key] = sum(traj_dict[key], [])           # size  == size of equal num_samples == len(loader_test)

    traj_file = os.path.join(args.save_dir, "predictor", "trajs.json")
    print("Saving predicted trajs to file: ", traj_file)
    with open(traj_file, 'w') as f:
        json.dump(traj_dict, f)

    return val_loss / len(loader_val), val_ade / len(loader_val), val_fde / len(loader_val)


def resume_model(args, resumed_epoch, model):
    resume_dict = torch.load(args.resume)
    model.load_state_dict(resume_dict['state_dict'])
    resumed_epoch = resume_dict['epoch']

    return resumed_epoch, model


if __name__ == "__main__":

    # 1. read argurments
    args = read_args_predictor()

    # 2. fixed randomization
    np.random.seed(1)
    torch.manual_seed(1)
    random.seed(1)

    # 3. load dataset
    dset_val, loader_val = load_datasets(args)
    print("val datasize = {}".format(len(dset_val)))

    # 4. load model
    model, mse_loss, _, _ = load_model(args)

    # 5. resume model
    resumed_epoch = 1
    if(args.resume != ""):
        resumed_epoch, model = resume_model(args, resumed_epoch, model)

    start_time = time.time()

    # 6. validate
    val_loss, val_ade, val_fde = test(args, model, resumed_epoch, mse_loss, dset_val, loader_val)

    print("epoch:{} val_loss:{:.5f} val_ade:{:.2f} val_fde:{:.2f} time(ms):{:.2f}".format(
        resumed_epoch, val_loss, val_ade, val_fde, (time.time() - start_time) * 1000))
