"""
Test Reconstructor
Author: Manh Huynh
"""
import time
import torch
import numpy as np
import random
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from net.traj_stgcnn import Traj_STGCNN
from load_dataset_reconstructor import PoseDataset
from common.utils import calculate_pose_ade, plot_pose_results, std_denormalize
from config import read_args_constructor


def load_datasets(args):

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

    return dset_val, loader_val


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
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    return model, mse_loss, optimizer, scheduler

def test(args, model, epoch, mse_loss, dset_val, loader_val):

    # 6.2 validate
    val_loss, val_ade = 0, 0
    model.eval()
    for val_it, samples in enumerate(loader_val):

        noisy_poses = Variable(samples['noisy_poses'])
        poses_gt = Variable(samples['poses_gt'])                         # pose ~ [batch_size, pose_features, obs_len, keypoints, instances]

        # read auxilary data
        video_names = samples['video_names']
        image_names = samples['image_names']

        if(args.use_cuda):
            noisy_poses, poses_gt = noisy_poses.cuda(), poses_gt.cuda()

        # forward
        pred_poses = model(noisy_poses)
        # pred_poses[present_idx] = noisy_poses[present_idx]

        loss = mse_loss(poses_gt, pred_poses)
        ade = calculate_pose_ade(poses_gt, pred_poses, dset_val.pose_mean, dset_val.pose_var)

        # de-normalize
        noisy_poses = std_denormalize(noisy_poses.data.cpu(), dset_val.pose_mean, dset_val.pose_var)
        poses_gt = std_denormalize(poses_gt.data.cpu(), dset_val.pose_mean, dset_val.pose_var)
        pred_poses = std_denormalize(pred_poses.data.cpu(), dset_val.pose_mean, dset_val.pose_var)

        # plot_pose_results(noisy_poses=noisy_poses,
        #                   poses_gt=poses_gt,
        #                   pred_poses=pred_poses,
        #                   epoch=epoch,
        #                   video_names=video_names,
        #                   image_names=image_names,
        #                   image_dir=args.image_dir,
        #                   pose_res_dir=args.pose_res_dir,
        #                   obs_len=args.obs_len)

        val_ade += ade
        val_loss += loss.item()

    return val_loss / len(loader_val), val_ade / len(loader_val)


def resume_model(args, resumed_epoch, model):
    resume_dict = torch.load(args.resume)
    model.load_state_dict(resume_dict['state_dict'])
    resumed_epoch = resume_dict['epoch']

    return resumed_epoch, model


if __name__ == "__main__":

    # 1. read argurments
    args = read_args_constructor()

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
    val_loss, val_ade = test(args, model, resumed_epoch, mse_loss, dset_val, loader_val)

    print("epoch:{} val_loss:{:.5f} val_ade:{:.2f} time(ms):{:.2f}".format(
        resumed_epoch, val_loss, val_ade, (time.time() - start_time) * 1000))
