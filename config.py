import argparse
import os

def read_args_constructor():

    parser = argparse.ArgumentParser()

    # network parameters
    parser.add_argument('--obs_len', type=int, default=10)
    parser.add_argument('--pred_len', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--optim', type=str, default='Adam',
                        help="ctype of optimizer: 'rmsprop' 'adam'")
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients to this magnitude')
    parser.add_argument('--nepochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='use gpu')
    parser.add_argument('--lr_step', type=int, default=50,
                        help='reduce learning rate every lr_step')

    # train/val parameters
    parser.add_argument('--data_dir', type=str, default="train_val_data/JAAD",
                        help='data dir consisting of train and validation data')
    parser.add_argument('--d_size', type=str, default="small",
                        help='d_size: small, medium, large')
    parser.add_argument('--resume', type=str, default="",
                        help='resume a trained model?')
    parser.add_argument('--hc_poses', action='store_true', default=False,
                        help='generate high confident poses')
    parser.add_argument('--add_noise', action='store_true', default=False,
                        help='add noise to high confident poses ')

    # debugging parameters
    parser.add_argument('--info_fre', type=int, default=10,
                        help='print out log every x interations')
    parser.add_argument('--save_fre', type=int, default=5,
                        help='save model every x epochs')
    parser.add_argument('--save_dir', type=str, default='./save',
                        help='save directory')
    parser.add_argument('--image_dir', type=str, default="/home/manhh/github/datasets/JAAD/images",
                        help='must be specified if plot_sample is true')

    args = parser.parse_args()

    # other argurments
    args.save_model_dir = os.path.join(args.save_dir, "reconstructor", "model")
    args.save_log_dir = os.path.join(args.save_dir, "reconstructor", "log")

    if(args.hc_poses):
        args.train_data = os.path.join(args.data_dir, "reconstructor", "train_{}_hcposes.joblib".format(args.d_size))
        args.val_data = os.path.join(args.data_dir, "reconstructor", "val_{}_hcposes.joblib".format(args.d_size))
        args.pose_res_dir = os.path.join(args.save_dir, "reconstructor", "pose_res_hcposes")

    else:
        args.train_data = os.path.join(args.data_dir, "reconstructor", "train_{}.joblib".format(args.d_size))
        args.val_data = os.path.join(args.data_dir, "reconstructor", "val_{}.joblib".format(args.d_size))
        args.pose_res_dir = os.path.join(args.save_dir, "reconstructor", "pose_res")

    # make dirs if not exists
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)
    if not os.path.exists(args.save_log_dir):
        os.makedirs(args.save_log_dir)
    if not os.path.exists(args.pose_res_dir):
        os.makedirs(args.pose_res_dir)

    print(args)

    return args


def read_args_predictor():

    parser = argparse.ArgumentParser()

    # network parameters
    parser.add_argument('--obs_len', type=int, default=10)
    parser.add_argument('--pred_len', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--optim', type=str, default='Adam',
                        help="ctype of optimizer: 'rmsprop' 'adam'")
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients to this magnitude')
    parser.add_argument('--nepochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='use gpu')
    parser.add_argument('--lr_step', type=int, default=5,
                        help='reduce learning rate every lr_step')
    # train/val parameters
    parser.add_argument('--data_dir', type=str, default="train_val_data/JAAD",
                        help='data dir consisting of train and validation data')
    parser.add_argument('--d_size', type=str, default="small",
                        help='d_size: small, medium, large')
    parser.add_argument('--resume', type=str, default="",
                        help='resume a trained model?')

    # debugging parameters
    parser.add_argument('--info_fre', type=int, default=10,
                        help='print out log every x interations')
    parser.add_argument('--save_fre', type=int, default=5,
                        help='save model every x epochs')
    parser.add_argument('--save_dir', type=str, default='./save',
                        help='save directory')
    parser.add_argument('--image_dir', type=str, default="/home/manhh/github/datasets/JAAD/images",
                        help='must be specified if plot_sample is true')

    args = parser.parse_args()

    # other argurments
    args.save_model_dir = os.path.join(args.save_dir, "predictor", "model")
    args.save_log_dir = os.path.join(args.save_dir, "predictor", "log")
    args.pose_res_dir = os.path.join(args.save_dir, "predictor", "pose_res")
    args.train_data = os.path.join(args.data_dir, "predictor", "train_{}.joblib".format(args.d_size))
    args.val_data = os.path.join(args.data_dir, "predictor", "val_{}.joblib".format(args.d_size))

    # make dirs if not exists
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)
    if not os.path.exists(args.save_log_dir):
        os.makedirs(args.save_log_dir)
    if not os.path.exists(args.pose_res_dir):
        os.makedirs(args.pose_res_dir)

    print(args)

    return args
