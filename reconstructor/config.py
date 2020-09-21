import argparse
import os

def read_args_constructor():

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
    parser.add_argument('--nepochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='use gpu')
    parser.add_argument('--lr_step', type=int, default=50,
                        help='reduce learning rate every lr_step')

    # train/val parameters
    parser.add_argument('--dset', type=str, default="JAAD",
                        help='JAAD or TITAN')
    parser.add_argument('--dsize', type=str, default="large",
                        help='small, medium, large')
    parser.add_argument('--resume', type=str, default="",
                        help='resume a trained model?')
    parser.add_argument('--add_noise', action='store_true', default=False,
                        help='add noise to pose')
    parser.add_argument('--flip', action='store_true', default=False,
                        help='flip pose')
    parser.add_argument('--info_fre', type=int, default=10,
                        help='print out log every x interations')
    parser.add_argument('--save_fre', type=int, default=10,
                        help='save model every x epochs')
    parser.add_argument('--save_dir', type=str, default='./save/reconstructor',
                        help='save model directory')
    parser.add_argument('--image_width', type=int, default=1920,
                        help='image_width')
    parser.add_argument('--image_height', type=str, default=1080,
                        help='image_height')
    # for testing
    parser.add_argument('--test_data', type=str, default="train_val_data/JAADreconstructor/val_large.joblib",
                        help='should be specified')
    parser.add_argument('--image_dir', type=str, default="/home/manhh/github/datasets/JAAD/images",
                        help='must be specified if plot_sample is true')
    parser.add_argument('--pose_res_dir', type=str, default=None,
                        help='directory of result pose')
    args = parser.parse_args()

    args.train_data = "train_val_data/{}/reconstructor/train_{}.joblib".format(args.dset, args.dsize)
    args.val_data = "train_val_data/{}/reconstructor/val_{}.joblib".format(args.dset, args.dsize)
    args.save_model_dir = os.path.join(args.save_dir, "model")
    args.save_log_dir = os.path.join(args.save_dir, "log")

    # make dirs if not exists
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)
    if not os.path.exists(args.save_log_dir):
        os.makedirs(args.save_log_dir)
    if args.pose_res_dir is not None:
        os.makedirs(args.pose_res_dir)

    print(args)

    return args
