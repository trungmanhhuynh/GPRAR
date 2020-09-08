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
    parser.add_argument('--nepochs', type=int, default=50,
                        help='number of epochs')
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='use gpu')
    parser.add_argument('--lr_step', type=int, default=50,
                        help='reduce learning rate every lr_step')

    # train/val parameters
    parser.add_argument('--train_data', type=str, default="train_val_data/JAAD/reconstructor/train_large_hcposes.joblib",
                        help='train  data')
    parser.add_argument('--val_data', type=str, default="train_val_data/JAAD/reconstructor/val_large_hcposes.joblib",
                        help='validation data')
    parser.add_argument('--resume', type=str, default="",
                        help='resume a trained model?')
    parser.add_argument('--hc_poses', action='store_true', default=False,
                        help='generate high confident poses')
    parser.add_argument('--add_noise', action='store_true', default=False,
                        help='add noise to high confident poses ')
    parser.add_argument('--flip', action='store_true', default=False,
                        help='flip pose')

    # debugging parameters
    parser.add_argument('--info_fre', type=int, default=10,
                        help='print out log every x interations')
    parser.add_argument('--save_fre', type=int, default=5,
                        help='save model every x epochs')
    parser.add_argument('--save_model_dir', type=str, default='./save/reconstructor/model',
                        help='save model directory')
    parser.add_argument('--save_log_dir', type=str, default='./save/reconstructor/log',
                        help='save model directory')
    parser.add_argument('--image_dir', type=str, default="/home/manhh/github/datasets/JAAD/images",
                        help='must be specified if plot_sample is true')
    parser.add_argument('--pose_res_dir', type=str, default="pose_results",
                        help='directory of result pose')

    args = parser.parse_args()

    # make dirs if not exists
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)
    if not os.path.exists(args.save_log_dir):
        os.makedirs(args.save_log_dir)
    if not os.path.exists(args.pose_res_dir):
        os.makedirs(args.pose_res_dir)

    print(args)

    return args
