import argparse
import os

def read_args():

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

    # train/val parameters
    parser.add_argument('--data_dir', type=str, default="data/JAAD",
                        help='data dir consisting of train and validation data')
    parser.add_argument('--data_size', type=str, default="small",
                        help='data_size: small, medium, large')
    parser.add_argument('--resume', type=str, default="",
                        help='resume a trained model?')
    parser.add_argument('--reconstruct_pose', action='store_true', default=False,
                        help='run mode pose reconstruction')
    parser.add_argument('--occl_ratio', type=float, default=0,
                        help='occlusion ratio for reconstruction mode')

    # debugging parameters
    parser.add_argument('--info_fre', type=int, default=10,
                        help='print out log every x interations')
    parser.add_argument('--save_fre', type=int, default=5,
                        help='save model every x epochs')
    parser.add_argument('--save_dir', type=str, default='./save',
                        help='save directory')

    args = parser.parse_args()

    # other argurments
    args.save_model_dir = os.path.join(args.save_dir, "model")
    args.save_log_dir = os.path.join(args.save_dir, "log")
    args.train_data = os.path.join(args.data_dir, args.data_size, "train_data.joblib")
    args.val_data = os.path.join(args.data_dir, args.data_size, "val_data.joblib")

    # make dirs if not exists
    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)
    if not os.path.exists(args.save_log_dir):
        os.makedirs(args.save_log_dir)

    print(args)

    return args
