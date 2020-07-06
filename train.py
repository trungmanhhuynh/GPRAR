"""
Train module

Author: Manh Huynh
Last Update: 06/23/2020

"""
import argparse
import torch
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
args = parser.parse_args()

print('*'*30)
print("Start training....")
print(args)


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


# 2.load model
model = Model()
#model.apply(weights_init)


# 3. define loss function 
mse_loss = torch.nn.MSELoss()  

# 4. train settings
optimizer = getattr(optim, args.optim)(model.parameters(), lr = args.learning_rate)


# 5. load check points ?




# 6. start training
for e in range(args.nepochs):


    for it, samples in enumerate(loader_train):
        
        model.train()
        optimizer.zero_grad()    

        pose = Variable(samples[0])                        # pose ~ [batch_size, pose_features, obs_len, keypoints, instances]   
                                                           # e.g. ~ [128, 3, 10, 25, 1]
        gt_locations =  Variable(samples[1])               # gt_locations ~ [batch_size, pred_len, 2]

        #forward
        pred_locations = model(pose)                                      # pred_locations ~ [batch_size, pred_len, 2]
        loss = mse_loss(pred_locations, gt_locations)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)



        print(loss.item()) 



    # 6.1 validate






