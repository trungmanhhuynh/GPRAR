import torch
from torch import nn

class LSTM(nn.Module):
    def __init__(self, obs_len = 10, pred_len = 10, dropout = 0.1, use_cuda = True):
        super(LSTM, self).__init__()

        self.use_cuda = use_cuda
        self.obs_len = obs_len
        self.pred_len = pred_len

        self.rnn_size = 128                                      # rnn_size of ALL LSTMs
        self.input_size = 2     
        self.output_size = 2

        self.lstm    = nn.LSTM(self.input_size, self.rnn_size, num_layers=1) 
        self.relu = nn.ReLU() 
        self.dropout = torch.nn.Dropout(dropout) 
        self.last  = nn.Linear(self.rnn_size, self.output_size)


    def init_target_hidden_states(self, batch_size):
       # Initialize states for all targets in current batch
        h0 = torch.zeros(1, batch_size, self.rnn_size)
        c0 = torch.zeros(1, batch_size, self.rnn_size)
        if(self.use_cuda): 
            h0, c0 = h0.cuda(), c0.cuda() 
        return h0, c0

    def forward(self, traj_in):

        #  traj_in ~  # [batch_size, obs_len , 2]
        batch_size = traj_in.shape[0]
        h_t, c_t = self.init_target_hidden_states(batch_size)
        traj_in = traj_in.permute(1,0,2)                     #  [obs_len, batch_size , 2]   

        # obseve duration - learn hidden states
        for t in range(self.obs_len):
           traj_t, h_t, c_t = self.forward_t(traj_in[t,:,:].unsqueeze(0) , h_t, c_t)

        
        # predict 
        traj_pred = [] 
        for t in range(self.pred_len):
            traj_pred.append(traj_t)
            traj_t, h_t, c_t = self.forward_t(traj_t, h_t, c_t)

        traj_pred =torch.stack(traj_pred).squeeze(1)         #  [pred_len, batch_size, 2]    
        traj_pred = traj_pred.permute(1,0,2)                 #  [batch_size, pred_len, 2]   
   
        return traj_pred

    def forward_t(self, traj_t, h_t, c_t):

        # traj_t ~ [1, batch_size , input_size] 
        output, (h_next, c_next) = self.lstm(traj_t, (h_t, c_t)) #  [1, batch_size, rnn_size]
        output= self.relu(output)
        output= self.dropout(output)
        pred_t = self.last(output)                                 #  [1, batch_size, output_size]

        return pred_t, h_next, c_next

    __call__ = forward