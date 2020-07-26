import torch
import torch.nn as nn
import torch.nn.functional as F
from net.graph import Graph



class ConvTemporalGraphical(nn.Module):

    # source: https://github.com/yysijie/st-gcn/blob/master/net/utils/tgcn.py
    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size  # spatial kernel size
        self.conv = nn.Conv2d(
            in_channels,                           
            out_channels * kernel_size,             # e.g.  64*3, where 3 is number of adjacent matrix
            kernel_size=(t_kernel_size, 1),         # (1, 1) by default
            padding=(t_padding, 0),                 # (0, 0) by default
            stride=(t_stride, 1),                   # (1, 1) by default
            dilation=(t_dilation, 1),               # (1, 1) by default
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size   # A.shape = (3,25,25)

        x = self.conv(x)    # x input ~ (batch_size, in_channels, height, width)
                            # x output ~ (batch_size, out_channels, height, width)
                            # e.g. (128,3,10, 25) -> (128,192,10,25) 

        n, kc, t, v = x.size()                                          # e.g. n = 128, kc = 192, t = 10, v = 25 
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)     # e.g. x.shape ~ (128, 3, 64, 10, 25)

        x = torch.einsum('nkctv,kvw->nctw', (x, A))                     # e.g. (128, 3, 64, 10, 25)*(3, 25, 25) -> (128, 64, 10, 25)

        return x.contiguous(), A


class st_gcn(nn.Module):
    # source: https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2                    # kernel_size = (9,3)
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,                       
                (kernel_size[0], 1),                # e.g. (9,  1)
                (stride, 1),                        # e.g. (1, 1)
                padding,                            # e.g. (4, 0)
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):

        res = self.residual(x)          # x.shape (128, 3, 10, 25)
        x, A = self.gcn(x, A)           # e.g. A.shape (3,25,25)
                                        # e.g. x.shape (128, 64, 10, 25) ~ (batch_size, out_channels, obs_len, nodes)
        x = self.tcn(x) + res           # (128, 64, 10, 25) --> (128, 64, 10, 25)


        return self.relu(x), A



class Model(nn.Module):

    # source: https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""Spatial temporal graph convolutional networks.

    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_args (dict): The arguments for building the graph
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """

    def __init__(self, pose_features = 3, output_feats = 2, obs_len = 10, pred_len = 10, edge_importance_weighting = False):
        super().__init__()

        # load graph
        self.graph = Graph()
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False) # A ~ [1,25,25] with strategy = 'uniform'
                                                                                 # and layout = 'pose_25'
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        self.data_bn = nn.BatchNorm1d(pose_features * A.size(1))

        self.st_gcn_networks = nn.ModuleList((
            st_gcn(pose_features, 64, (temporal_kernel_size, spatial_kernel_size), 1, residual=False),
            st_gcn(64, 64,  (temporal_kernel_size, spatial_kernel_size), 1),
            st_gcn(64, 64,  (temporal_kernel_size, spatial_kernel_size), 1),
            st_gcn(64, 64,  (temporal_kernel_size, spatial_kernel_size), 1),
            st_gcn(64, 128,  (temporal_kernel_size, spatial_kernel_size), 1),
            st_gcn(128, 128,  (temporal_kernel_size, spatial_kernel_size), 1),
            st_gcn(128, 128,  (temporal_kernel_size, spatial_kernel_size), 1),
            st_gcn(128, 256,  (temporal_kernel_size, spatial_kernel_size), 1),
            st_gcn(256, 256,  (temporal_kernel_size, spatial_kernel_size), 1),
            st_gcn(256, 256,  (temporal_kernel_size, spatial_kernel_size), 1),
        ))

        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # TCN for prediction
        self.fcn = nn.Conv2d(256, 2, kernel_size=1)

    def forward(self, x):

        # data normalization
        N, C, T, V, M = x.size()                    # x ~ [batch_size, pose_features, obs_len, nodes, instances]   
                                                    # e.g. ~ [128, 3, 10, 25, 1]
                                                    # e.g. N = 128, C = 3, T = 10, V = 25, M = 1

        x = x.permute(0, 4, 3, 1, 2).contiguous()  # (128, 3, 10, 25, 1) --> (128, 1, 25, 3, 10)
        x = x.view(N * M, V * C, T)                # (128, 1, 25, 3, 10) --> (128, 75, 10)
        x = self.data_bn(x)                         
        x = x.view(N, M, V, C, T)                  # (128, 75, 10) --> (128, 1, 25, 3, 10)
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # (128, 1, 25, 3, 10) --> ( 128, 1, 3, 10, 25) 
        x = x.view(N * M, C, T, V)                 # (128, 1, 3, 10, 25) --> (128, 3, 10, 25) 
                                                   # ~ (batch_size, in_channel, obs_len, nodes)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)          # e.g. x shape ~ (128 , 256 , 10, 25 ) 
                                                       #  ~ (batch_size, out_channels, obs_len, nodes)

        # global pooling
        # x = F.avg_pool2d(x, x.size()[2:])   # (128 , 256 , 10, 25 )  --> (128, 256, 1, 1)
        # x = x.view(N, M, -1, 1, 1).mean(dim=1)


        # pose representation 
        x = F.avg_pool2d(x, (1, 25) )      # (128 , 256 , 10, 25 ) --> (128 , 256 , 10, 1)   
                      
        # prediction
        x = self.fcn(x)                    # (128 , 256 , 10, 1) --> (128 , 2 , 10, 1)                      
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(x.size(0), x.size(1), -1)       # (128 , 2 , 10, 1) --> (128 , 10 , 2)
        

        return x