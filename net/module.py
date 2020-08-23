"""
Module

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from net.graph import Graph


class conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bn=True):
        super().__init__()

        if(use_bn):
            self.cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(kernel_size, 1),                # e.g. (9,  1)
                    stride=(stride, 1),                         # e.g. (1, 1)
                    padding=(padding, 0),                            # e.g. (4, 0)
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        else:
            self.cnn = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(kernel_size, 1),                # e.g. (9,  1)
                    stride=(stride, 1),                         # e.g. (1, 1)
                    padding=(padding, 0),                            # e.g. (4, 0)
                ),
            )

    def __call__(self, pose_in):
        # pose_in shape must be (batch_size, in_channels, height, width)
        # or (batch_size, in_channels, obs_len, 1)  for 1D convolution over observed traj

        return self.cnn(pose_in)


class deconv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dropout=0.2):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, 1),                # e.g. (9,  1)
                stride=(stride, 1),                         # e.g. (1, 1)
                padding=(padding, 0),                            # e.g. (4, 0)
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def __call__(self, pose_in):
        # pose_in shape must be (batch_size, in_channels, height, width)
        # or (batch_size, in_channels, obs_len, 1)  for 1D convolution over observed traj

        return self.cnn(pose_in)


class sgcn(nn.Module):

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
        - Input[1]: Input graph adjacency matripose_in in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matripose_in for output data in :math:`(K, V, V)` format

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
                 bias=True,
                 dropout=0):
        super().__init__()

        self.kernel_size = kernel_size  # spatial kernel size
        self.sgcn = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * kernel_size,             # e.g.  64*3, where 3 is number of adjacent matripose_in
                kernel_size=(t_kernel_size, 1),         # (1, 1) by default
                padding=(t_padding, 0),                 # (0, 0) by default
                stride=(t_stride, 1),                   # (1, 1) by default
                dilation=(t_dilation, 1),               # (1, 1) by default
                bias=bias),
            nn.BatchNorm2d(out_channels * kernel_size),
            # nn.ReLU(inplace=True),
            nn.Dropout(dropout, inplace=True),
        )

    def forward(self, pose_in, A):

        assert A.size(0) == self.kernel_size   # A.shape = (3,25,25)
        pose_in = self.sgcn(pose_in)    # pose_in input ~ (batch_size, in_channels, height, width)
        # pose_in output ~ (batch_size, out_channels, height, width)
        # e.g. (128,3,10, 25) -> (128,192,10,25)

        n, kc, t, v = pose_in.size()                                          # e.g. n = 128, kc = 192, t = 10, v = 25
        pose_in = pose_in.view(n, self.kernel_size, kc // self.kernel_size, t, v)     # e.g. pose_in.shape ~ (128, 3, 64, 10, 25)

        pose_in = torch.einsum('nkctv,kvw->nctw', (pose_in, A))                     # e.g. (128, 3, 64, 10, 25)*(3, 25, 25) -> (128, 64, 10, 25)

        return pose_in.contiguous(), A


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
        - Input[1]: Input graph adjacency matripose_in in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matripose_in for output data in :math:`(K, V, V)` format

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

        # spatial graph cnn
        self.sgcn = sgcn(in_channels=in_channels,
                         out_channels=out_channels,
                         kernel_size=kernel_size[1],
                         dropout=dropout)

        # temporal graph cnn
        self.tgcn = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),                # e.g. (9,  1)
                (stride, 1),                        # e.g. (1, 1)
                padding,                            # e.g. (4, 0)
            ),
            nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
            nn.Dropout(dropout, inplace=True)
        )

        # residual connection
        if not residual:
            self.residual = lambda pose_in: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda pose_in: pose_in
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
                # nn.ReLU(inplace=True)
            )

        # self.relu = nn.ReLU(inplace=True)

    def forward(self, pose_in, A):

        res = self.residual(pose_in)          # pose_in.shape (128, 3, 10, 25)
        pose_in, A = self.sgcn(pose_in, A)           # e.g. A.shape (3,25,25)
        # e.g. pose_in.shape (128, 64, 10, 25) ~ (batch_size, out_channels, obs_len, nodes)
        pose_in = self.tgcn(pose_in) + res           # (128, 64, 10, 25) --> (128, 64, 10, 25)

        return pose_in, A
