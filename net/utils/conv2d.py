import torch.nn as nn


class conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_size, 1),                    # e.g. (9,  1)
                stride=(stride, 1),                              # e.g. (1, 1)
                padding=(padding, 0),                            # e.g. (4, 0)
            ),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def __call__(self, data_in):
        # data_in shape (batch_size, in_channels, height, width)
        # or (batch_size, in_channels, T, 1)  for 1D convolution over observed traj
        return self.cnn(data_in)
