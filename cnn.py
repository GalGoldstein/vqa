import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import platform

"""
https://medium.com/swlh/deep-learning-for-image-classification-creating-cnn-from-scratch-using-pytorch-d9eeb7039c12
"""


class CNN(nn.Module):
    def __init__(self, padding=0, pooling='max'):
        super(CNN, self).__init__()

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        running_on_linux = 'Linux' in platform.platform()
        # self.device = 'cpu' if (torch.cuda.is_available() and not running_on_linux) else self.device

        # formula to calc original_length (or width) of tensor after the conv layer:
        # (2 * padding_value) + previous_len_of_row_before_conv_layer - (kernel_size - 1) = len_of_row_after_conv_layer
        # for example: 2*2 + 224 - (3 - 1) = 226
        self.convolutions = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=padding), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2, 2) if pooling == 'max' else nn.AvgPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=padding), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2) if pooling == 'max' else nn.AvgPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=padding), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2) if pooling == 'max' else nn.AvgPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=padding), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2) if pooling == 'max' else nn.AvgPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2) if pooling == 'max' else nn.AvgPool2d(2, 2),
        )

        # note 1: We don't want padding in the last layers, since then the final output tensor will include
        # padding pixels, and this tensor is much smaller than the original image (e.g. 5x5x256)
        # note 2: BatchNorm2d is recommended after each conv layer, and the parameter is need to get is the current
        # number of filters (which is depth of tensor, which is output value of the last conv layer

    def forward(self, x):
        with torch.cuda.amp.autocast():
            x = self.convolutions(x)  # input: x.shape =  [batch_size, 3, 224, 224]
            x = x.permute(0, 2, 3, 1)  # [batch_size, 256, 5, 5] -> [batch_size, 5, 5, 256]
            x = x.reshape([x.size(0), x.size(1) * x.size(2), -1])  # [batch_size, 5, 5, 256]  -> [batch_size, 25, 256]
            return x  # [batch_size, 25, 256]. 25=K=Number of regions, 256=d=Dimension of each region


if __name__ == "__main__":
    cnn = CNN(padding=2, pooling='max')

    n_params = sum([len(params.detach().cpu().numpy().flatten()) for params in list(cnn.parameters())])
    print(f'============ # CNN parameters: {n_params}============')

    output = cnn(torch.randn((16, 3, 224, 224)))
