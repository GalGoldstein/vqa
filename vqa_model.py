import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import cnn
import lstm

class VQA(nn.Module):
    def __init__(self):
        super(VQA, self).__init__()
        self.cnn = cnn.Xception()
        self.lstm = lstm.LSTM()

    def forward(self, im):
        pass
