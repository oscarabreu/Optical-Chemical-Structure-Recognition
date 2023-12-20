from efficientnet_pytorch import EfficientNet
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, dim, drop_rate):
        super(Encoder, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        in_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Linear(in_features=in_features, out_features=dim, bias=True)
        self.relu = nn.Relu()
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.backbone(x)  # pass input x through the backbone EfficientNet
        x = self.relu(x)      # apply ReLU activation function
        batch_size = x.size(0)
        dim = x.size(-1)
        x = x.view(batch_size, -1, dim)  # reshape x as needed
        x = self.dropout(x)  # apply dropout
        return x
