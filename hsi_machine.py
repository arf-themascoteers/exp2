import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from torchvision.models import resnet50, ResNet50_Weights


class HsiMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.N_BANDS = 177
        self.band_net = nn.Sequential(
            nn.Linear(self.N_BANDS, 1),
            #nn.BatchNorm1d(num_features=64),
            #nn.ReLU(),
            #nn.Dropout(),
            #nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.band_net(x)
        return x

