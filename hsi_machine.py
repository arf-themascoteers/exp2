import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from torchvision.models import resnet50, ResNet50_Weights


class HsiMachine(nn.Module):
    def __init__(self):
        super().__init__()
        self.N_BANDS = 190
        self.band_net = nn.Sequential(
            nn.Linear(self.N_BANDS, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Dropout(),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.band_net(x)
        return x

