import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from torchvision.models import resnet50, ResNet50_Weights


class HsiConv1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.N_BANDS = 177
        self.band_net = nn.Sequential(
            nn.Conv1d(1,8,5),
            nn.LeakyReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(8, 16, 5),
            nn.LeakyReLU(),
            nn.MaxPool1d(3),
            nn.Conv1d(16, 32, 5),
            nn.LeakyReLU(),
            nn.MaxPool1d(5),
            nn.Flatten(),
            nn.Linear(64,16),
            nn.Dropout(),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        x = self.band_net(x)
        return x


if __name__ == "__main__":
    machine = HsiConv1d()
    x = torch.rand(5,1,177)
    x = machine(x)