import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class Generator(nn.Module):
    def __init__(self,data_size,latent_dim):
        super(Generator, self).__init__()
        
        self.relu = nn.ReLU(inplace=True)
        
        self.linear1 = nn.Linear(data_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.mish = Mish()
        self.linear2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512, latent_dim)
        self.bn3 = nn.BatchNorm1d(latent_dim)
        
        self.linear4 = nn.Linear(latent_dim, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.linear5 = nn.Linear(512, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.linear6 = nn.Linear(1024, data_size)
        

    def forward(self, x):

        x1 = self.linear1(x)
        x2 = self.bn1(x1)
        x3 = self.mish(x2)
        x4 = self.linear2(x3)
        x5 = self.bn2(x4)
        x6 = self.mish(x5)
        x7 = self.linear3(x6)
        x8 = self.bn3(x7)
        x8 = self.mish(x8)
        
        x9 = self.linear4(x8)
        x10 = self.bn4(x9)
        x11 = self.mish(x10)
        x12 = self.linear5(x11 + x6)
        x13 = self.bn5(x12)
        x14 = self.mish(x13)
        x15 = self.linear6(x14 + x3)

        return self.relu(x15 + x)


class Discriminator(nn.Module):
    def __init__(self,data_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_size, 1024),
            Mish(),
            nn.Linear(1024, 512),
            Mish(),
            nn.Linear(512, 256),
            Mish(),
        )

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(256, 1))

    def forward(self, data):
        out = self.model(data)
        validity = self.adv_layer(out)
        return validity
