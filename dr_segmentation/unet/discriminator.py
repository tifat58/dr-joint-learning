import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticModel(nn.Module):

    def __init__(self):
        super(CriticModel, self).__init__()

        # self.critic = nn.Sequential(
        #     nn.Linear(524288, 2048),
        #     nn.ReLU(),
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1)
        # )

        self.critic = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x = x.view(x.shape[0], -1)
        x = self.critic(x)
        return x