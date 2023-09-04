import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, N_z):
        super().__init__()

        self.N_z = N_z

        self.main = nn.Sequential(
            nn.Linear(self.N_z, 50, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50, affine=True),
            nn.Linear(50, 25, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(25, affine=True),
            nn.Linear(25, 11, bias=False),
            nn.LeakyReLU(),
            nn.BatchNorm1d(11, affine=True),
            nn.Linear(11, 8)
        )

    def forward(self, input):
        return self.main(input)
