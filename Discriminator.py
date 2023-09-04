import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(8, 50),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(1),
            nn.Linear(50, 25),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(1),
            nn.Linear(25, 10),
            nn.LeakyReLU(),
            nn.InstanceNorm1d(1),
            nn.Linear(10, 1, bias=False)
        )

    def forward(self, input):
        return self.main(input)
