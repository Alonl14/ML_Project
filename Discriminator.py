import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(7, 25),
            nn.LeakyReLU(),
            nn.BatchNorm1d(25),
            nn.Linear(25, 25),
            nn.LeakyReLU(),
            nn.BatchNorm1d(25),
            nn.Linear(25, 10),
            nn.LeakyReLU(),
            nn.BatchNorm1d(10),
            nn.Linear(10, 1),
        )

    def forward(self, input):
        return self.main(input)
