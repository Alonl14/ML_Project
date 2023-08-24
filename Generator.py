import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        n_z = 25

        self.main = nn.Sequential(
            nn.Linear(n_z, 25),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(25),
            nn.Linear(25, 25),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(25),
            nn.Linear(25, 25),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(25),
            nn.Linear(25, 9),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.BatchNorm1d(9),
            nn.Linear(9, 7),
            nn.Tanh()
        )

    def forward(self, input):
        return 5 * self.main(input)
