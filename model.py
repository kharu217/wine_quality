import torch.nn as nn

class wine_q(nn.Module) :
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(11, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Linear(16, 1),
            nn.ReLU()
        )

    def forward(self, x) :
        return self.layers(x)