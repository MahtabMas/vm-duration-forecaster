from torch import nn


class VMDurationNet(nn.Module):
    def __init__(self, number_of_features=11, dropout_rate=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(number_of_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1),
        )

    def forward(self, job_features):
        return self.layers(job_features).squeeze(1)