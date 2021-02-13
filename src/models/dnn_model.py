import torch
from torch import nn


class DNNModel(nn.Module):
    def __init__(self, input_size):
        super(DNNModel, self).__init__()

        # DNN model
        self.dnn_model = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(2048, 1024),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(128, 2),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        return self.dnn_model(x)
