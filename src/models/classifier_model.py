import torch
from torch import nn


class ClassifierModel(nn.Module):
    def __init__(self, input_size):
        super(ClassifierModel, self).__init__()

        # Single softmax model
        self.classifier = nn.Linear(input_size, 2)

        # DNN softmax model
        # self.classifier = nn.Sequential(
        #     nn.Linear(input_size, 1024),
        #     nn.Dropout(0.2),
        #     nn.LeakyReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(1024, 512),
        #     nn.Dropout(0.2),
        #     nn.LeakyReLU(),
        #     nn.Linear(512, 256),
        #     nn.Dropout(0.2),
        #     nn.LeakyReLU(),
        #     nn.Linear(256, 128),
        #     nn.Dropout(0.2),
        #     nn.LeakyReLU(),
        #     nn.Linear(128, 2),
        # )
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        return self.classifier(x)
