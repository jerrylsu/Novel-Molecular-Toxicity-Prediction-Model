import torch
from torch import nn


class SoftmaxModel(nn.Module):
    def __init__(self, input_size):
        super(SoftmaxModel, self).__init__()

        # Softmax model
        self.softmax_model = nn.Linear(input_size, 2)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def copy_weights(self, softmax_layer: torch.nn.Linear) -> None:
        """
        Utility method to copy the weights of self into the given encoder and decoder, where
        encoder and decoder should be instances of torch.nn.Linear.
        :param : encoder Linear unit
        :return: None
        """
        softmax_layer.weight.data.copy_(self.softmax_model.weight)
        softmax_layer.bias.data.copy_(self.softmax_model.bias)

    def forward(self, x):
        return self.softmax_model(x)
