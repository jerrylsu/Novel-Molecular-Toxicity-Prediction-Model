import torch
from torch import nn
from torch.autograd import Variable


class LDAutoEncoderLayer(nn.Module):
    def __init__(self, input_size, output_size):
        """非线性去噪自编码层，用于构造堆叠式自编码器。

        :param input_size: 输入特征的维度。
        :param output_size: 输出特征的维度。
        """
        super(LDAutoEncoderLayer, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_size, input_size),
            nn.ReLU(),
        )

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

    def forward(self, x):
        # 单独训练每一个自编码器
        x = x.detach()
        # 添加噪声
        # x = ...
        y = self.encoder(x)

        if self.training:
            x_reconstruct = self.decoder(y)
            # loss = self.criterion(x_reconstruct, Variable(x.data, requires_grad=False))
            loss = self.criterion(x_reconstruct, x.clone().detach().requires_grad_(False))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return y.detach(), loss if self.training else None


class StackedAutoEncoderModel(nn.Module):
    def __init__(self):
        """ 堆叠式自编码器，从非线性去噪自编码层构造。每一个去噪自编码层均是独立训练。
        """
        super(StackedAutoEncoderModel, self).__init__()

        self.ae1 = LDAutoEncoderLayer(5576, 6000)
        self.ae2 = LDAutoEncoderLayer(6000, 3000)
        self.ae3 = LDAutoEncoderLayer(3000, 1500)
        self.ae4 = LDAutoEncoderLayer(1500, 750)
        self.ae5 = LDAutoEncoderLayer(750, 375)
        self.ae6 = LDAutoEncoderLayer(375, 3)

    def forward(self, x):
        a1, loss1 = self.ae1(x)
        a2, loss2 = self.ae2(a1)
        a3, loss3 = self.ae3(a2)
        a4, loss4 = self.ae4(a3)
        a5, loss5 = self.ae5(a4)
        a6, loss6 = self.ae6(a5)

        if self.training:
            return a6, loss6
        else:
            return a6, self.reconstruct(a6)

    def reconstruct(self, x):
        a5_reconstruct = self.ae6.reconstruct(x)
        a4_reconstruct = self.ae5.reconstruct(a5_reconstruct)
        a3_reconstruct = self.ae4.reconstruct(a4_reconstruct)
        a2_reconstruct = self.ae3.reconstruct(a3_reconstruct)
        a1_reconstruct = self.ae2.reconstruct(a2_reconstruct)
        x_reconstruct = self.ae1.reconstruct(a1_reconstruct)
        return x_reconstruct


class ClassifierLayer(nn.Module):
    def __init__(self):
        super(ClassifierLayer, self).__init__()

        self.classifier = nn.Linear(3, 2)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        return self.classifier(x)
