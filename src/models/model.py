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
            # nn.ReLU(),
            nn.ELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_size, input_size),
            # nn.ReLU(),
            nn.ELU(),
        )

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.001)

    def forward(self, x):
        # 单独训练每一个自编码器
        x = x.detach()
        # 添加噪声
        # x = ...
        y = self.encoder(x)

        if self.training:
            x_reconstruct = self.decoder(y)
            loss = self.criterion(x_reconstruct, Variable(x.data, requires_grad=False))
            # loss = self.criterion(x_reconstruct, x.clone().detach().requires_grad_(False))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return y.detach(), loss if self.training else None

    def reconstruct(self, x):
        return self.decoder(x)


class StackedAutoEncoderModel(nn.Module):
    def __init__(self, input_size, output_size: int = 3):
        """ 堆叠式自编码器，从非线性去噪自编码层构造。每一个去噪自编码层均是独立训练。
        """
        super(StackedAutoEncoderModel, self).__init__()

        # self.embedding = nn.Embedding(input_size, 2)
        # self.ae1 = LDAutoEncoderLayer(input_size * 2, 512)
        self.ae1 = LDAutoEncoderLayer(input_size, 2048)
        self.ae2 = LDAutoEncoderLayer(2048, 1024)
        self.ae3 = LDAutoEncoderLayer(1024, 512)
        self.ae4 = LDAutoEncoderLayer(512, 256)
        self.ae5 = LDAutoEncoderLayer(256, output_size)

    def forward(self, x):
        # x = self.embedding(x.long()).view(x.size(0), -1)
        # a1, loss1 = self.ae1(x.float())
        a1, loss1 = self.ae1(x)
        a2, loss2 = self.ae2(a1)
        a3, loss3 = self.ae3(a2)
        a4, loss4 = self.ae4(a3)
        a5, loss5 = self.ae5(a4)

        if self.training:
            return a5, loss5
        else:
            return a5, self.reconstruct(a5)

    def reconstruct(self, x):
        a5_reconstruct = self.ae5.reconstruct(x)
        a4_reconstruct = self.ae4.reconstruct(a5_reconstruct)
        a3_reconstruct = self.ae3.reconstruct(a4_reconstruct)
        a2_reconstruct = self.ae2.reconstruct(a3_reconstruct)
        x_reconstruct = self.ae1.reconstruct(a2_reconstruct)
        return x_reconstruct


class ClassifierLayer(nn.Module):
    def __init__(self, input_size):
        super(ClassifierLayer, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
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
        return self.classifier(x)
