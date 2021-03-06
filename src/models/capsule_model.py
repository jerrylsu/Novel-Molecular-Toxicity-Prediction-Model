#
# Dynamic Routing Between Capsules
# https://arxiv.org/pdf/1710.09829.pdf
#
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from src.trainer_sdae_model import MODEL_DIR

class CapsuleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CapsuleConvLayer, self).__init__()

        self.conv0 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=40000,  # fixme constant
                               stride=1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu(x)
        return x


class ConvUnit(nn.Module):
    def __init__(self, in_channels):
        super(ConvUnit, self).__init__()

        self.conv0 = nn.Conv1d(in_channels=in_channels,
                               out_channels=8,  # fixme constant
                               kernel_size=8,   # fixme constant
                               stride=2,        # fixme constant
                               bias=True)

    def forward(self, x):
        x = self.conv0(x)
        return x


class CapsuleLayer(nn.Module):
    def __init__(self, in_units, in_channels, num_units, unit_size, use_routing):
        super(CapsuleLayer, self).__init__()

        self.in_units = in_units
        self.in_channels = in_channels
        self.num_units = num_units
        self.use_routing = use_routing

        if self.use_routing:
            # In the paper, the deeper capsule layer(s) with capsule inputs (DigitCaps) use a special routing algorithm
            # that uses this weight matrix.
            self.W = nn.Parameter(torch.randn(1, in_channels, num_units, unit_size, in_units))  # in_caps, out_caps, out_len, in_len
        else:
            # The first convolutional capsule layer (PrimaryCapsules in the paper) does not perform routing.
            # Instead, it is composed of several convolutional units, each of which sees the full input.
            # It is implemented as a normal convolutional layer with a special nonlinearity (squash()).
            def create_conv_unit(unit_idx):
                unit = ConvUnit(in_channels=in_channels)
                self.add_module("unit_" + str(unit_idx), unit)
                return unit
            self.units = [create_conv_unit(i) for i in range(self.num_units)]

    @staticmethod
    def squash(s):
        # This is equation 1 from the paper.
        mag_sq = torch.sum(s**2, dim=2, keepdim=True)
        mag = torch.sqrt(mag_sq)
        s = (mag_sq / (1.0 + mag_sq)) * (s / mag)
        return s

    def forward(self, x):
        if self.use_routing:
            return self.routing(x)
        else:
            return self.no_routing(x)

    def no_routing(self, x):
        # Get output for each unit.
        # Each will be (batch, channels, len).
        u = [self.units[i](x) for i in range(self.num_units)]

        # Stack all unit outputs (batch, unit, channels, len).
        u = torch.stack(u, dim=1)

        # Flatten to (batch, unit, output).
        u = u.view(x.size(0), self.num_units, -1)

        # Return squashed outputs.
        return CapsuleLayer.squash(u)

    def routing(self, x):
        batch_size = x.size(0)

        # (batch, in_units, features) -> (batch, features, in_units)
        x = x.transpose(1, 2)

        # (batch, features, in_units) -> (batch, features, num_units, in_units, 1)
        x = torch.stack([x] * self.num_units, dim=2).unsqueeze(4)

        # (batch, features, in_units, unit_size, num_units)
        W = torch.cat([self.W] * batch_size, dim=0)

        # Transform inputs by weight matrix.
        # (batch_size, features, num_units, unit_size, 1)
        u_hat = torch.matmul(W, x)

        # Initialize routing logits to zero.
        b_ij = Variable(torch.zeros(1, self.in_channels, self.num_units, 1)).to(x.device)  # in_caps out_caps

        # Iterative routing.
        num_iterations = 3
        for iteration in range(num_iterations):
            # Convert routing logits to softmax.
            # (batch, features, num_units, 1, 1)
            c_ij = F.softmax(b_ij, dim=1)   # dim = 1 by jerry. default dim=None
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            # Apply routing (c_ij) to weighted inputs (u_hat).
            # (batch_size, 1, num_units, unit_size, 1)
            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            # (batch_size, 1, num_units, unit_size, 1)
            v_j = CapsuleLayer.squash(s_j)

            # (batch_size, features, num_units, unit_size, 1)
            v_j1 = torch.cat([v_j] * self.in_channels, dim=1)

            # (1, features, num_units, 1)
            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)

            # Update b_ij (routing)
            b_ij = b_ij + u_vj1

        return v_j.squeeze(1)


class CapsuleModel(nn.Module):
    def __init__(self,
                 conv_inputs,
                 conv_outputs,
                 num_primary_units,
                 primary_unit_size,
                 num_output_units,
                 output_unit_size):
        super(CapsuleModel, self).__init__()

        self.fc = torch.nn.Linear(227989, 512)
        self.relu = torch.nn.LeakyReLU()
        # self.sdae_model = torch.load(os.path.join(MODEL_DIR, "sdae1024-512-256-128_model-p3-c3-f5.pt")).eval()

        self.conv1 = CapsuleConvLayer(in_channels=conv_inputs,
                                      out_channels=conv_outputs)

        self.primary = CapsuleLayer(in_units=0,
                                    in_channels=conv_outputs,     # 1
                                    num_units=num_primary_units,  # 16
                                    unit_size=primary_unit_size,  # 16*253
                                    use_routing=False)

        self.digits = CapsuleLayer(in_units=num_primary_units,    # 16
                                   in_channels=primary_unit_size, # 16*253
                                   num_units=num_output_units,    # 2
                                   unit_size=output_unit_size,    # 2
                                   use_routing=True)
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(64, 2),
        )

        reconstruction_size = 512
        self.reconstruct0 = nn.Linear(4, 32)
        self.reconstruct1 = nn.Linear(32, 64)
        self.reconstruct2 = nn.Linear(64, reconstruction_size)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        # sdae_encoded = self.sdae_model.encoder(x).unsqueeze(1)
        # sdae_encoded = self.sdae_model.encoder[0](x).unsqueeze(1)   # auto-encoder layer0
        x = self.fc(x) # auto-encoder layer0
        sdae_encoded = self.relu(x).unsqueeze(1)
        # x = self.conv1(x)
        x = self.primary(sdae_encoded)
        x = self.digits(x).squeeze(-1)
        x = x.view(x.shape[0], -1)
        x = self.fc2(x)
        return x, sdae_encoded

    def criterion1(self, input_origin, predict, target, size_average=True):
        return self.margin_loss(predict, target, size_average) + self.reconstruction_loss(input_origin, predict, size_average)

    def margin_loss(self, predict, target, size_average=True):
        batch_size = predict.size(0)

        # ||vc|| from the paper.
        v_mag = torch.sqrt((predict**2).sum(dim=2, keepdim=True))

        # Calculate left and right max() terms from equation 4 in the paper.
        zero = Variable(torch.zeros(1)).to(predict.device)
        m_plus = 0.9
        m_minus = 0.1
        max_l = torch.max(m_plus - v_mag, zero).view(batch_size, -1)**2
        max_r = torch.max(v_mag - m_minus, zero).view(batch_size, -1)**2

        # This is equation 4 from the paper.
        loss_lambda = 0.5
        T_c = target.unsqueeze(1)
        L_c = T_c * max_l + loss_lambda * (1.0 - T_c) * max_r
        L_c = L_c.sum(dim=1)

        if size_average:
            L_c = L_c.mean()

        return L_c

    def reconstruction_loss(self, input_origin, predict, size_average=True):
        input_origin = input_origin.squeeze(1).view(-1)
        # Get the lengths of capsule outputs.
        v_mag = torch.sqrt((predict**2).sum(dim=2))

        # Get index of longest capsule output.
        _, v_max_index = v_mag.max(dim=1)
        v_max_index = v_max_index.data

        # Use just the winning capsule's representation (and zeros for other capsules) to reconstruct input image.
        batch_size = predict.size(0)
        all_masked = [None] * batch_size
        for batch_idx in range(batch_size):
            # Get one sample from the batch.
            input_batch = predict[batch_idx]

            # Copy only the maximum capsule index from this batch sample.
            # This masks out (leaves as zero) the other capsules in this sample.
            batch_masked = Variable(torch.zeros(input_batch.size())).to(predict.device)
            batch_masked[v_max_index[batch_idx]] = input_batch[v_max_index[batch_idx]]
            all_masked[batch_idx] = batch_masked

        # Stack masked capsules over the batch dimension.
        masked = torch.stack(all_masked, dim=0)

        # Reconstruct input image.
        masked = masked.view(predict.size(0), -1)
        output = self.relu(self.reconstruct0(masked))
        output = self.relu(self.reconstruct1(output))
        output = self.sigmoid(self.reconstruct2(output))
        output = output.view(-1)

        # The reconstruction loss is the sum squared difference between the input image and reconstructed image.
        # Multiplied by a small number so it doesn't dominate the margin (class) loss.
        error = (output - input_origin).view(output.size(0), -1)
        error = error**2
        error = torch.sum(error, dim=1) * 0.0005

        # Average over batch
        if size_average:
            error = error.mean()

        return error
