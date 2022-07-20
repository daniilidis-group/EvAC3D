import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils import sample_circle
from .layers import CResnetBlockConv1d, CBatchNorm1d, ResnetBlockFC

class ContourNet(nn.Module):
    def __init__(self, input_dim=20, 
                            batch_size=8, 
                            bottleneck_size=128, 
                            point_input_dim=2, 
                            output_dim=2, 
                            decoder_layers=5,
                            point_encoder=False,
                            big_encoder=False,
                            hidden_size=256,
                            device="cuda"):
        super(ContourNet, self).__init__()

        # build feature extractor
        self.bottleneck_size = bottleneck_size
        self.point_input_dim = point_input_dim
        self.output_dim = output_dim

        # encoder
        self.encoder = models.resnet34() if big_encoder \
                                    else models.resnet18(pretrained=False)

        self.encoder.conv1 = nn.Conv2d(input_dim, 64, kernel_size=(7, 7),
                stride=(2, 2), padding=(3, 3), bias=False)
        self.encoder.fc = nn.Linear(in_features=512, out_features=self.bottleneck_size)

        self.decoder = DecoderCBatchNorm2(dim=point_input_dim, 
                                            c_dim=self.bottleneck_size,
                                            output_dim=self.output_dim,
                                            hidden_size=hidden_size,
                                            n_blocks=decoder_layers)

    def forward(self, x, samples):
        points = samples
        # extract features
        event_features = self.encoder(x)
        # append the features to all the points
        output_points = self.decoder(points, event_features.unsqueeze(-1))
        # batched points [batch_size, 2, N]
        return output_points


# Taken from Occupancy networks: https://github.com/autonomousvision/occupancy_networks
class DecoderCBatchNorm2(nn.Module):
    ''' Decoder with CBN class 2.
    It differs from the previous one in that the number of blocks can be
    chosen.
    Args:
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        n_blocks (int): number of ResNet blocks
    '''

    def __init__(self, dim=3, c_dim=128, output_dim=1,
                 hidden_size=256, n_blocks=5):
        super().__init__()
        self.conv_p = nn.Conv1d(dim, hidden_size, 1)
        self.blocks = nn.ModuleList([
            CResnetBlockConv1d(c_dim, hidden_size) for i in range(n_blocks)
        ])

        self.bn = CBatchNorm1d(c_dim, hidden_size)
        self.conv_out = nn.Conv1d(hidden_size, output_dim, 1)
        self.actvn = nn.ReLU()

    def forward(self, x, latent):
        p = x
        c = latent
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()
        net = self.conv_p(p)
        for block in self.blocks:
            net = block(net, c)
        out = self.conv_out(self.actvn(self.bn(net, c)))
        return out

class ResNetLayers(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, input_dim, 1)
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.actvn = nn.ReLU()
        self.conv2 = nn.Conv1d(input_dim, input_dim, 1)
        self.bn2 = nn.BatchNorm1d(input_dim)

    def forward(self, x):
        x_conv = self.conv1(x)
        x_conv = self.bn1(x_conv)
        x_conv = self.actvn(x_conv)
        x_conv = self.conv2(x_conv)
        x_conv = self.bn2(x_conv)
        return x + x_conv

class NaiveDecoder(nn.Module):

    def __init__(self, dim=3, c_dim=128, output_dim=1,
                 hidden_size=256, n_blocks=5):
        super().__init__()
        self.conv_p = nn.Conv1d(dim+c_dim, hidden_size, 1)
        self.map_blocks = nn.ModuleList([
            nn.Conv1d(hidden_size+c_dim, hidden_size, 1) for i in range(n_blocks)
        ])

        self.res_blocks = nn.ModuleList([
            ResNetLayers(hidden_size) for i in range(n_blocks)
        ])

        self.conv_out = nn.Conv1d(hidden_size+c_dim, output_dim, 1)
        self.actvn = nn.ReLU()

    def forward(self, x, latent):
        # map to hidden_size
        x = x.transpose(1, 2)
        latent = latent.repeat(1, 1, x.shape[-1])

        net = torch.cat([x, latent], dim=1)
        net = self.conv_p(net)

        # append c to latent and then map
        for i in range(len(self.map_blocks)):
            net = torch.cat([net, latent], dim=1)
            net = self.map_blocks[i](net)
            net = self.actvn(net)
            net = self.res_blocks[i](net)
            net = self.actvn(net)

        # last layer
        net = torch.cat([net, latent], dim=1)
        out = self.conv_out(net)
        return out

class SimpleNaiveDecoder(nn.Module):

    def __init__(self, dim=3, c_dim=128, output_dim=1,
                 hidden_size=256, n_blocks=5):
        super().__init__()
        self.conv_p = nn.Conv1d(dim, hidden_size, 1)

        self.res_blocks = nn.ModuleList([
            ResNetLayers(hidden_size) for i in range(n_blocks)
        ])

        self.conv_out = nn.Conv1d(hidden_size+c_dim+dim, output_dim, 1)
        self.actvn = nn.ReLU()

    def forward(self, x, latent):
        # map to hidden_size
        x = x.transpose(1, 2)
        latent = latent.repeat(1, 1, x.shape[-1])
        net = self.conv_p(x)

        # append c to latent and then map
        for i in range(len(self.res_blocks)):
            net = self.res_blocks[i](net)
            net = self.actvn(net)

        # last layer
        net = torch.cat([net, latent, x], dim=1)
        out = self.conv_out(net)
        return out

def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class SimplePointnet(nn.Module):
    ''' PointNet-based encoder network.
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.fc_0 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.fc_0(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_3(self.actvn(net))

        # Recude to  B x F
        net = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(net))

        return c


class ResnetPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks.
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.block_0 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_1 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_2 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_3 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.block_4 = ResnetBlockFC(2*hidden_dim, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        batch_size, T, D = p.size()

        # output size: B x T X F
        net = self.fc_pos(p)
        net = self.block_0(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_1(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_2(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_3(net)
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.block_4(net)

        # Recude to  B x F
        net = self.pool(net, dim=1)

        c = self.fc_c(self.actvn(net))

        return c
