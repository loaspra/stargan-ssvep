"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import copy
import math

from munch import Munch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


"""
A hierarchical model that combines coarse-grained temporal dynamics with fine-grained temporal learning.
Args:
    d_model (int): The number of expected features in the input (required by the Transformer).
    nhead (int): The number of heads in the multiheadattention models (required by the Transformer).
    num_layers (int): The number of encoder layers in the Transformer.
    cnn_in_channels (int): Number of channels in the input image (required by the FineGrainedTemporalLearning module).
    cnn_out_channels (int): Number of channels produced by the convolution (required by the FineGrainedTemporalLearning module).
    cnn_kernel_size (int or tuple): Size of the convolving kernel (required by the FineGrainedTemporalLearning module).
Methods:
    forward(x):
        Forward pass through the hierarchical model.
        Args:
            x (Tensor): Input tensor of shape (sequence_length, batch_size, d_model).
        Returns:
            Tensor: Output tensor after fusion of coarse and fine-grained temporal features.
"""
class FineGrainedTemporalLearning(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(FineGrainedTemporalLearning, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.elu = nn.ELU()
        self.max_pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        x = self.conv1d(x)
        x = self.batch_norm(x)
        x = self.elu(x)
        x = self.max_pool(x)
        return x

class HierarchicalCoarseToFineTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, cnn_in_channels, cnn_out_channels, cnn_kernel_size):
        super(HierarchicalCoarseToFineTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers)
        self.ftl = FineGrainedTemporalLearning(cnn_in_channels, cnn_out_channels, cnn_kernel_size)

    def forward(self, x):
        # Coarse-grained temporal dynamics
        x_coarse = self.transformer(x)
        
        # Fine-grained temporal learning
        x_fine = self.ftl(x)
        
        # Fusion
        x_fused = x_coarse + x_fine
        return x_fused

class InformationPurificationUnit(nn.Module):
    def __init__(self):
        super(InformationPurificationUnit, self).__init__()

    def forward(self, x):
        # Logarithmic power operation
        x_power = torch.log1p(torch.mean(x**2, dim=-1))
        return x_power


"""
ResBLock: 
    1. Conv1d
    2. Conv1d
    3. residual
    4. downsample (average pooling)
    5. normalize (instance norm)

The residual block helps to avoid the vanishing gradient problem.

The flow is described on the 'forward' function. It sums the output of the shortcut function and the residual function.

The shorcut function applies a 1x1 convolution to the input to match the output dimension. 
And then an average pooling is applied to the input to downsample it.

Then, the residual funciton applies a normalization, an activation function, a convolution,
"""
class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv1d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv1d(dim_in, dim_out, 3, 1, 1)
        
        if self.normalize:
            self.norm1 = nn.InstanceNorm1d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm1d(dim_in, affine=True)

        if self.learned_sc:
            self.conv1x1 = nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False)
        
        # CBAM  (Channel Attention Module)
        self.cbam = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim_out, dim_out // 16, 1, padding=0, bias=True),
            nn.ReLU(),
            nn.Conv1d(dim_out // 16, dim_out, 1, padding=0, bias=True),
            nn.Sigmoid()
        )


    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool1d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize: 
            x = self.norm1(x)                     
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool1d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    # With CBAM
    # def _residual(self, x):
    #     if self.normalize:
    #         x = self.norm1(x)
    #     x = self.actv(x)
    #     x = self.conv1(x)
    #     if self.downsample:
    #         x = F.avg_pool1d(x, 2)
    #     if self.normalize:
    #         x = self.norm2(x)
    #     x = self.actv(x)
    #     x = self.conv2(x)
    #     x = x * self.cbam(x)
    #     return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance
    

"""
AdaIN function:
    1. InstanceNorm2d
    2. Linear
    3. view
    4. chunk
    5. normalize
    6. add
    7. multiply

The AdaIN function is used to normalize the input to the AdaINResBlk.

Ada stands for Adaptive and IN stands for Instance Normalization.
"""

class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta
    

"""
AdaINResBlk:
    1. AdaIN
    2. activation function
    3. Conv1d
    4. Conv1d
    5. AdaIN
    6. activation function
    7. Conv1d
    8. shortcut
    9. residual

The AdaINResBlk is a residual block that uses the AdaIN function to normalize the input.

The flow is described on the 'forward' function. It sums the output of the shortcut function and the residual function.
However, the output is divided by the square root of 2 in this case.
"""
class AdainResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, style_dim=64, w_hpf=0,
                 actv=nn.LeakyReLU(0.2), upsample=False):
        super().__init__()
        self.w_hpf = w_hpf
        self.actv = actv
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out, style_dim)

    def _build_weights(self, dim_in, dim_out, style_dim=64):
        self.conv1 = nn.Conv1d(dim_in, dim_out, 3, 1, 1)
        self.conv2 = nn.Conv1d(dim_out, dim_out, 3, 1, 1)
        self.norm1 = AdaIN(style_dim, dim_in)
        self.norm2 = AdaIN(style_dim, dim_out)
        if self.learned_sc:
            self.conv1x1 = nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        return x

    def _residual(self, x, s):
        x = self.norm1(x, s)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        x = self.norm2(x, s)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x, s):
        out = self._residual(x, s)
        if self.w_hpf == 0:
            out = (out + self._shortcut(x)) / math.sqrt(2)
        return out


class HighPass(nn.Module):
    def __init__(self, w_hpf, device):
        super(HighPass, self).__init__()
        self.register_buffer('filter',
                             torch.tensor([[-1, -1, -1],
                                           [-1, 8., -1],
                                           [-1, -1, -1]]) / w_hpf)

    def forward(self, x):
        filter = self.filter.unsqueeze(0).unsqueeze(1).repeat(x.size(1), 1, 1, 1)
        return F.Conv1d(x, filter, padding=1, groups=x.size(1))


class Generator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, max_conv_dim=512, w_hpf=1, new_k_size=3):
        super().__init__()
        dim_in = 2 ** 14 // img_size
        self.img_size = img_size
        self.from_rgb = nn.Conv1d(3, dim_in, 3, 1, 1) # used
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        self.to_rgb = nn.Sequential(
            nn.InstanceNorm1d(dim_in, affine=True),
            nn.LeakyReLU(0.2),
            nn.Conv1d(dim_in, 3, 1, 1, 0))
        
        # Addition
        self.fine_grained_temporal = FineGrainedTemporalLearning(img_size, dim_in, new_k_size)
        
        # down/up-sampling blocks
        repeat_num = int(np.log2(img_size)) - 4
        if w_hpf > 0:
            repeat_num += 1
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(
                ResBlk(dim_in, dim_out, normalize=True, downsample=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_in, style_dim,
                               w_hpf=w_hpf, upsample=True))  # stack-like
            dim_in = dim_out

        # bottleneck blocks
        for _ in range(2):
            self.encode.append(
                ResBlk(dim_out, dim_out, normalize=True))
            self.decode.insert(
                0, AdainResBlk(dim_out, dim_out, style_dim, w_hpf=w_hpf))

        if w_hpf > 0:
            device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
            self.hpf = HighPass(w_hpf, device)

    def forward(self, x, s, masks=None):
        x = self.from_rgb(x)
        # x = self.fine_grained_temporal(x)
        cache = {}
        for block in self.encode:
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                cache[x.size(2)] = x
            x = block(x)

        for block in self.decode:
            x = block(x, s)
            if (masks is not None) and (x.size(2) in [32, 64, 128]):
                mask = masks[0] if x.size(2) in [32] else masks[1]
                mask = F.interpolate(mask, size=x.size(2), mode='bilinear')
                x = x + self.hpf(mask * cache[x.size(2)])

        x = self.to_rgb(x)
        return x     


class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super().__init__()
        layers = []
        layers += [nn.Linear(latent_dim, 512)]
        layers += [nn.ReLU()]
        for _ in range(3):
            layers += [nn.Linear(512, 512)]
            layers += [nn.ReLU()]
        self.shared = nn.Sequential(*layers)

        self.unshared = nn.ModuleList()
        for _ in range(num_domains):
            self.unshared += [nn.Sequential(nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, 512),
                                            nn.ReLU(),
                                            nn.Linear(512, style_dim))]

    def forward(self, z, y):
        h = self.shared(z)
        out = []
        for layer in self.unshared:
            out += [layer(h)]
        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=2, max_conv_dim=512):
        super().__init__()
        
        dim_in = 2**14 // img_size # why 2**14?
        blocks = []
        blocks += [nn.Conv1d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv1d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)
        

        self.unshared = nn.ModuleList()
        
        for _ in range(num_domains):
            self.unshared += [nn.Linear(dim_out, style_dim)]
    

    def forward(self, x, y):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        # output shape: (batch, dim_out) (8, 764928)
        out = []
        for layer in self.unshared:
            out += [layer(h)]

        out = torch.stack(out, dim=1)  # (batch, num_domains, style_dim)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        s = out[idx, y]  # (batch, style_dim)
        return s


class Discriminator(nn.Module):
    def __init__(self, img_size=256, num_domains=2, max_conv_dim=512):
        super().__init__()
        dim_in = 2**14 // img_size
        blocks = []
        blocks += [nn.Conv1d(3, dim_in, 3, 1, 1)]

        repeat_num = int(np.log2(img_size)) - 2
        for _ in range(repeat_num):
            dim_out = min(dim_in*2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample=True)]
            dim_in = dim_out

        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv1d(dim_out, dim_out, 4, 1, 0)]
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [nn.Conv1d(dim_out, num_domains, 1, 1, 0)]
        self.main = nn.Sequential(*blocks)

    def forward(self, x, y):
        out = self.main(x)
        out = out.view(out.size(0), -1)  # (batch, num_domains)
        idx = torch.LongTensor(range(y.size(0))).to(y.device)
        out = out[idx, y]  # (batch)
        return out


def build_model(args):
    generator = nn.DataParallel(Generator(args.img_size, args.style_dim, w_hpf=args.w_hpf))
    mapping_network = nn.DataParallel(MappingNetwork(args.latent_dim, args.style_dim, args.num_domains))
    style_encoder = nn.DataParallel(StyleEncoder(args.img_size, args.style_dim, args.num_domains))
    discriminator = nn.DataParallel(Discriminator(args.img_size, args.num_domains))
    generator_ema = copy.deepcopy(generator)
    mapping_network_ema = copy.deepcopy(mapping_network)
    style_encoder_ema = copy.deepcopy(style_encoder)

    nets = Munch(generator=generator,
                 mapping_network=mapping_network,
                 style_encoder=style_encoder,
                 discriminator=discriminator)
    nets_ema = Munch(generator=generator_ema,
                     mapping_network=mapping_network_ema,
                     style_encoder=style_encoder_ema)

    return nets, nets_ema
