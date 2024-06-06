#pylint: disable=no-member
import torch
import math
import torch.nn as nn
from fairseq.modules import ConvTBC, LinearizedConvolution, VGGBlock
from torch import Tensor

"""
    codes of different convolution blocks for some inital analyse
    VGGBlocks, 2D, with  ((32, 3, 2, 2, False),) * 2 ,[(64, 3, 2, 2, True), (128, 3, 2, 2, True)]
    Simp2D, only two layers of 2d conv, no layernorm(batchnorm),downsample directly
    Simp2D_v2, two layers of 2dconv, no downsample on frequency dim
    Simple1D, two layers of 1d conv, downsample directly 
    Resnet2D, similar to VGGBlocks, residual
    all implementations with init params (input_mel_bins, output_embed_dim),
    input is BxTxC, and output is TxBxC, output x and padding_mask

"""

def lengths_to_padding_mask(lengths:Tensor, x:Tensor):
    """
        generate padding mask from lengths,
        this may also be used as fake tokens to make PositionEmbedding happy. 
        for 1 is just padding_idx in fairseq Dictionary
        we also input coresponded feature to make sure mask length equal(maybe not exact for downsample tricks)
    """
    max_lengths, bsz = x.shape[:2]
   
    encoder_padding_mask = torch.arange(
        max_lengths
    ).to(  # a (T, ) tensor with [0, ..., T-1]
        x.device
    ).view(  # move to the right device
        1, max_lengths
    ).expand(  # reshape to (1, T)-shaped tensor
        bsz, -1
    ) >= lengths.view(  # expand to (B, T)-shaped tensor
        bsz, 1
    ).expand(
        -1, max_lengths
    )
    return encoder_padding_mask



VGG_CONFIG_SMALL=[(32, 3, 2, 2, True)] * 2
VGG_CONFIG_BASE= [(64, 3, 2, 2, True), (128, 3, 2, 2, True)]


class VGGEncoder(nn.Module):
    def __init__(
        self,
        num_mel_bins= 80,
        output_dim= 512,
        vgg_configs=VGG_CONFIG_SMALL,
    ):
        super().__init__()
        self.num_mel_bins= num_mel_bins
        self.output_dim= output_dim
        self.conv_layers= nn.ModuleList()
        self.pooling_kernel_sizes =[]
        self.in_channels = 1
        in_channels = self.in_channels
        input_feat_per_channel = num_mel_bins
        for _config in vgg_configs:
            (
                out_channels,
                conv_kernel_size,
                pooling_kernel_size,
                num_conv_layers,
                layer_norm,
            ) = _config
            self.conv_layers.append(
                VGGBlock(
                    in_channels,
                    out_channels,
                    conv_kernel_size,
                    pooling_kernel_size,
                    num_conv_layers,
                    input_dim = input_feat_per_channel,
                    layer_norm=layer_norm
                )
            )
            self.pooling_kernel_sizes.append(pooling_kernel_size)
            in_channels= out_channels
            input_feat_per_channel = self.conv_layers[-1].output_dim
        vgg_out_dim = self.conv_layers[-1].total_output_dim
        self.out_project= nn.Linear(vgg_out_dim, self.output_dim)
    
    def forward(self, fbank, fbk_lengths):
        bsz, seq_len, _ = fbank.shape
        x= fbank.view(bsz, seq_len, self.in_channels, self.num_mel_bins)
        x= x.transpose(1,2).contiguous()
        for layer in self.conv_layers:
            x= layer(x)
        input_lengths= fbk_lengths.clone()
        for s in self.pooling_kernel_sizes:
            input_lengths = (input_lengths.float()/s).ceil().long()
        #(B,C,T,fea)->(T,B,C*feature)
        bsz, _, out_seq_len, _ = x.shape
        x= x.permute(2,0,1,3).contiguous().view(out_seq_len, bsz, -1)
        x= self.out_project(x)
        padding_mask = lengths_to_padding_mask(input_lengths, x)
        return x, padding_mask


def VGG_Small(num_mel_bins, output_dim):
    return VGGEncoder( num_mel_bins, output_dim,VGG_CONFIG_SMALL,)

def VGG_Base(num_mel_bins, output_dim):
    return VGGEncoder(num_mel_bins, output_dim, VGG_CONFIG_BASE)


class Shallow2D(nn.Module):
    def __init__(
        self, 
        num_mel_bins,
        output_dim,
        conv_channels= (128,128)
    ):
        super().__init__()
        self.num_mel_bins= num_mel_bins
        self.output_dim= output_dim
        self.in_channels= input_channels= 1
        self.conv_layers=nn.ModuleList()
        self.pooling_kernel_sizes= []
        for out_channels in conv_channels:
            self.conv_layers.append(
                nn.Conv2d(
                    input_channels, out_channels,
                    (3,3),
                    stride=(2,1),
                    padding=(1,1)
                )
            )
            input_channels = out_channels
            self.conv_layers.append(nn.ReLU())
            self.pooling_kernel_sizes.append(2)
        conv_agg_dim = num_mel_bins*conv_channels[-1]
        self.out_proj= nn.Linear(conv_agg_dim, output_dim)
    
    def forward(self, fbank, fbk_lengths):
        bsz, seq_len, _ = fbank.shape
        x= fbank.view(bsz, seq_len, self.in_channels, self.num_mel_bins)
        x= x.transpose(1,2).contiguous()
        for layer in self.conv_layers:
            x= layer(x)
        input_lengths= fbk_lengths.clone()
        for s in self.pooling_kernel_sizes:
            input_lengths = (input_lengths.float()/s).ceil().long()
        #(B,C,T,fea)->(T,B,C*feature)
        bsz, _, out_seq_len, _ = x.shape
        x= x.permute(2,0,1,3).contiguous().view(out_seq_len, bsz, -1)
        x= self.out_proj(x)
        padding_mask = lengths_to_padding_mask(input_lengths, x)
        return x, padding_mask

def Shallow2d_Base(num_mel_bins, output_dim):
    return Shallow2D(num_mel_bins, output_dim, (64,64))


class Shallow1D(nn.Module):
    """
        1D convolution with LinearizedConvolution, maybe easy for online
    """
    def __init__(
        self, 
        num_mel_bins,
        output_dim,
        kernel_sizes= (5,5),
        mid_channels= 1024,
    ):
        super().__init__()
        self.num_mel_bins= num_mel_bins
        self.output_dim= output_dim
        self.n_layers= len(kernel_sizes)
        in_channels= num_mel_bins
        out_channels= output_dim
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                in_channels if i == 0 else mid_channels // 2,
                mid_channels if i < self.n_layers - 1 else out_channels * 2,
                k,
                stride=2,
                padding=k // 2,
            )
            for i, k in enumerate(kernel_sizes)
        )
    
    def get_out_seq_lens_tensor(self, in_seq_lens_tensor):
        out = in_seq_lens_tensor.clone()
        for _ in range(self.n_layers):
            out = ((out.float() - 1) / 2 + 1).floor().long()
        return out

    def forward(self, fbank, fbk_lengths):
        x = fbank.transpose(1, 2).contiguous()  # -> B x (C x D) x T
        for conv in self.conv_layers:
            x = conv(x)
            x = nn.functional.glu(x, dim=1)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # -> T x B x (C x D)
        olengths= self.get_out_seq_lens_tensor(fbk_lengths)
        padding_mask = lengths_to_padding_mask(olengths, x)
        return x, padding_mask


def Shallow1d_Base(num_mel_bins, output_dim):
    return Shallow1D(num_mel_bins, output_dim, (5,5), 1024)


import torchvision
RESNET_CONFIG_BASE= [(64, 2, 4), (128, 2, 4)]
RESNET_CONFIG_SMALL= [(64, 2, 2), (128, 2, 2)]


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,num_mel_bins, output_dim, block, res_config=RESNET_CONFIG_BASE):
        super(ResNet, self).__init__()
        self.num_mel_bins = num_mel_bins
        self.output_dim= output_dim
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layers= nn.ModuleList()
        self.pooling_kernel_sizes= []
        for block_cfg in res_config:
            (featmap, stride, nlayers) = block_cfg
            self.layers.append(self._make_layer(block, featmap, nlayers, stride=stride))
            self.pooling_kernel_sizes.append(stride)
        
        conv_agg_dim= self._infer_shape()
        self.out_proj= nn.Linear(conv_agg_dim, output_dim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _infer_shape(self):
        samp_bsz,samp_seqlen = 10, 23
        x= torch.randn(samp_bsz, 1, samp_seqlen, self.num_mel_bins)
        x = self.conv1(x)
        for layer in self.layers:
            x= layer(x)
        _, featmap, _, nchannels= x.shape
        return featmap*nchannels

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, fbank, fbk_lengths):
        x = fbank.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        for layer in self.layers:
            x= layer(x)
        input_lengths= fbk_lengths.clone()
        for s in self.pooling_kernel_sizes:
            input_lengths = (input_lengths.float()/s).ceil().long()
        #(B,C,T,fea)->(T,B,C*feature)
        bsz, _, out_seq_len, _ = x.shape
        x= x.permute(2,0,1,3).contiguous().view(out_seq_len, bsz, -1)
        x= self.out_proj(x)
        padding_mask = lengths_to_padding_mask(input_lengths, x)
        return x, padding_mask

def Resnet_Small(num_mel_bins, output_dim):
    return ResNet(num_mel_bins, output_dim, BasicBlock, RESNET_CONFIG_SMALL)

def Resnet_Base(num_mel_bins, output_dim):
    return ResNet(num_mel_bins, output_dim, BasicBlock, RESNET_CONFIG_BASE)

Audio_Convs_Mapping={
    "vgg_small":VGG_Small,
    "vgg_base": VGG_Small,
    "shallow2d_base":Shallow2d_Base,
    "shallow1d_base":Shallow1d_Base,
    "resnet_base":Resnet_Base,
    "resnet_small":Resnet_Small
}

def get_available_convs():
    return list(Audio_Convs_Mapping.keys())

def get_conv(conv_type="vgg_base"):
    if conv_type not in Audio_Convs_Mapping:
        raise ValueError(f"conv_type should be one of {get_available_convs()}")
    return Audio_Convs_Mapping[conv_type]



        


            
