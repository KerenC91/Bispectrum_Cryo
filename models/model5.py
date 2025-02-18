import torch
from torch import nn
import numpy as np
from models.model_utils import ResnetBlock, ConvBlock, MidLayer
from models.network_swinir import RSTB, PatchEmbed, PatchUnEmbed
import pdb

class HeadBS5(nn.Module):
    """HeadBS5 module - no transformers

    Args:
        channels (list): list of #channels in each upsampling layer
        pre_residuals (int, optional): number of residual blocks before upsampling. Default: 64
        down_conv_channels (list): list of #channels in each down_conv blocks
        up_residuals (int, optional): number of residual blocks in each upsampling module. Default: 0
    """
    def __init__(self, device, input_len, signals_count, channels,
            pre_residuals,
            pre_conv_channels,
            up_residuals,
            b_maxout,
            post_residuals,
            pow_2_channels,
            reduce_height,
            last_ch,
            activation,
             bs_channels=2
          ):
        super(HeadBS5, self).__init__()
        
        self.device = device
        # update later to be from outside
        self.linear = nn.Linear(last_ch, signals_count)
        self.act_fn = activation
        self.embed_dim = last_ch
  
        # Initialize learnable output factor
        # self.f = torch.nn.Parameter(torch.ones(1))

        # Create pre_conv layer
        self.pre_conv = self._set_pre_conv_layers(bs_channels, 
                                                  pre_conv_channels, 
                                                  pre_residuals)
      
        self.reduce_height = self._set_reduce_height(pre_conv_channels[-1], 
                                                     reduce_height)
                
        # Create post layer - only residuals, count set by input parameter
        self.post_conv = self._set_post_conv_layers(pre_conv_channels[-1], post_residuals)
           
    def _set_reduce_height(self, last_pre_conv_ch, reduce_height):
        c1 = last_pre_conv_ch
        # Create pre middle layer - reduce height
        reduce_height_layers = []
        cnt, k, s, add_conv_2= reduce_height

        for _ in range(cnt):
            reduce_height_layers.append(nn.Conv2d(in_channels=c1, out_channels=c1, 
                kernel_size=(k, 1), stride=(s, 1)))
        if add_conv_2:
            reduce_height_layers.append(nn.Conv2d(in_channels=c1, out_channels=c1, 
                kernel_size=(2, 1), stride=(2, 1)))
        return nn.Sequential(*reduce_height_layers)
        
    def _set_pre_conv_layers(self, bs_channels, pre_conv_channels, pre_residuals):
        pre_convs = []
        
        c0 = pre_conv_channels[0]
        
        # add first pre_conv layer: bs_channels-->pre_conv_channels[0]
        # add residuals after layer
        pre_convs.append(ConvBlock(bs_channels, c0, kernel_size=3, padding=1))
        # add resnets - no change in channels dim
        for _ in range(pre_residuals):
            pre_convs.append(ResnetBlock(c0, c0))
        
        # add additional pre_convs layer: pre_conv_channels[i]-->pre_conv_channels[i + 1]
        # add residuals after each layer
        # pre_conv_channels set by input parameter
        for i in range(len(pre_conv_channels) -1):
            in_c = pre_conv_channels[i]
            out_c = pre_conv_channels[i + 1]
            pre_convs.append(ResnetBlock(in_c, out_c))
            for _ in range(pre_residuals):
                pre_convs.append(ResnetBlock(out_c, out_c))
        return nn.Sequential(*pre_convs)

    def _set_post_conv_layers(self, c2, post_residuals):
        post_convs = []
        last_channels = int(c2)
        for i in range(post_residuals):
            post_convs.append(ResnetBlock(last_channels, last_channels, one_d=True, kernel_size=5))
        return nn.Sequential(*post_convs)
    
    def forward(self, x):
        """
        forward pass
        Args:
            x (Tensor): B x C x T # 100X100X2

        Returns:
            Tensor: B x C x (2^#channels * T) # 100X100X(2^#channels * 2)
        """
        x = self.pre_conv(x)
        # for BXCXHXW reduce dimension to BXCX1XW
        x = self.reduce_height(x)
        x = x.squeeze(2)
        # x *= self.f
        x = self.post_conv(x)  
        # x *= self.f
        x = self.linear(x.transpose(1, 2))
        x = self.act_fn(x).transpose(2, 1)

        return x

