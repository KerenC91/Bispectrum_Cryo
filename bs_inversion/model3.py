import torch
from torch import nn
import numpy as np
from model_utils import ResnetBlock, ConvBlock, MidLayer


class HeadBS3(nn.Module):
    """HeadBS3 module - same as 2, without mid layer

    Args:
        channels (list): list of #channels in each upsampling layer
        pre_residuals (int, optional): number of residual blocks before upsampling. Default: 64
        down_conv_channels (list): list of #channels in each down_conv blocks
        up_residuals (int, optional): number of residual blocks in each upsampling module. Default: 0
    """
    def __init__(self, device, input_len, channels,
          pre_residuals,
          pre_conv_channels,
          up_residuals,
          b_maxout,
          post_residuals,
          pow_2_channels,
          reduce_height,
          last_ch,
          bs_channels=2
          ):
        super(HeadBS3, self).__init__()
        self.device = device
        # Initialize learnable output factor
        self.f = torch.nn.Parameter(torch.ones(1))

        # Create pre_conv layer
        self.pre_conv = self._set_pre_conv_layers(bs_channels, 
                                                  pre_conv_channels, 
                                                  pre_residuals)

        self.dilated_conv = self._set_dilated_conv_layers(in_nc=pre_conv_channels[-1], 
                                                          nc=pre_conv_channels[-1], 
                                                          out_nc=pre_conv_channels[-1])       

        self.reduce_height = self._set_reduce_height(pre_conv_channels[-1], 
                                                     reduce_height)
        
        # Create middle layer           
        # self.mid, c2 = self._set_mid_layers(pre_conv_channels[-1], pow_2_channels,
        #                      up_residuals, b_maxout)
        
        # Create post layer - only residuals, count set by input parameter
        self.post_conv = self._set_post_conv_layers(pre_conv_channels[-1], post_residuals)

    def _set_reduce_height(self, last_pre_conv_ch, reduce_height):
        c1 = last_pre_conv_ch
        # Create pre middle layer - reduce height
        reduce_height_layers = []
        cnt, k, s = reduce_height
        for _ in range(cnt):
            reduce_height_layers.append(nn.Conv2d(in_channels=c1, out_channels=c1, 
                kernel_size=(k, 1), stride=(s, 1)))
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
    
    def _set_dilated_conv_layers(self, in_nc, nc, out_nc):
        dilated_convs = []
        
        dilated_convs.append(nn.Conv2d(in_channels=in_nc, out_channels=nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        dilated_convs.append(nn.ReLU(inplace=True))
        dilated_convs.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        dilated_convs.append(nn.ReLU(inplace=True))
        dilated_convs.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        dilated_convs.append(nn.ReLU(inplace=True))
        dilated_convs.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=4, dilation=4, bias=True))
        dilated_convs.append(nn.ReLU(inplace=True))
        dilated_convs.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=3, dilation=3, bias=True))
        dilated_convs.append(nn.ReLU(inplace=True))
        dilated_convs.append(nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=2, dilation=2, bias=True))
        dilated_convs.append(nn.ReLU(inplace=True))
        dilated_convs.append(nn.Conv2d(in_channels=nc, out_channels=out_nc, kernel_size=3, stride=1, padding=1, dilation=1, bias=True))
        return nn.Sequential(*dilated_convs)


    def _set_mid_layers(self, last_pre_conv_ch, pow_2_channels,
                        up_residuals, b_maxout):
        mid_layers = []
        c1 = last_pre_conv_ch
        # set input channel
        if pow_2_channels:
            # create channels ranging from c1 to next smaller power of 2, keep until getting 8
            # additional layer for every channel
            get_closest_pow2_d = lambda x: np.power(2, int(np.log(x) / np.log(2)))
            c2 = get_closest_pow2_d(c1)#512, 256, 128, 64, 32, 16, 8
            mid_layers.append(ConvBlock(c1, c2, kernel_size=3, padding=1, one_d=True))
            #print(f'ConvBlock {ch0}, {ch1}')

            while c2/2. > 0:
                in_channels = c2
                out_channels = int(c2/2.)
                c2 = int(c2/2.)
                layer = MidLayer(in_channels, out_channels, residuals=up_residuals, \
				 b_maxout=b_maxout)
                mid_layers.append(layer)
                #print(f'mid_layers {in_channels}, {out_channels}')
                if int(c2) == 8:
                    break
        else:
            # create layer for c1-->c2
            c2 = 8
            mid_layers.append(ConvBlock(c1, c2, kernel_size=3, padding=1, one_d=True))
           
        return nn.Sequential(*mid_layers), c2
 

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
        #print(x.shape)
        #x = x.unsqueeze(0) # reshape to [B x 2 x 100 x 100]
        #print(x.shape)

        x = self.pre_conv(x)
        #x = self.dilated_conv(x)
        #print(x.shape)
        # for BXCXHXW reduce dimension to BXCX1XW
        x = self.reduce_height(x)
        #print(x.shape)
        x = x.squeeze(2)
        #print(x.shape)

        #x = self.mid(x)
        #print(x.shape)
        x *= self.f
        
        x2 = self.post_conv(x)
        #print(x2.shape)
        x2 *= self.f
        
        return x2# pre, post

