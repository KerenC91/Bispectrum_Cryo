import torch
from torch import nn
import numpy as np

class ResnetBlock(nn.Module):
    """Residual Block
    Args:
        in_channels (int): number of channels in input data
        out_channels (int): number of channels in output 
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, one_d=False):
        super(ResnetBlock, self).__init__()
        self.build_conv_block(in_channels, out_channels, one_d, kernel_size=kernel_size)

    def build_conv_block(self, in_channels, out_channels, one_d, kernel_size=3):
        padding = (kernel_size -1)//2
        if not one_d:
            conv = nn.Conv2d
            norm = nn.BatchNorm2d
        else:
            conv = nn.Conv1d
            norm = nn.BatchNorm1d

        self.conv1 = nn.Sequential(
            conv(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            norm(out_channels),
            nn.ELU()
        )
        self.conv2 = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            norm(out_channels),
        )
        if in_channels != out_channels:
            self.down = nn.Sequential(
                conv(in_channels, out_channels, kernel_size=1, bias=False),
                norm(out_channels)
            )
        else:
            self.down = None
        
        self.act = nn.ELU()

    def forward(self, x):
        """
        Args:
            x (Tensor): B x C x T
        """
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.down is not None:
            residual = self.down(residual)
        return self.act(out + residual)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, one_d=False, dilation=1):
        super(ConvBlock, self).__init__()
        if not one_d:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
            self.bn = nn.BatchNorm1d(out_channels)

        self.act = nn.ELU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class MaxOut2D(nn.Module):
    """
    Pytorch implementation of MaxOut on channels for an input that is C x H x W.
    Reshape input from N x C x H x W --> N x H*W x C --> perform MaxPool1D on dim 2, i.e. channels --> reshape back to
    N x C//maxout_kernel x H x W.
    """
    def __init__(self, max_out):
        super(MaxOut2D, self).__init__()
        self.max_out = max_out
        self.max_pool = nn.MaxPool1d(max_out)

    def forward(self, x):
        batch_size = x.shape[0]
        channels = x.shape[1]
        height = x.shape[2]
        width = x.shape[3]
        # Reshape input from N x C x H x W --> N x H*W x C
        x_reshape = torch.permute(x, (0, 2, 3, 1)).view(batch_size, height * width, channels)
        # Pool along channel dims
        x_pooled = self.max_pool(x_reshape)
        # Reshape back to N x C//maxout_kernel x H x W.
        return torch.permute(x_pooled, (0, 2, 1)).view(batch_size, channels // self.max_out, height, width).contiguous()


class MaxOut1D(nn.Module):
    """
    Pytorch implementation of MaxOut on channels for an input that is C x H x W.
    Reshape input from N x C x H x W --> N x H*W x C --> perform MaxPool1D on dim 2, i.e. channels --> reshape back to
    N x C//maxout_kernel x H x W.
    """
    def __init__(self, kernel_size=1, stride=2):
        super(MaxOut1D, self).__init__()
        self.max_out = kernel_size
        self.max_pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        batch_size = x.shape[0]
        channels = x.shape[1]
        layer = x.shape[2]
        #print(f'channels={channels}')
        #print(f'self.max_out={self.max_out}')

        #print(x.shape)
        # Reshape input from B x C x L --> B x L x C
        x_reshape = torch.permute(x, (0, 2, 1))
        #print(x_reshape.shape)
        # Pool along channel dims
        x_pooled = self.max_pool(x_reshape)
        #print(x_pooled.shape)
        # Reshape back to B x C//maxout_kernel x L.
        return torch.permute(x_pooled, (0, 2, 1)).contiguous()

class MidLayer(nn.Module):
    """Applies 1D pooling/conv1d operator over input tensor.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        residuals (int, optional): number of residual blocks. Default=0
    """
    def __init__(self, in_channels, out_channels, residuals=0, b_maxout=False, dilation=1):
        super(MidLayer, self).__init__()
        # TODO: try umsampling with bilinear interpolation 
        #self.upsample = nn.Upsample(scale_factor=2, mode='linear')
        #consider switching that with Pool1d
        self.b_maxout = b_maxout
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, dilation=dilation) 
        self.maxout = MaxOut1D(kernel_size=1, stride=2)
        torch.nn.init.xavier_uniform_(self.conv.weight)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ELU()     
        # self.pool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        
        if residuals != 0:
            # resnet blocks
            layers = []
            for _ in range(residuals):
                layers.append(
                    ResnetBlock(out_channels, out_channels, one_d=True)
                    )
            self.res_blocks = nn.Sequential(*layers)
        else:
            self.res_blocks = None


    def forward(self, x):
        """
        Args:
            x (Tensor): B x in_channels x T
        
        Returns:
            Tensor of shape (B, out_channels, T x 2)
        """
        # pool network
        B, C, T = x.shape
        # upsample
        # x = x.unsqueeze(dim=3)
        # x = F.upsample(x, size=(T*2, 1), mode='bilinear').squeeze(3)
        #x = self.upsample(x)
        # x = self.pad(x)
        
        if self.b_maxout == True:
            x = self.maxout(x)
        else:
            x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        #x = self.pool(x)

        # pass through resnet blocks to improve internal representations
        # of data
        if self.res_blocks != None:
            x = self.res_blocks(x)
        return x
    