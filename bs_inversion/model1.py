import torch
from torch import nn
import numpy as np
from model_utils import ResnetBlock, ConvBlock, MidLayer
from hparams import hparams
    
class HeadBS1(nn.Module):
    """HeadBS1 module - Original one with reshape

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
        super(HeadBS1, self).__init__()

        self.device = device
        # Initialize learnable output factor
        self.f = torch.nn.Parameter(torch.ones(1))

        # Create pre_conv layer
        self.pre_conv = self._set_pre_conv_layers(bs_channels, 
                                                  pre_conv_channels, 
                                                  pre_residuals)
        
        
        # Create middle layer
           
        self.mid, ch1 = self._set_mid_layers(pre_conv_channels[-1], input_len, 
                                        pow_2_channels, up_residuals, b_maxout)
        
        # Create post layer - only residuals, count set by input parameter
        self.post_conv = self._set_post_conv_layers(ch1, post_residuals)
 
        
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

    def _set_mid_layers(self, last_pre_conv_ch, input_len, pow_2_channels,
                        up_residuals, b_maxout):
        mid_layers = []
        #print('start poooling')
        ch0 = last_pre_conv_ch * input_len # 8*100=800
        if pow_2_channels:
            # create channels ranging from c1 to next smaller power of 2, keep until getting 8
            # additional layer for every channel
            get_closest_pow2_d = lambda x: np.power(2, int(np.log(x) / np.log(2)))
            ch1 = get_closest_pow2_d(ch0)#512, 256, 128, 64, 32, 16, 8
            mid_layers.append(ConvBlock(ch0, ch1, kernel_size=3, padding=1, one_d=True))
            #print(f'ConvBlock {ch0}, {ch1}')

            while ch1/2. > 0:
                in_channels = ch1
                out_channels = int(ch1/2.)
                ch1 = int(ch1/2.)
                layer = MidLayer(in_channels, out_channels, residuals=up_residuals, \
                                 b_maxout=b_maxout, dilation=hparams.dilation_mid)
                mid_layers.append(layer)
                #print(f'mid_layers {in_channels}, {out_channels}')
                if int(ch1) == 8:
                    break
        else:
            # create layer for ch0-->ch1
            ch1 = 8
            mid_layers.append(ConvBlock(ch0, ch1, kernel_size=3, padding=1, one_d=True))
           
        return nn.Sequential(*mid_layers), ch1
 

    def _set_post_conv_layers(self, ch1, post_residuals):
        post_convs = []
        last_channels = int(ch1)
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
        #print('start Head')
        #print(x.shape)

        x = self.pre_conv(x)
        #print(x.shape)

        s1, _, _, s4 = x.shape

        x = x.reshape(s1, -1, s4)
        #print(x.shape)

        x = self.mid(x)
        #print(x.shape)
        x *= self.f
        
        x2 = self.post_conv(x)
        #print(x2.shape)
        x2 *= self.f
        #print('end Head')
        
        return x2# pre, post
        
       
class CNNBS(nn.Module):
    """CNNBS  - CNN for BS inversion

    Args:
        n_heads (int): Number of heads
        layer_channels (list): list of #channels of each layer
    """
    def __init__(self, device, input_len, n_heads, 
         channels,
         b_maxout,
         pre_conv_channels,
         pre_residuals, 
         up_residuals,
         post_residuals,
         pow_2_channels,
         reduce_height, 
         head_class,
         linear_ch,
         activation):

        super(CNNBS, self).__init__()
        self.device = device
        self.n_heads = n_heads
        self.linear = nn.Linear(linear_ch, 1)
        self.Head = head_class
        self.act_fn = activation
            
        self.heads = nn.ModuleList([self.Head(device, input_len, channels, b_maxout=b_maxout,
                pre_conv_channels=pre_conv_channels, 
                pre_residuals=pre_residuals, up_residuals=up_residuals,
                post_residuals=post_residuals, pow_2_channels=pow_2_channels, 
                reduce_height=reduce_height, last_ch=linear_ch)
                                    for _ in range(n_heads)])

    def forward(self, x):

        # Pass over Heads in "parallel"
        if self.n_heads > 1:
            post_list = []
            #print('Pass over Heads in parallel- start')

            for head in self.heads:
                post = head(x)

                post_list.append(post)
                #print(head.f)
            #print('Pass over Heads in parallel- end')
            # Sum Heads outputs
            post = torch.sum(torch.stack(post_list), dim=0)
        else:
            # Pass over Head
            post = self.heads[0](x)
            
        
        # Post output - actually used
        rs1 = self.linear(post.transpose(1, 2))
        rs1 = self.act_fn(rs1).transpose(2, 1)
        
        return rs1 
